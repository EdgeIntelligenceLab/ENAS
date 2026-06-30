"""
ENAS v2.1 — Main NAS class.

Three-stage hybrid search:
    1. Random sampling (broad exploration)
    2. Top-K accuracy-first selection
    3. Mutation refinement

See README.md for usage examples and full parameter documentation.
"""

import os
import re
import json
import datetime
import subprocess
from pathlib import Path
from multiprocessing import Process, Queue

import numpy as np
import tensorflow as tf

from enas.search_space import (
    SEARCH_SPACE,
    sample_random_architecture,
    architecture_to_key,
)
from enas.estimators import estimate_ram, estimate_flash, estimate_macc
from enas.blocks import build_model
from enas.scoring import compute_score
from enas.mutations import mutate_architecture
from enas.quantization import quantize_model


class ENAS:
    """ENAS v2.1 — Edge Neural Architecture Search."""

    def __init__(
        self,
        max_ram,
        max_flash,
        max_macc,
        path_to_training_set,
        val_split,
        cache=False,
        input_shape=(50, 50, 3),
        save_path='./',
        # ── Search configuration ──────────────────────────────────────
        n_random_stage=30,
        top_k_select=8,
        n_mutations_per_arch=5,
        proxy_epochs=3,
        proxy_data_fraction=0.20,
        parallel_workers=2,
        # ── Scoring weights ───────────────────────────────────────────
        accuracy_weight=0.80,
        efficiency_weight=0.15,
        headroom_weight=0.05,
        # ── Search space bounds ───────────────────────────────────────
        max_k=12,
        max_cells=5,
        # ── Optional seeding ──────────────────────────────────────────
        seed_k=None,
        seed_c=None,
    ):
        self.path_to_training_set = path_to_training_set
        self.num_classes          = len(next(os.walk(path_to_training_set))[1])
        self.val_split            = val_split
        self.input_shape          = input_shape
        self.max_ram              = max_ram
        self.max_flash            = max_flash
        self.max_macc             = max_macc
        self.cache                = cache
        self.save_path            = Path(save_path)
        self.n_random_stage       = n_random_stage
        self.top_k_select         = top_k_select
        self.n_mutations_per_arch = n_mutations_per_arch
        self.proxy_epochs         = proxy_epochs
        self.proxy_data_fraction  = proxy_data_fraction
        self.parallel_workers     = parallel_workers
        self.accuracy_weight      = accuracy_weight
        self.efficiency_weight    = efficiency_weight
        self.headroom_weight      = headroom_weight
        self.max_k                = max_k
        self.max_cells            = max_cells
        self.seed_k               = seed_k
        self.seed_c               = seed_c

        os.makedirs(self.save_path, exist_ok=True)
        self.path_to_resulting_model = (
            self.save_path / 'resulting_architecture.h5'
        )
        self.path_to_quantized_resulting_model = (
            self.save_path / 'resulting_architecture.tflite'
        )

        self._cache_path = self.save_path / 'enas_v2_1_cache.json'
        self._arch_cache = self._load_cache()

        self._k_range         = [k for k in SEARCH_SPACE["k"] if k <= max_k]
        self._num_cells_range = [c for c in SEARCH_SPACE["num_cells"]
                                 if c <= max_cells]

    # ── Cache I/O ──────────────────────────────────────────────────────────

    def _load_cache(self):
        if self._cache_path.exists():
            try:
                with open(self._cache_path, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_cache(self):
        with open(self._cache_path, 'w') as f:
            json.dump(self._arch_cache, f, indent=2)

    # ── Dataset loading ────────────────────────────────────────────────────

    def _color_mode(self):
        return 'rgb' if self.input_shape[2] == 3 else 'grayscale'

    def load_training_set(self, batch_size=1):
        cm = self._color_mode()
        kw = dict(
            directory=self.path_to_training_set,
            labels='inferred', label_mode='categorical',
            color_mode=cm, batch_size=batch_size,
            image_size=self.input_shape[0:2],
            shuffle=True, seed=11,
            validation_split=self.val_split,
        )
        train_ds = tf.keras.utils.image_dataset_from_directory(
            subset='training', **kw)
        val_ds   = tf.keras.utils.image_dataset_from_directory(
            subset='validation', **kw)
        pf = tf.data.AUTOTUNE
        if self.cache:
            return (train_ds.cache().prefetch(pf), val_ds.cache().prefetch(pf))
        return train_ds.prefetch(pf), val_ds.prefetch(pf)

    def _load_proxy_set(self, batch_size=32):
        cm = self._color_mode()
        kw = dict(
            directory=self.path_to_training_set,
            labels='inferred', label_mode='categorical',
            color_mode=cm, batch_size=batch_size,
            image_size=self.input_shape[0:2],
            shuffle=True, seed=42,
            validation_split=self.val_split,
        )
        train_ds = tf.keras.utils.image_dataset_from_directory(
            subset='training', **kw)
        val_ds   = tf.keras.utils.image_dataset_from_directory(
            subset='validation', **kw)
        n_train = max(1, int(
            tf.data.experimental.cardinality(train_ds).numpy()
            * self.proxy_data_fraction
        ))
        n_val = max(1, int(
            tf.data.experimental.cardinality(val_ds).numpy()
            * self.proxy_data_fraction
        ))
        pf = tf.data.AUTOTUNE
        return (train_ds.take(n_train).prefetch(pf),
                val_ds.take(n_val).prefetch(pf))

    def load_test_set(self, path_to_test_set, batch_size=1):
        cm = self._color_mode()
        test_ds = tf.keras.utils.image_dataset_from_directory(
            directory=path_to_test_set,
            labels='inferred', label_mode='categorical',
            color_mode=cm, batch_size=batch_size,
            image_size=self.input_shape[0:2]
        )
        pf = tf.data.AUTOTUNE
        if self.cache:
            return test_ds.cache().prefetch(pf)
        return test_ds.prefetch(pf)

    # ── Proxy evaluation ──────────────────────────────────────────────────

    def _proxy_evaluate_process(self, q, architecture):
        est_ram   = estimate_ram(self.input_shape, architecture)
        est_flash = estimate_flash(self.input_shape, architecture,
                                   self.num_classes)
        est_macc  = estimate_macc(self.input_shape, architecture)

        model, real_macc, cell_limited = build_model(
            self.input_shape, self.num_classes, architecture
        )

        infeasible = (
            cell_limited
            or est_ram   > self.max_ram
            or est_flash > self.max_flash
            or real_macc > self.max_macc
        )

        key = architecture_to_key(architecture)

        if infeasible:
            q.put({
                "key": key, "arch": architecture,
                "val_acc": -1.0, "score": -1.0,
                "est_ram": est_ram, "est_flash": est_flash,
                "est_macc": real_macc, "feasible": False,
            })
            return

        train_ds, val_ds = self._load_proxy_set(batch_size=32)
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(
            optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=1,
            restore_best_weights=True, verbose=0
        )

        hist = model.fit(
            train_ds, epochs=self.proxy_epochs,
            validation_data=val_ds,
            callbacks=[early_stop], verbose=0
        )
        val_acc = float(np.amax(hist.history['val_accuracy']))
        score = compute_score(
            val_acc, est_ram, est_flash, real_macc,
            self.max_ram, self.max_flash, self.max_macc,
            self.accuracy_weight, self.efficiency_weight,
            self.headroom_weight
        )

        q.put({
            "key": key, "arch": architecture,
            "val_acc": round(val_acc, 4), "score": score,
            "est_ram": est_ram, "est_flash": est_flash,
            "est_macc": real_macc, "feasible": True,
        })

    def _evaluate_batch(self, architectures):
        results = []
        pending = []
        for arch in architectures:
            key = architecture_to_key(arch)
            if key in self._arch_cache:
                cached = self._arch_cache[key]
                results.append(cached)
                print(f"  [CACHE] k={arch['k']}, cells={len(arch['cells'])} "
                      f"→ score={cached.get('score', '?'):.4f}")
            else:
                pending.append(arch)

        batch_size = self.parallel_workers
        for i in range(0, len(pending), batch_size):
            batch = pending[i:i + batch_size]
            processes = []

            for arch in batch:
                q = Queue()
                p = Process(target=self._proxy_evaluate_process,
                            args=(q, arch))
                p.start()
                processes.append((p, q, arch))

            for p, q, arch in processes:
                p.join()
                if not q.empty():
                    result = q.get()
                    self._arch_cache[result["key"]] = result
                    results.append(result)
                    feas = "✓" if result["feasible"] else "✗"
                    print(f"  k={arch['k']:2d}, cells={len(arch['cells'])} "
                          f"[{arch['cells'][0]['block_type'][:3]}...] → "
                          f"val_acc={result['val_acc']:.4f}  "
                          f"score={result['score']:.4f}  {feas}")

            self._save_cache()
        return results

    # ── 3-Stage hybrid search ─────────────────────────────────────────────

    def _stage1_random(self):
        candidates = []
        if self.seed_k is not None and self.seed_c is not None:
            seed_arch = {
                "k": self.seed_k,
                "cells": [{
                    "block_type": "standard_conv", "kernel_size": 3,
                    "stride": 2, "skip_connection": False,
                    "activation": "relu", "expansion_ratio": 1,
                } for _ in range(self.seed_c)]
            }
            candidates.append(seed_arch)
            n_random = self.n_random_stage - 1
            print(f"\n[ENAS v2.1] Stage 1: Seeded random "
                  f"(1 seed + {n_random} random)\n")
        else:
            n_random = self.n_random_stage
            print(f"\n[ENAS v2.1] Stage 1: Random search "
                  f"({n_random} candidates)\n")

        candidates += [
            sample_random_architecture(
                self._k_range, self._num_cells_range)
            for _ in range(n_random)
        ]
        return self._evaluate_batch(candidates)

    def _stage2_top_k(self, stage1_results):
        """
        Two-phase Stage 2 selection:
            Phase A: accuracy-first filter (top 50% by val_acc)
            Phase B: score-based final ranking among survivors
        """
        feasible = [r for r in stage1_results
                    if r.get("feasible") and r.get("score", -1) > 0]
        if not feasible:
            return []

        # Phase A
        feasible.sort(key=lambda r: r.get("val_acc", 0), reverse=True)
        acc_cutoff = max(len(feasible) // 2, self.top_k_select)
        survivors = feasible[:acc_cutoff]

        # Phase B
        survivors.sort(key=lambda r: r.get("score", 0), reverse=True)
        top_k = survivors[:self.top_k_select]

        print(f"\n[ENAS v2.1] Stage 2: Top-{len(top_k)} survivors "
              f"(Phase A: {len(survivors)}, Phase B: {len(top_k)})")
        return top_k

    def _stage3_refine(self, top_k_results):
        if not top_k_results:
            return []
        print(f"\n[ENAS v2.1] Stage 3: Refinement "
              f"({len(top_k_results)} × {self.n_mutations_per_arch})\n")

        mutants = []
        for r in top_k_results:
            for _ in range(self.n_mutations_per_arch):
                mutants.append(mutate_architecture(r["arch"], 1))
        return self._evaluate_batch(mutants)

    # ── Public API ─────────────────────────────────────────────────────────

    def search(self, save_search_history=False):
        start = datetime.datetime.now()

        stage1 = self._stage1_random()
        top_k  = self._stage2_top_k(stage1)
        if not top_k:
            print("\nNo feasible solution found.\n")
            return

        stage3 = self._stage3_refine(top_k)
        all_results = stage1 + stage3
        feasible = [r for r in all_results
                    if r.get("feasible") and r.get("score", -1) > 0]
        if not feasible:
            print("\nNo feasible solution found.\n")
            return

        best = max(feasible, key=lambda r: r["score"])
        end  = datetime.datetime.now()

        self.resulting_architecture = best["arch"]
        self.resulting_score        = best["score"]

        arch   = best["arch"]
        k_best = arch["k"]
        c_best = len(arch["cells"])

        # NanoNAS-compatible architecture line for parser
        print(
            f"\nResulting architecture: "
            f"{{'k': {k_best}, 'c': {c_best}, "
            f"'RAM': {best['est_ram']}, "
            f"'Flash': {best['est_flash']}, "
            f"'MACC': {best['est_macc']}, "
            f"'max_val_acc': {best['val_acc']}, "
            f"'score': {best['score']}}}\n"
        )
        print(f"Cell configuration:")
        for i, cell in enumerate(arch["cells"]):
            print(f"  Cell {i + 1}: {cell}")
        print(f"\nElapsed time (search): {end - start}\n")

    def _train_process(self, training_epochs, training_learning_rate,
                       training_batch_size):
        train_ds, val_ds = self.load_training_set(training_batch_size)
        model, _, _ = build_model(
            self.input_shape, self.num_classes,
            self.resulting_architecture
        )

        steps = max(1, tf.data.experimental.cardinality(train_ds).numpy())
        lr = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=training_learning_rate,
            decay_steps=training_epochs * steps,
            alpha=1e-6
        )
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(optimizer=opt, loss='categorical_crossentropy',
                      metrics=['accuracy'])

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(self.path_to_resulting_model),
                save_weights_only=False, monitor='val_accuracy',
                mode='max', save_best_only=True, verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy', patience=15,
                restore_best_weights=True, verbose=1
            ),
        ]
        hist = model.fit(
            train_ds, epochs=training_epochs,
            validation_data=val_ds, validation_freq=1,
            callbacks=callbacks
        )
        best_acc = round(np.amax(hist.history['val_accuracy']), 3)
        print(f'\nmax val acc: {best_acc}')
        print(f"\nKeras model saved in: {self.path_to_resulting_model}\n")

    def train(self, training_epochs, training_learning_rate,
              training_batch_size):
        start = datetime.datetime.now()
        p = Process(target=self._train_process,
                    args=(training_epochs, training_learning_rate,
                          training_batch_size))
        p.start()
        p.join()
        print(f"Elapsed time (training): "
              f"{datetime.datetime.now() - start}\n")

    def _apply_ptq_process(self):
        train_ds, _ = self.load_training_set()
        model = tf.keras.models.load_model(self.path_to_resulting_model)
        custom_tmp = str(self.save_path / 'tmp_tflite')
        quantize_model(model, train_ds,
                       self.path_to_quantized_resulting_model,
                       custom_tmp_dir=custom_tmp)
        print(f"\nTflite model saved in: "
              f"{self.path_to_quantized_resulting_model.resolve()}\n")

    def apply_uint8_post_training_quantization(self):
        p = Process(target=self._apply_ptq_process)
        p.start()
        p.join()

    def _test_keras_process(self, path_to_test_set):
        test_ds = self.load_test_set(path_to_test_set)
        model = tf.keras.models.load_model(self.path_to_resulting_model)
        results = model.evaluate(test_ds, verbose=0)
        print(f"\nKeras model test accuracy: {results[1]}\n", flush=True)

    def test_keras_model(self, path_to_test_set):
        p = Process(target=self._test_keras_process,
                    args=(path_to_test_set,))
        p.start()
        p.join()

    def _test_tflite_process(self, path_to_test_set):
        from enas.quantization import evaluate_tflite
        test_ds = self.load_test_set(path_to_test_set)
        acc = evaluate_tflite(self.path_to_quantized_resulting_model,
                              test_ds)
        print(f"\nTflite model test accuracy: {acc}", flush=True)
        print(f"\nTflite model in: "
              f"{self.path_to_quantized_resulting_model.resolve()}\n",
              flush=True)

    def test_tflite_model(self, path_to_test_set):
        p = Process(target=self._test_tflite_process,
                    args=(path_to_test_set,))
        p.start()
        p.join()
