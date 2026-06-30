"""
NanoNAS — Baseline greedy coordinate-ascent NAS.

Reference implementation for comparison against ENAS v2.1.

Search space: 2D (k, c)
    k = number of kernels in first conv layer
    c = number of additional cells

Search strategy: greedy coordinate ascent
    1. Fix k, increase c until val accuracy stops improving
    2. Increase k, repeat
    3. Terminate when increasing k stops helping

Hardware evaluation: full TFLite conversion + stm32tflm
    (~5–10 min per candidate; this is the main bottleneck)
"""

import os
import re
import datetime
import subprocess
import tempfile
import shutil
from pathlib import Path
from multiprocessing import Process, Queue

import numpy as np
import tensorflow as tf


class NanoNAS:
    """Baseline NanoNAS implementation."""

    def __init__(self, max_ram, max_flash, max_macc,
                 path_to_training_set, val_split,
                 cache=False, input_shape=(50, 50, 3),
                 save_path='./'):
        self.path_to_training_set = path_to_training_set
        self.num_classes  = len(next(os.walk(path_to_training_set))[1])
        self.val_split    = val_split
        self.input_shape  = input_shape
        self.max_ram      = max_ram
        self.max_flash    = max_flash
        self.max_macc     = max_macc
        self.cache        = cache
        self.save_path    = Path(save_path)
        os.makedirs(self.save_path, exist_ok=True)
        self.path_to_resulting_model = (
            self.save_path / 'resulting_architecture.h5'
        )
        self.path_to_quantized_resulting_model = (
            self.save_path / 'resulting_architecture.tflite'
        )

    def Model(self, k, c):
        """Build the NanoNAS architecture: Conv → BN → ReLU → (Pool+Conv+BN+ReLU)*c"""
        kernel_size = (3, 3)
        pool_size = (2, 2)
        pool_strides = (2, 2)
        number_of_cells_limited = False
        macc = 0

        inputs = tf.keras.Input(shape=self.input_shape)
        n = k
        multiplier = 2

        c_in = self.input_shape[2]
        x = tf.keras.layers.Conv2D(n, kernel_size, padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        macc += c_in * kernel_size[0] * kernel_size[1] * x.shape[1] * x.shape[2] * x.shape[3]

        for i in range(1, c + 1):
            if x.shape[1] <= 1 or x.shape[2] <= 1:
                number_of_cells_limited = True
                break
            n = int(np.ceil(n * multiplier))
            multiplier = multiplier - 2 ** -i
            x = tf.keras.layers.MaxPooling2D(
                pool_size=pool_size, strides=pool_strides,
                padding='valid')(x)
            x = tf.keras.layers.Conv2D(n, kernel_size, padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
            c_in = x.shape[3]
            macc += c_in * kernel_size[0] * kernel_size[1] * x.shape[1] * x.shape[2] * x.shape[3]

        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        outputs = tf.keras.layers.Dense(self.num_classes,
                                         activation='softmax')(x)
        macc += x.shape[1] * outputs.shape[1]

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model, macc, number_of_cells_limited

    def load_training_set(self, batch_size=1):
        cm = 'rgb' if self.input_shape[2] == 3 else 'grayscale'
        train_ds = tf.keras.utils.image_dataset_from_directory(
            directory=self.path_to_training_set,
            labels='inferred', label_mode='categorical',
            color_mode=cm, batch_size=batch_size,
            image_size=self.input_shape[0:2],
            shuffle=True, seed=11,
            validation_split=self.val_split, subset='training')
        val_ds = tf.keras.utils.image_dataset_from_directory(
            directory=self.path_to_training_set,
            labels='inferred', label_mode='categorical',
            color_mode=cm, batch_size=batch_size,
            image_size=self.input_shape[0:2],
            shuffle=True, seed=11,
            validation_split=self.val_split, subset='validation')
        pf = tf.data.AUTOTUNE
        if self.cache:
            return train_ds.cache().prefetch(pf), val_ds.cache().prefetch(pf)
        return train_ds.prefetch(pf), val_ds.prefetch(pf)

    def compile_model(self, model, learning_rate):
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=opt, loss='categorical_crossentropy',
                      metrics=['accuracy'])

    def quantize_model(self, model, train_ds, path_to_tflite):
        def representative_dataset():
            for data in train_ds.rebatch(1).take(150):
                yield [tf.dtypes.cast(data[0], tf.float32)]

        custom_tmp = str(self.save_path / 'tmp_tflite')
        os.makedirs(custom_tmp, exist_ok=True)
        old = tempfile.tempdir
        tempfile.tempdir = custom_tmp
        try:
            conv = tf.lite.TFLiteConverter.from_keras_model(model)
            conv.optimizations = [tf.lite.Optimize.DEFAULT]
            conv.representative_dataset = representative_dataset
            conv.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            conv.inference_input_type = tf.uint8
            conv.inference_output_type = tf.uint8
            tflite = conv.convert()
        finally:
            tempfile.tempdir = old
            shutil.rmtree(custom_tmp, ignore_errors=True)
        with open(path_to_tflite, 'wb') as f:
            f.write(tflite)

    def evaluate_flash_and_peak_ram_occupancy(self, model, train_ds):
        """Use stm32tflm binary for ground-truth measurement."""
        path = self.save_path / 'temp.tflite'
        self.quantize_model(model, train_ds, path)
        proc = subprocess.Popen(["./stm32tflm", path],
                                 stdout=subprocess.PIPE)
        try:
            outs, _ = proc.communicate(timeout=15)
            flash, ram = re.findall(r'\d+', str(outs))
            os.remove(path)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            os.remove(path)
            raise RuntimeError("stm32tflm timed out")
        return int(flash), int(ram)

    def _evaluate_process(self, q, k, c):
        train_ds, val_ds = self.load_training_set(16)
        model, macc, limited = self.Model(k, c)
        self.compile_model(model, 0.001)
        hist = model.fit(train_ds, epochs=3,
                         validation_data=val_ds,
                         validation_freq=1, verbose=0)
        try:
            flash, ram = self.evaluate_flash_and_peak_ram_occupancy(
                model, train_ds)
        except Exception:
            flash, ram = -1, -1

        feasible = (macc <= self.max_macc and flash <= self.max_flash
                    and ram <= self.max_ram and not limited)

        q.put({
            'k': k,
            'c': c if not limited else f"{c} (Not feasible)",
            'RAM': ram if ram <= self.max_ram
                   else f"{ram} (Outside the upper bound of {ram - self.max_ram} Byte)",
            'Flash': flash if flash <= self.max_flash
                     else f"{flash} (Outside the upper bound of {flash - self.max_flash} Byte)",
            'MACC': macc if macc <= self.max_macc
                    else f"{macc} (Outside the upper bound of {macc - self.max_macc} MAC)",
            'max_val_acc': np.around(np.amax(hist.history['val_accuracy']),
                                     decimals=3) if feasible else -3
        })

    def search(self, save_search_history=False):
        start = datetime.datetime.now()
        best = {'k': -1, 'c': -1, 'max_val_acc': -2}
        new  = {'k': -1, 'c': -1, 'max_val_acc': -1}

        k = 1
        while new['max_val_acc'] > best['max_val_acc']:
            best = new
            c = -1
            prev    = {'k': -1, 'c': -1, 'max_val_acc': -2}
            current = {'k': -1, 'c': -1, 'max_val_acc': -1}
            while current['max_val_acc'] > prev['max_val_acc']:
                prev = current
                c += 1
                q = Queue()
                p = Process(target=self._evaluate_process, args=(q, k, c))
                p.start()
                p.join()
                if q.empty():
                    current = {'k': k, 'c': c, 'max_val_acc': -1}
                else:
                    current = q.get()
                print(f"\n\n{current}\n\n")
            new = prev
            k += 1

        end = datetime.datetime.now()
        if 0 < best['max_val_acc']:
            print(f"Resulting architecture: {best}\n")
            print(f"Elapsed time (search): {end - start}\n")
            self.resulting_architecture = best
        else:
            print("No feasible solution found.\n")

    # train / quantize / test methods kept identical to ENAS for parity.
    # See enas/enas_v2_1.py for analogous implementations.
