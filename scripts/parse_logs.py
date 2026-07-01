#!/usr/bin/env python
"""
parse_logs.py
=============
Parse experiment log files into structured CSV summaries.

Reads from results/raw_logs/<method>_<dataset>/<hardware>_size<N>_run<R>/
Writes to results/parsed_csv/.

Outputs:
    - {method}_{dataset}_summary.csv         : one row per experiment
    - {method}_{dataset}_search_history.csv  : one row per candidate evaluated
    - {method}_{dataset}_cell_configs.csv    : per-cell architecture details (ENAS v2+)

Usage:
    python scripts/parse_logs.py
    python scripts/parse_logs.py --method enas --dataset vww
    python scripts/parse_logs.py --input results/raw_logs/enas_vww --output results/parsed_csv
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT      = Path(__file__).resolve().parent.parent
DEFAULT_INPUT  = REPO_ROOT / "results" / "raw_logs"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "parsed_csv"


def find_log_file(folder_path):
    for fname in os.listdir(folder_path):
        if fname.startswith("log") and fname.endswith(".txt"):
            return os.path.join(folder_path, fname)
    return None


def parse_folder_name(folder):
    m = re.match(r"^(.+)_size(\d+)_run(\d+)$", folder)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3))
    return folder, None, None


def parse_float(val_str):
    if val_str is None:
        return None
    m = re.search(r"np\.float64\(([\d\.]+)\)", val_str)
    if m:
        return float(m.group(1))
    try:
        return float(val_str)
    except (ValueError, TypeError):
        return None


def detect_version(text):
    if "ENAS v2.1" in text or "v2_1" in text:
        return "enas_v2_1"
    if "Cell configuration:" in text or "ENAS v2" in text:
        return "enas_v2"
    if "[ENAS]" in text:
        return "enas_v1"
    return "nanonas"


def experiment_status(text):
    if "[SKIPPED] Min possible RAM" in text:
        return "INFEASIBLE"
    if "Experiment finished at" in text:
        if "No feasible solution found" in text and "Resulting architecture" not in text:
            if "Elapsed time (search)" not in text:
                return "INFEASIBLE"
            return "NO_FEASIBLE_ARCH"
        return "COMPLETED"
    if "[ERROR]" in text:
        return "FAILED"
    if "Resulting architecture" in text:
        return "SEARCH_DONE"
    return "IN_PROGRESS"


def parse_cells(text):
    """Extract per-cell configuration block from ENAS v2+ logs."""
    cell_rows = []
    block = re.search(
        r"Cell configuration:(.*?)(?=\nElapsed|\nmax val|\Z)",
        text, re.DOTALL
    )
    if not block:
        return cell_rows
    pattern = re.compile(r"Cell\s+(\d+):\s*\{([^}]+)\}")
    for m in pattern.finditer(block.group(1)):
        idx = int(m.group(1))
        body = m.group(2)

        def extract(key):
            km = re.search(rf"'{key}':\s*(['\"]?)([^,'\"}}]+)\1", body)
            if not km:
                return None
            v = km.group(2).strip()
            if v == "True":  return True
            if v == "False": return False
            try: return int(v)
            except: pass
            try: return float(v)
            except: pass
            return v

        cell_rows.append({
            "cell_index":      idx,
            "block_type":      extract("block_type"),
            "kernel_size":     extract("kernel_size"),
            "stride":          extract("stride"),
            "skip_connection": extract("skip_connection"),
            "activation":      extract("activation"),
            "expansion_ratio": extract("expansion_ratio"),
        })
    return cell_rows


def parse_candidates(text):
    """Parse search candidates regardless of log version."""
    rows = []

    # ENAS v2.x candidate format
    p_v2 = re.compile(
        r"k=\s*(\d+),\s*cells?=\s*(\d+)(?:\s*\[[^\]]*\])?\s*→\s*"
        r"val_acc=([\-\d\.]+)\s+score=([\-\d\.]+)\s+(✓|✗)"
    )
    matches_v2 = list(p_v2.finditer(text))
    if matches_v2:
        for m in matches_v2:
            rows.append({
                "k": int(m.group(1)),
                "c": int(m.group(2)),
                "val_acc": float(m.group(3)),
                "score": float(m.group(4)),
                "feasible": m.group(5) == "✓",
            })
        return rows

    # ENAS v1 candidate format
    p_v1 = re.compile(
        r"k=\s*(\d+),\s*c=\s*(\d+)\s*→\s*val_acc=([\-\d\.]+)\s+score=([\-\d\.]+)\s+(✓|✗)"
    )
    for m in p_v1.finditer(text):
        rows.append({
            "k": int(m.group(1)), "c": int(m.group(2)),
            "val_acc": float(m.group(3)),
            "score": float(m.group(4)),
            "feasible": m.group(5) == "✓",
        })
    if rows:
        return rows

    # NanoNAS dict format
    p_nn = re.compile(
        r"\{'k':\s*(\d+),\s*'c':\s*([^,]+),\s*"
        r"(?:'RAM':\s*([^,]+),\s*'Flash':\s*([^,]+),\s*'MACC':\s*([^,]+),\s*)?"
        r"'max_val_acc':\s*([^}]+)\}"
    )
    for m in p_nn.finditer(text):
        c_raw = m.group(2).strip().strip("'\"")
        c_m = re.search(r"\d+", c_raw)
        c_val = int(c_m.group()) if c_m else 0
        acc = parse_float(m.group(6).strip())
        rows.append({
            "k": int(m.group(1)), "c": c_val,
            "val_acc": acc if acc and acc >= 0 else None,
            "score": None,
            "feasible": acc is not None and acc >= 0,
        })
    return rows


def parse_log(log_path, hardware, input_size, run):
    with open(log_path, "r") as f:
        text = f.read()

    version = detect_version(text)
    status  = experiment_status(text)

    shape = re.search(r"Input shape:\s*\((\d+),\s*(\d+),\s*(\d+)\)", text)
    if shape:
        input_size = int(shape.group(1))

    # Architecture
    arch_match = re.search(
        r"Resulting architecture:\s*\{.*?'k':\s*(\d+).*?'c':\s*(\d+).*?"
        r"'RAM':\s*(\d+).*?'Flash':\s*(\d+).*?'MACC':\s*(\d+).*?"
        r"'max_val_acc':\s*(np\.float64\([\d\.]+\)|[\d\.]+)"
        r"(?:.*?'score':\s*([\d\.]+))?",
        text, re.DOTALL
    )
    k = c = nas_ram = nas_flash = nas_macc = nas_acc = score = None
    if arch_match:
        k         = int(arch_match.group(1))
        c         = int(arch_match.group(2))
        nas_ram   = int(arch_match.group(3))
        nas_flash = int(arch_match.group(4))
        nas_macc  = int(arch_match.group(5))
        nas_acc   = parse_float(arch_match.group(6))
        score     = float(arch_match.group(7)) if arch_match.group(7) else None

    # Timing
    sm = re.search(r"Elapsed time \(search\):\s*([0-9:\.]+)", text)
    tm = re.search(r"Elapsed time \(training\):\s*([0-9:\.]+)", text)
    rm = re.search(r"Total runtime:\s*([0-9:\.]+)", text)

    # Accuracy
    vm = re.search(r"max val acc:\s*([\d\.]+)", text)
    km = re.search(r"Keras model test accuracy:\s*([\d\.]+)", text)
    lm = re.search(r"Tflite model test accuracy:\s*([\d\.]+)", text)

    # Block types
    block_types = re.findall(r"'block_type':\s*'([^']+)'", text)
    block_summary = (json.dumps(dict(Counter(block_types)))
                     if block_types else None)

    candidates = parse_candidates(text)
    cells      = parse_cells(text) if version in ("enas_v2", "enas_v2_1") else []

    summary = {
        "hardware":         hardware,
        "input_size":       input_size,
        "run":              run,
        "status":           status,
        "version":          version,
        "best_k":           k,
        "best_c":           c,
        "nas_ram":          nas_ram,
        "nas_flash":        nas_flash,
        "nas_macc":         nas_macc,
        "nas_val_acc":      nas_acc,
        "score":            score,
        "block_types":      block_summary,
        "total_candidates": len(candidates),
        "feasible_candidates": sum(1 for r in candidates if r["feasible"]),
        "search_time":      sm.group(1) if sm else None,
        "train_val_acc":    float(vm.group(1)) if vm else None,
        "keras_test_acc":   float(km.group(1)) if km else None,
        "tflite_test_acc":  float(lm.group(1)) if lm else None,
        "train_time":       tm.group(1) if tm else None,
        "total_time":       rm.group(1) if rm else None,
    }

    candidate_rows = []
    for c_row in candidates:
        candidate_rows.append({
            "hardware":   hardware,
            "input_size": input_size,
            "run":        run,
            **c_row
        })

    cell_rows = []
    for cell in cells:
        cell_rows.append({
            "hardware":   hardware,
            "input_size": input_size,
            "run":        run,
            **cell
        })

    return summary, candidate_rows, cell_rows


def parse_directory(input_dir, label):
    summary_rows, candidate_rows, cell_rows = [], [], []
    for folder in sorted(os.listdir(input_dir)):
        folder_path = os.path.join(input_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        hw, size, run = parse_folder_name(folder)
        if run is None:
            continue
        log = find_log_file(folder_path)
        if log is None:
            continue
        try:
            s, c, cell = parse_log(log, hw, size, run)
            summary_rows.append(s)
            candidate_rows.extend(c)
            cell_rows.extend(cell)
            print(f"  [OK]  {hw} {size}x{size} run={run} → {s['status']}")
        except Exception as e:
            print(f"  [ERR] {folder}: {e}")
    return summary_rows, candidate_rows, cell_rows


def write_csv(rows, path, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description="Parse experiment logs to CSV.")
    p.add_argument("--input",   type=str, default=None)
    p.add_argument("--output",  type=str, default=str(DEFAULT_OUTPUT))
    p.add_argument("--method",  type=str, default=None)
    p.add_argument("--dataset", type=str, default=None)
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Process either a single directory or all method/dataset subdirs
    if args.input:
        dirs = [(args.input, Path(args.input).name)]
    else:
        dirs = []
        for sub in sorted(DEFAULT_INPUT.glob("*")):
            if not sub.is_dir():
                continue
            if args.method and args.method not in sub.name:
                continue
            if args.dataset and args.dataset not in sub.name:
                continue
            dirs.append((str(sub), sub.name))

    summary_fields = [
        "hardware", "input_size", "run", "status", "version",
        "best_k", "best_c",
        "nas_ram", "nas_flash", "nas_macc",
        "nas_val_acc", "score", "block_types",
        "total_candidates", "feasible_candidates",
        "search_time", "train_val_acc", "keras_test_acc",
        "tflite_test_acc", "train_time", "total_time",
    ]
    candidate_fields = ["hardware", "input_size", "run",
                        "k", "c", "val_acc", "score", "feasible"]
    cell_fields      = ["hardware", "input_size", "run", "cell_index",
                        "block_type", "kernel_size", "stride",
                        "skip_connection", "activation", "expansion_ratio"]

    for input_dir, label in dirs:
        print(f"\n── Parsing {input_dir} ──")
        s, c, cell = parse_directory(input_dir, label)
        if not s:
            print(f"  (no experiments found)")
            continue
        write_csv(s, os.path.join(args.output, f"{label}_summary.csv"),
                  summary_fields)
        write_csv(c, os.path.join(args.output, f"{label}_search_history.csv"),
                  candidate_fields)
        if cell:
            write_csv(cell, os.path.join(args.output, f"{label}_cell_configs.csv"),
                      cell_fields)
        print(f"  ✔ {len(s)} summary rows, {len(c)} candidates, {len(cell)} cells")


if __name__ == "__main__":
    main()
