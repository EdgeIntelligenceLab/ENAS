#!/usr/bin/env python3
"""
build_camera_ready_tables.py
============================
Regenerate the paper's main result tables (with mean +/- std and Wilcoxon
significance) directly from the parsed result CSVs shipped in
results/parsed_csv/. No GPU, no re-training, runs in seconds.

This reproduces:
  - Table 2  Aggregate trade-off (+ significance)
  - Table 3  Measured resource footprint of selected models (weakness f)
  - Table 4  Focused 50/64 study (warm-seeded)
  - Table 5  Per-resolution sweep
  - Table 6  Per-hardware best (VWW)

Usage:
  python scripts/build_camera_ready_tables.py [--csv-dir results/parsed_csv]
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

HW_ORDER = ['STM32L010RBT6','NUCLEO-L010RB','ArduinoNano33IoT','NUCLEO-L412KB',
            'RaspberryPiPico','ArduinoNano33BLE','ArduinoNiclaVision','STM32H743ZI']


def _parse_time(s):
    s = str(s).strip()
    if s in ('', 'nan'):
        return np.nan
    if ':' in s:
        h, m, rest = s.split(':')
        return int(h) * 60 + int(m) + float(rest) / 60
    try:
        return float(s)
    except ValueError:
        return np.nan


def load(path):
    d = pd.read_csv(path)
    d = d[d['Status'].astype(str).str.upper() == 'COMPLETED'].copy()
    for c in ['NAS RAM (bytes)', 'NAS Flash (bytes)', 'NAS MACC',
              'TFLite Test Acc', 'Best k']:
        if c in d:
            d[c] = pd.to_numeric(d[c], errors='coerce')
    d['Px'] = d['Input Size'].astype(str).str.extract(r'(\d+)').astype(int)
    d['acc'] = d['TFLite Test Acc'] * 100
    d['stime'] = d['Search Time'].map(_parse_time)
    return d


def load_measured(path):
    m = pd.read_csv(path)
    m = m[m['Status'] == 'ok'].copy()
    for c in ['Measured Flash (bytes)', 'Measured RAM (bytes)']:
        m[c] = pd.to_numeric(m[c], errors='coerce')
    m['Px'] = m['Input Size'].astype(int)
    return m


def table2_aggregate(N, E, name):
    print(f"\n=== Table 2 — Aggregate ({name}) ===")
    for nm, df in [('NanoNAS', N), ('ENAS', E)]:
        c = df.groupby(['Hardware', 'Px']).agg(a=('acc', 'mean'), t=('stime', 'mean'))
        print(f"  {nm:8s} acc {c.a.mean():5.2f}+/-{c.a.std():.2f}  "
              f"median {c.a.median():5.2f}  search {c.t.mean():5.1f}+/-{c.t.std():4.1f} min")
    cN = N.groupby(['Hardware', 'Px']).agg(a=('acc', 'mean'), t=('stime', 'mean'))
    cE = E.groupby(['Hardware', 'Px']).agg(a=('acc', 'mean'), t=('stime', 'mean'))
    j = cN.join(cE, lsuffix='_N', rsuffix='_E').dropna()
    print(f"  speedup {cN.t.mean()/cE.t.mean():.2f}x | "
          f"acc Wilcoxon p={stats.wilcoxon(j.a_N, j.a_E).pvalue:.3g} | "
          f"search Wilcoxon p={stats.wilcoxon(j.t_N, j.t_E).pvalue:.3g}")


def table3_resource(N, E, M, name):
    print(f"\n=== Table 3 — Measured resource footprint ({name}) ===")
    me = M
    e = me.groupby(['Hardware', 'Px']).agg(ram=('Measured RAM (bytes)', 'mean'),
                                           fl=('Measured Flash (bytes)', 'mean')).reset_index()
    ea = E.groupby(['Hardware', 'Px']).agg(mc=('NAS MACC', 'mean'),
                                           ac=('acc', 'mean')).reset_index()
    e = e.merge(ea, on=['Hardware', 'Px'])
    n = N.groupby(['Hardware', 'Px']).agg(ram=('NAS RAM (bytes)', 'mean'),
                                          fl=('NAS Flash (bytes)', 'mean'),
                                          mc=('NAS MACC', 'mean'),
                                          ac=('acc', 'mean')).reset_index()
    for nm, t in [('NanoNAS', n), ('ENAS', e)]:
        print(f"  {nm:8s} RAM {t.ram.mean()/1024:5.1f}+/-{t.ram.std()/1024:4.1f}KB  "
              f"Flash {t.fl.mean()/1024:5.1f}+/-{t.fl.std()/1024:4.1f}KB  "
              f"MACC {t.mc.mean()/1e6:.2f}M")
    j = e.merge(n, on=['Hardware', 'Px'], suffixes=('_E', '_N'))
    mt = j[abs(j.ac_E - j.ac_N) <= 1.0]
    print(f"  ratio all     E/N: RAM {e.ram.mean()/n.ram.mean():.2f}x  "
          f"Flash {e.fl.mean()/n.fl.mean():.2f}x  MACC {e.mc.mean()/n.mc.mean():.2f}x")
    print(f"  ratio matched E/N (n={len(mt)}): RAM {mt.ram_E.mean()/mt.ram_N.mean():.2f}x  "
          f"Flash {mt.fl_E.mean()/mt.fl_N.mean():.2f}x  MACC {mt.mc_E.mean()/mt.mc_N.mean():.2f}x")


def table5_per_resolution(N, E, name):
    print(f"\n=== Table 5 — Per-resolution ({name}) ===")
    for px in [32, 48, 50, 64, 72, 80, 96, 112, 128]:
        cN = N[N.Px == px].groupby('Hardware').agg(a=('acc', 'mean'), t=('stime', 'mean'))
        cE = E[E.Px == px].groupby('Hardware').agg(a=('acc', 'mean'), t=('stime', 'mean'))
        if len(cN) == 0:
            continue
        print(f"  {px:3d} N={len(cN)}  Nano {cN.a.mean():5.1f}+/-{cN.a.std():4.1f} / "
              f"ENAS {cE.a.mean():5.1f}+/-{cE.a.std():4.1f}  d{cE.a.mean()-cN.a.mean():+5.2f} | "
              f"speedup {cN.t.mean()/cE.t.mean():.2f}x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv-dir', default='results/parsed_csv')
    a = ap.parse_args()
    d = Path(a.csv_dir)
    Nv, Ev = load(d / 'nanonas_vww_summary.csv'), load(d / 'enas_vww_summary.csv')
    Nc, Ec = load(d / 'nanonas_cancer_summary.csv'), load(d / 'enas_cancer_summary.csv')
    M = load_measured(d / 'enas_measured_resources.csv')
    Mv, Mc = M[M.Dataset == 'VWW'], M[M.Dataset == 'Cancer']

    table2_aggregate(Nv, Ev, 'VWW'); table2_aggregate(Nc, Ec, 'Cancer')
    table3_resource(Nv, Ev, Mv, 'VWW'); table3_resource(Nc, Ec, Mc, 'Cancer')
    table5_per_resolution(Nv, Ev, 'VWW'); table5_per_resolution(Nc, Ec, 'Cancer')
    print("\nDone. These match the camera-ready Tables 2, 3, 5.")


if __name__ == '__main__':
    main()
