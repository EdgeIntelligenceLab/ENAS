# ENAS Framework Architecture

## Pipeline Overview

```
Phase 0 — Feasibility Check     (analytical, Eq. 3)
   ↓
Phase 1 — 3-Stage Hybrid Search
   ├─ Stage 1: 30 random candidates
   ├─ Stage 2: top-K (=8) accuracy-first selection
   └─ Stage 3: 40 mutations (5 mutants × 8 survivors)
   ↓
Phase 2 — Full training         (100 epochs, cosine LR)
   ↓
Phase 3 — INT8 PTQ + evaluation
```

## Cell-Based Architecture Template

```
Input (W × H × 3)
   ↓
Stem: Conv2D(k, 3×3) → BN → ReLU
   ↓
Cell 1: (block_type, kernel, stride, skip, activation, expansion)
   ↓
Cell 2: …
   ↓
Cell N: …                       (N ∈ [1, 5])
   ↓
GlobalAveragePool → Dropout(0.4) → Dense(2) → Softmax
```

## Key Code Modules

| Module | Role |
|---|---|
| `enas/enas_v2_1.py`     | Main ENAS class with 3-stage hybrid search |
| `enas/search_space.py`  | Cell-based parameter space + random sampler |
| `enas/blocks.py`        | Block builders (std_conv, DS, bottleneck) |
| `enas/estimators.py`    | Analytical RAM/Flash/MACC (Eq. 3) |
| `enas/scoring.py`       | Multi-objective scoring (Eq. 6) |
| `enas/mutations.py`     | Stage 3 mutation strategies |
| `enas/quantization.py`  | INT8 PTQ conversion |
