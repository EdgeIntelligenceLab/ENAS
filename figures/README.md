# Figures

The paper's canonical figures are committed in `output/`:

| File | Paper | How produced |
| --- | --- | --- |
| `feasibility_matrix.png` | Figure 2 | analytical feasibility grid (committed as used in the paper) |
| `fig4_pareto.pdf` | Figure 3 | **reproducible**: `python figures/scripts/plot_pareto.py --csv-dir results/parsed_csv` |
| `fig_delta_combined.pdf` | Figure 4 | 4-panel Δ heatmaps (committed as used in the paper) |

`plot_pareto.py` regenerates Figure 3 exactly from the shipped CSVs (legend in the
top-right). The other plotting scripts (`plot_heatmaps.py`, `plot_pipeline.py`) are
provided as illustrative utilities; the committed PDFs/PNGs in `output/` are the exact
figures used in the paper.
