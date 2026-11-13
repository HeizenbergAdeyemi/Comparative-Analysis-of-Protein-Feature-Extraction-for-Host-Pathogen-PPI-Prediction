# Results

| Path | Contents |
|---|---|
| `figures/<organism>/<feature>/` | `<model>_confusion_matrix.png` and `<model>_roc_curve.png` for rf, svm, xgb |
| `figures/<organism>/README.md` | Gallery with all plots and metrics for that organism |
| `metrics/results_summary.csv` | Confusion counts, accuracy, precision, recall, F1, ROC-AUC for all 90 combinations |
| `summary/` | Aggregate heatmaps and the PSSM-collapse figure |

Feature folder names: `aac`, `dipeptide` (DPC), `kmer`, `pssm`, `esm2`, `protbert`.

## Provenance of the numbers

The plots in `figures/` are the original figures produced at the end of the project, renamed systematically. `results_summary.csv` was **reconstructed from those figures** (confusion-matrix cell counts read from each heatmap, ROC-AUC from each legend, rounded to 2 decimals as printed), and every row was checked against the known per-organism class totals. Accuracy, precision, recall and F1 are derived from the counts. If `evaluate_model.py` also writes metrics to disk, treat that output as authoritative and replace this file with it.
