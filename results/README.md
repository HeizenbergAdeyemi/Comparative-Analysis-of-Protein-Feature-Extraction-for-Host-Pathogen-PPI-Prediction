# Results

| Path | Contents |
|---|---|
| `figures/<organism>/<feature>/` | `<model>_confusion_matrix.png` and `<model>_roc_curve.png` for rf, svm, xgb |
| `figures/<organism>/README.md` | Gallery with all plots and metrics for that organism |
| `metrics/results_summary.csv` | Confusion counts, accuracy, precision, recall, F1, ROC-AUC for all 90 combinations |
| `summary/` | Aggregate heatmaps and the PSSM-collapse figure |

Feature folder names: `aac`, `dipeptide` (DPC), `kmer`, `pssm`, `esm2`, `protbert`. In the pipeline code these are `AAC`, `DPC`, `KMER`, `PSSM`, `ESM2`, `ProtBert`, and models are `RF`, `SVM`, `XGB`. The shipped figures were drawn by an earlier plotting routine and use the lower-case names above.

## Provenance of the numbers

The plots in `figures/` are the original figures produced at the end of the project, renamed systematically. `results_summary.csv` was **reconstructed from those figures** (confusion-matrix cell counts read from each heatmap, ROC-AUC from each legend, rounded to 2 decimals as printed), and every row was checked against the known per-organism class totals. Accuracy, precision, recall and F1 are derived from the counts. `evaluate_model.py` and `train_model.py` write per-run metrics to `outputs/<organism>/<feature>/` (`*_metrics.json`, `summary_metrics.json`) and `outputs/pipeline_summary.json`; if you still have those from the original run, they are authoritative and should replace this file.
