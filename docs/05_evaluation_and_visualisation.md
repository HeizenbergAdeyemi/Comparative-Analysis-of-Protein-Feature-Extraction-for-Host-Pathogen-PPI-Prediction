# Part 5 — Evaluation and visualisation

## Training and testing procedure

For each organism group and each feature set, every model is trained on the training portion and evaluated on a held-out test portion. The test-set totals visible in the saved confusion matrices are:

| Organism | Negative | Positive | Total | Majority-class accuracy |
|---|---|---|---|---|
| `bacillus` | 1893 | 508 | 2401 | 0.788 |
| `human_hiv1` | 355 | 2483 | 2838 | 0.875 |
| `human_sars_cov2` | 126 | 64 | 190 | 0.663 |
| `plant` | 513 | 104 | 617 | 0.831 |
| `plasmodium` | 146 | 147 | 293 | 0.502 |

> **TODO (author):** state the split ratio, whether it is stratified, the random seed, and whether sequences or pairs were split (this matters for leakage; see Part 7).

## Confusion matrices

Layout used in every saved figure: **rows = actual class, columns = predicted class, class 0 = non-interacting, class 1 = interacting**, so the cells are `[[TN, FP], [FN, TP]]`. Drawn as an annotated seaborn heatmap with the `Blues` colormap and titled `"{MODEL} - {organism} ({feature})"`.

## ROC curves and AUC

The ROC curve plots true-positive rate against false-positive rate across decision thresholds, with the dashed diagonal as the chance line. AUC summarises it as the probability that a random positive is scored above a random negative. It needs continuous scores (probabilities or decision values) and is insensitive to the chosen threshold. Figures are titled `"ROC - {MODEL} - {organism} ({feature})"` with the AUC (2 decimals) in the legend. An AUC of exactly 0.50 means no ranking information.

## Accuracy, precision, recall and F1

With `TP, TN, FP, FN` from the confusion matrix:

| Metric | Formula | Guard needed |
|---|---|---|
| Accuracy | (TP + TN) / (TP + TN + FP + FN) | none |
| Precision | TP / (TP + FP) | zero when the model never predicts positive |
| Recall | TP / (TP + FN) | zero if there are no positives |
| F1 | 2 · P · R / (P + R) | zero when P + R = 0 |

Use `zero_division=0` in scikit-learn so a collapsed model reports 0 rather than raising (Part 6).

## Reading the results correctly

- **Always compare accuracy with the majority-class baseline.** Predicting everything negative scores 0.788 on Bacillus; everything positive scores 0.875 on HIV-1.
- **Collapsed models can look good on some metrics.** The PSSM models for HIV-1 and Plasmodium predict all-positive, which gives F1 ≈ 0.93 and 0.67 on those sets while AUC is 0.50. AUC and the confusion matrix expose this; F1 alone does not.
- **AUC ≈ 1.00 is a warning, not a result.** It occurs for AAC/DPC/K-mer on HIV-1 and SARS-CoV-2 (Part 7).

## Results organisation and saving

```
results/
├── metrics/results_summary.csv
├── summary/                               # aggregate figures
└── figures/<organism>/<feature>/
    ├── rf_confusion_matrix.png   ├── rf_roc_curve.png
    ├── svm_confusion_matrix.png  ├── svm_roc_curve.png
    └── xgb_confusion_matrix.png  └── xgb_roc_curve.png
```

Organism identifiers: `bacillus`, `human_hiv1`, `human_sars_cov2`, `plant`, `plasmodium`. Feature identifiers: `aac`, `dipeptide`, `kmer`, `pssm`, `esm2`, `protbert`. Each organism folder has a `README.md` gallery with all plots and metrics.

> **TODO (author):** if `evaluate_model.py` writes its own metrics file, point to it here and note that `results_summary.csv` in this repository was reconstructed from the saved plots (see `results/README.md`).
