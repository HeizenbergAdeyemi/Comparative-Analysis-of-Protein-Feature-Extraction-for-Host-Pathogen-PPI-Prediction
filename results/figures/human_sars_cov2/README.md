# Human–SARS-CoV-2 — results gallery

Confusion matrices (rows = actual, columns = predicted; class 1 = interacting) and ROC curves for every feature × model combination. Numeric values: [`../../metrics/results_summary.csv`](../../metrics/results_summary.csv).

Test set shown in the plots: **126 negative / 64 positive** (190 pairs; majority-class accuracy = 0.663).


## AAC

| Model | Confusion matrix | ROC curve | Acc | Prec | Rec | F1 | AUC |
|---|---|---|---|---|---|---|---|
| RF | <img src="aac/rf_confusion_matrix.png" width="230"> | <img src="aac/rf_roc_curve.png" width="230"> | 1.000 | 1.000 | 1.000 | 1.000 | 1.00 |
| SVM | <img src="aac/svm_confusion_matrix.png" width="230"> | <img src="aac/svm_roc_curve.png" width="230"> | 0.984 | 0.984 | 0.969 | 0.976 | 1.00 |
| XGB | <img src="aac/xgb_confusion_matrix.png" width="230"> | <img src="aac/xgb_roc_curve.png" width="230"> | 0.984 | 0.955 | 1.000 | 0.977 | 1.00 |

## DPC

| Model | Confusion matrix | ROC curve | Acc | Prec | Rec | F1 | AUC |
|---|---|---|---|---|---|---|---|
| RF | <img src="dipeptide/rf_confusion_matrix.png" width="230"> | <img src="dipeptide/rf_roc_curve.png" width="230"> | 1.000 | 1.000 | 1.000 | 1.000 | 1.00 |
| SVM | <img src="dipeptide/svm_confusion_matrix.png" width="230"> | <img src="dipeptide/svm_roc_curve.png" width="230"> | 0.984 | 1.000 | 0.953 | 0.976 | 1.00 |
| XGB | <img src="dipeptide/xgb_confusion_matrix.png" width="230"> | <img src="dipeptide/xgb_roc_curve.png" width="230"> | 1.000 | 1.000 | 1.000 | 1.000 | 1.00 |

## K-mer

| Model | Confusion matrix | ROC curve | Acc | Prec | Rec | F1 | AUC |
|---|---|---|---|---|---|---|---|
| RF | <img src="kmer/rf_confusion_matrix.png" width="230"> | <img src="kmer/rf_roc_curve.png" width="230"> | 0.995 | 1.000 | 0.984 | 0.992 | 1.00 |
| SVM | <img src="kmer/svm_confusion_matrix.png" width="230"> | <img src="kmer/svm_roc_curve.png" width="230"> | 0.974 | 1.000 | 0.922 | 0.959 | 1.00 |
| XGB | <img src="kmer/xgb_confusion_matrix.png" width="230"> | <img src="kmer/xgb_roc_curve.png" width="230"> | 0.984 | 0.969 | 0.984 | 0.977 | 1.00 |

## PSSM

| Model | Confusion matrix | ROC curve | Acc | Prec | Rec | F1 | AUC |
|---|---|---|---|---|---|---|---|
| RF | <img src="pssm/rf_confusion_matrix.png" width="230"> | <img src="pssm/rf_roc_curve.png" width="230"> | 0.663 | 0.000 | 0.000 | 0.000 | 0.50 |
| SVM | <img src="pssm/svm_confusion_matrix.png" width="230"> | <img src="pssm/svm_roc_curve.png" width="230"> | 0.663 | 0.000 | 0.000 | 0.000 | 0.50 |
| XGB | <img src="pssm/xgb_confusion_matrix.png" width="230"> | <img src="pssm/xgb_roc_curve.png" width="230"> | 0.663 | 0.000 | 0.000 | 0.000 | 0.50 |

## ESM2

| Model | Confusion matrix | ROC curve | Acc | Prec | Rec | F1 | AUC |
|---|---|---|---|---|---|---|---|
| RF | <img src="esm2/rf_confusion_matrix.png" width="230"> | <img src="esm2/rf_roc_curve.png" width="230"> | 0.837 | 0.770 | 0.734 | 0.752 | 0.93 |
| SVM | <img src="esm2/svm_confusion_matrix.png" width="230"> | <img src="esm2/svm_roc_curve.png" width="230"> | 0.868 | 0.791 | 0.828 | 0.809 | 0.94 |
| XGB | <img src="esm2/xgb_confusion_matrix.png" width="230"> | <img src="esm2/xgb_roc_curve.png" width="230"> | 0.895 | 0.824 | 0.875 | 0.848 | 0.93 |

## ProtBert

| Model | Confusion matrix | ROC curve | Acc | Prec | Rec | F1 | AUC |
|---|---|---|---|---|---|---|---|
| RF | <img src="protbert/rf_confusion_matrix.png" width="230"> | <img src="protbert/rf_roc_curve.png" width="230"> | 0.868 | 0.855 | 0.734 | 0.790 | 0.95 |
| SVM | <img src="protbert/svm_confusion_matrix.png" width="230"> | <img src="protbert/svm_roc_curve.png" width="230"> | 0.916 | 0.900 | 0.844 | 0.871 | 0.96 |
| XGB | <img src="protbert/xgb_confusion_matrix.png" width="230"> | <img src="protbert/xgb_roc_curve.png" width="230"> | 0.889 | 0.841 | 0.828 | 0.835 | 0.95 |
