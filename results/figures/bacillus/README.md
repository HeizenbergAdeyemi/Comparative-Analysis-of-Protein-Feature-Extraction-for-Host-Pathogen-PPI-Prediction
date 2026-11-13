# Bacillus — results gallery

Confusion matrices (rows = actual, columns = predicted; class 1 = interacting) and ROC curves for every feature × model combination. Numeric values: [`../../metrics/results_summary.csv`](../../metrics/results_summary.csv).

Test set shown in the plots: **1893 negative / 508 positive** (2401 pairs; majority-class accuracy = 0.788).


## AAC

| Model | Confusion matrix | ROC curve | Acc | Prec | Rec | F1 | AUC |
|---|---|---|---|---|---|---|---|
| RF | <img src="aac/rf_confusion_matrix.png" width="230"> | <img src="aac/rf_roc_curve.png" width="230"> | 0.930 | 0.943 | 0.713 | 0.812 | 0.95 |
| SVM | <img src="aac/svm_confusion_matrix.png" width="230"> | <img src="aac/svm_roc_curve.png" width="230"> | 0.843 | 0.739 | 0.402 | 0.520 | 0.88 |
| XGB | <img src="aac/xgb_confusion_matrix.png" width="230"> | <img src="aac/xgb_roc_curve.png" width="230"> | 0.931 | 0.849 | 0.819 | 0.834 | 0.96 |

## DPC

| Model | Confusion matrix | ROC curve | Acc | Prec | Rec | F1 | AUC |
|---|---|---|---|---|---|---|---|
| RF | <img src="dipeptide/rf_confusion_matrix.png" width="230"> | <img src="dipeptide/rf_roc_curve.png" width="230"> | 0.935 | 0.958 | 0.722 | 0.824 | 0.97 |
| SVM | <img src="dipeptide/svm_confusion_matrix.png" width="230"> | <img src="dipeptide/svm_roc_curve.png" width="230"> | 0.909 | 0.798 | 0.762 | 0.779 | 0.94 |
| XGB | <img src="dipeptide/xgb_confusion_matrix.png" width="230"> | <img src="dipeptide/xgb_roc_curve.png" width="230"> | 0.939 | 0.868 | 0.841 | 0.854 | 0.97 |

## K-mer

| Model | Confusion matrix | ROC curve | Acc | Prec | Rec | F1 | AUC |
|---|---|---|---|---|---|---|---|
| RF | <img src="kmer/rf_confusion_matrix.png" width="230"> | <img src="kmer/rf_roc_curve.png" width="230"> | 0.939 | 0.962 | 0.742 | 0.838 | 0.96 |
| SVM | <img src="kmer/svm_confusion_matrix.png" width="230"> | <img src="kmer/svm_roc_curve.png" width="230"> | 0.930 | 0.865 | 0.791 | 0.826 | 0.96 |
| XGB | <img src="kmer/xgb_confusion_matrix.png" width="230"> | <img src="kmer/xgb_roc_curve.png" width="230"> | 0.940 | 0.889 | 0.819 | 0.852 | 0.96 |

## PSSM

| Model | Confusion matrix | ROC curve | Acc | Prec | Rec | F1 | AUC |
|---|---|---|---|---|---|---|---|
| RF | <img src="pssm/rf_confusion_matrix.png" width="230"> | <img src="pssm/rf_roc_curve.png" width="230"> | 0.788 | 0.000 | 0.000 | 0.000 | 0.50 |
| SVM | <img src="pssm/svm_confusion_matrix.png" width="230"> | <img src="pssm/svm_roc_curve.png" width="230"> | 0.788 | 0.000 | 0.000 | 0.000 | 0.50 |
| XGB | <img src="pssm/xgb_confusion_matrix.png" width="230"> | <img src="pssm/xgb_roc_curve.png" width="230"> | 0.788 | 0.000 | 0.000 | 0.000 | 0.50 |

## ESM2

| Model | Confusion matrix | ROC curve | Acc | Prec | Rec | F1 | AUC |
|---|---|---|---|---|---|---|---|
| RF | <img src="esm2/rf_confusion_matrix.png" width="230"> | <img src="esm2/rf_roc_curve.png" width="230"> | 0.932 | 0.960 | 0.709 | 0.815 | 0.93 |
| SVM | <img src="esm2/svm_confusion_matrix.png" width="230"> | <img src="esm2/svm_roc_curve.png" width="230"> | 0.857 | 0.758 | 0.474 | 0.584 | 0.90 |
| XGB | <img src="esm2/xgb_confusion_matrix.png" width="230"> | <img src="esm2/xgb_roc_curve.png" width="230"> | 0.926 | 0.886 | 0.748 | 0.811 | 0.94 |

## ProtBert

| Model | Confusion matrix | ROC curve | Acc | Prec | Rec | F1 | AUC |
|---|---|---|---|---|---|---|---|
| RF | <img src="protbert/rf_confusion_matrix.png" width="230"> | <img src="protbert/rf_roc_curve.png" width="230"> | 0.929 | 0.945 | 0.707 | 0.809 | 0.94 |
| SVM | <img src="protbert/svm_confusion_matrix.png" width="230"> | <img src="protbert/svm_roc_curve.png" width="230"> | 0.876 | 0.781 | 0.577 | 0.664 | 0.93 |
| XGB | <img src="protbert/xgb_confusion_matrix.png" width="230"> | <img src="protbert/xgb_roc_curve.png" width="230"> | 0.928 | 0.870 | 0.778 | 0.821 | 0.95 |
