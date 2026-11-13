# Plant — results gallery

Confusion matrices (rows = actual, columns = predicted; class 1 = interacting) and ROC curves for every feature × model combination. Numeric values: [`../../metrics/results_summary.csv`](../../metrics/results_summary.csv).

Test set shown in the plots: **513 negative / 104 positive** (617 pairs; majority-class accuracy = 0.831).


## AAC

| Model | Confusion matrix | ROC curve | Acc | Prec | Rec | F1 | AUC |
|---|---|---|---|---|---|---|---|
| RF | <img src="aac/rf_confusion_matrix.png" width="230"> | <img src="aac/rf_roc_curve.png" width="230"> | 0.961 | 0.988 | 0.779 | 0.871 | 1.00 |
| SVM | <img src="aac/svm_confusion_matrix.png" width="230"> | <img src="aac/svm_roc_curve.png" width="230"> | 0.951 | 0.987 | 0.721 | 0.833 | 0.98 |
| XGB | <img src="aac/xgb_confusion_matrix.png" width="230"> | <img src="aac/xgb_roc_curve.png" width="230"> | 0.974 | 0.989 | 0.856 | 0.918 | 1.00 |

## DPC

| Model | Confusion matrix | ROC curve | Acc | Prec | Rec | F1 | AUC |
|---|---|---|---|---|---|---|---|
| RF | <img src="dipeptide/rf_confusion_matrix.png" width="230"> | <img src="dipeptide/rf_roc_curve.png" width="230"> | 0.968 | 1.000 | 0.808 | 0.894 | 1.00 |
| SVM | <img src="dipeptide/svm_confusion_matrix.png" width="230"> | <img src="dipeptide/svm_roc_curve.png" width="230"> | 0.971 | 1.000 | 0.827 | 0.905 | 1.00 |
| XGB | <img src="dipeptide/xgb_confusion_matrix.png" width="230"> | <img src="dipeptide/xgb_roc_curve.png" width="230"> | 0.982 | 1.000 | 0.894 | 0.944 | 1.00 |

## K-mer

| Model | Confusion matrix | ROC curve | Acc | Prec | Rec | F1 | AUC |
|---|---|---|---|---|---|---|---|
| RF | <img src="kmer/rf_confusion_matrix.png" width="230"> | <img src="kmer/rf_roc_curve.png" width="230"> | 0.963 | 1.000 | 0.779 | 0.876 | 1.00 |
| SVM | <img src="kmer/svm_confusion_matrix.png" width="230"> | <img src="kmer/svm_roc_curve.png" width="230"> | 0.956 | 1.000 | 0.740 | 0.851 | 1.00 |
| XGB | <img src="kmer/xgb_confusion_matrix.png" width="230"> | <img src="kmer/xgb_roc_curve.png" width="230"> | 0.974 | 1.000 | 0.846 | 0.917 | 0.99 |

## PSSM

| Model | Confusion matrix | ROC curve | Acc | Prec | Rec | F1 | AUC |
|---|---|---|---|---|---|---|---|
| RF | <img src="pssm/rf_confusion_matrix.png" width="230"> | <img src="pssm/rf_roc_curve.png" width="230"> | 0.831 | 0.000 | 0.000 | 0.000 | 0.50 |
| SVM | <img src="pssm/svm_confusion_matrix.png" width="230"> | <img src="pssm/svm_roc_curve.png" width="230"> | 0.831 | 0.000 | 0.000 | 0.000 | 0.50 |
| XGB | <img src="pssm/xgb_confusion_matrix.png" width="230"> | <img src="pssm/xgb_roc_curve.png" width="230"> | 0.831 | 0.000 | 0.000 | 0.000 | 0.50 |

## ESM2

| Model | Confusion matrix | ROC curve | Acc | Prec | Rec | F1 | AUC |
|---|---|---|---|---|---|---|---|
| RF | <img src="esm2/rf_confusion_matrix.png" width="230"> | <img src="esm2/rf_roc_curve.png" width="230"> | 0.921 | 1.000 | 0.529 | 0.692 | 0.99 |
| SVM | <img src="esm2/svm_confusion_matrix.png" width="230"> | <img src="esm2/svm_roc_curve.png" width="230"> | 0.942 | 0.986 | 0.663 | 0.793 | 0.98 |
| XGB | <img src="esm2/xgb_confusion_matrix.png" width="230"> | <img src="esm2/xgb_roc_curve.png" width="230"> | 0.951 | 0.951 | 0.750 | 0.839 | 0.98 |

## ProtBert

| Model | Confusion matrix | ROC curve | Acc | Prec | Rec | F1 | AUC |
|---|---|---|---|---|---|---|---|
| RF | <img src="protbert/rf_confusion_matrix.png" width="230"> | <img src="protbert/rf_roc_curve.png" width="230"> | 0.937 | 0.971 | 0.644 | 0.775 | 0.98 |
| SVM | <img src="protbert/svm_confusion_matrix.png" width="230"> | <img src="protbert/svm_roc_curve.png" width="230"> | 0.963 | 0.966 | 0.808 | 0.880 | 0.99 |
| XGB | <img src="protbert/xgb_confusion_matrix.png" width="230"> | <img src="protbert/xgb_roc_curve.png" width="230"> | 0.951 | 0.951 | 0.750 | 0.839 | 0.98 |
