"""Metrics and publication-friendly evaluation plots."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve
)

def evaluate_predictions(y_true, y_pred, y_score):
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    metrics["roc_auc"] = (
        float(roc_auc_score(y_true, y_score))
        if len(np.unique(y_true)) == 2 else float("nan")
    )
    return metrics

def save_evaluation(y_true, y_pred, y_score, output_dir, prefix):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_predictions(y_true, y_pred, y_score)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=["Non-interacting", "Interacting"]).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title(f"Confusion matrix: {prefix}")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_confusion_matrix.png", dpi=300)
    plt.close(fig)

    if len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        ax.set(xlabel="False positive rate", ylabel="True positive rate",
               title=f"ROC curve: {prefix}")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(output_dir / f"{prefix}_roc_curve.png", dpi=300)
        plt.close(fig)

    with (output_dir / f"{prefix}_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, allow_nan=True)
    return metrics
