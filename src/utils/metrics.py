import numpy as np
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    cohen_kappa_score,
    confusion_matrix,
    roc_curve
)


def compute_metrics(y_true, probs, threshold=None):
    """
    Compute classification metrics.

    If threshold is None:
        uses ROC Youden index (ONLY for validation use)
    """

    y_true = np.array(y_true)
    probs = np.array(probs)

    # -----------------------------
    # Threshold selection
    # -----------------------------
    if threshold is None:
        fpr, tpr, thresholds = roc_curve(y_true, probs)
        idx = np.argmax(tpr - fpr)
        threshold = thresholds[idx] if idx < len(thresholds) else 0.5

    preds = (probs > threshold).astype(int)

    # -----------------------------
    # Confusion matrix (safe)
    # -----------------------------
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # -----------------------------
    # Metrics
    # -----------------------------
    results = {
        "accuracy": accuracy_score(y_true, preds),
        "sensitivity": recall_score(y_true, preds, zero_division=0),
        "specificity": tn / (tn + fp + 1e-8),
        "precision": precision_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
        "auc": roc_auc_score(y_true, probs),
        "kappa": cohen_kappa_score(y_true, preds),
        "threshold": threshold,   # ✅ important
    }

    return results