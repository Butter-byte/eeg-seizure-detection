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


def compute_metrics(y_true, probs):
    """
    Computes classification metrics using optimal ROC threshold
    """

    y_true = np.array(y_true)
    probs = np.array(probs)

    # Optimal threshold (Youden Index)
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    thr = thresholds[np.argmax(tpr - fpr)]

    preds = (probs > thr).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()

    return {
        "accuracy": accuracy_score(y_true, preds),
        "sensitivity": recall_score(y_true, preds),
        "specificity": tn / (tn + fp + 1e-8),
        "precision": precision_score(y_true, preds),
        "f1": f1_score(y_true, preds),
        "auc": roc_auc_score(y_true, probs),
        "kappa": cohen_kappa_score(y_true, preds),
    }