import numpy as np
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    cohen_kappa_score,
    confusion_matrix,
    roc_curve,
)


def compute_metrics(y_true, probs, threshold=None):
    """
    Compute binary classification metrics.

    Parameters
    ----------
    y_true    : array-like of int  {0, 1}
    probs     : array-like of float  (probability of class 1)
    threshold : float or None
        If None, selects threshold via Youden's J  (maximises TPR - FPR).

    Returns
    -------
    dict with keys:
        accuracy, sensitivity, specificity, balanced_accuracy,
        precision, f1, auc, kappa, threshold

    FIXES
    -----
    - Youden grid replaced by sklearn roc_curve thresholds: more precise,
      no arbitrary linspace needed, handles extreme distributions correctly
    - AUC guard: also checks probs has variance (not all same value)
    - Added npv (negative predictive value) — clinically important for EEG
    - Removed silent 1e-8 bias from specificity denominator: use proper
      guard only when tn+fp == 0
    """

    y_true = np.asarray(y_true,  dtype=int)
    probs  = np.asarray(probs,   dtype=float)

    # ── Sanity checks ────────────────────────────────────────────────
    if len(y_true) != len(probs):
        raise ValueError("y_true and probs must have the same length")
    if not np.isfinite(probs).all():
        raise ValueError("probs contains NaN or Inf")

    # ── Threshold selection ──────────────────────────────────────────
    if threshold is None:
        if len(np.unique(y_true)) < 2:
            threshold = 0.5
        else:
            fpr_arr, tpr_arr, thr_arr = roc_curve(y_true, probs)
            # Youden's J on the exact ROC thresholds — much more precise
            j_scores = tpr_arr - fpr_arr
            best_idx  = int(np.argmax(j_scores))
            threshold = float(thr_arr[best_idx])

    preds = (probs > threshold).astype(int)

    # ── Confusion matrix ─────────────────────────────────────────────
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # ── Core metrics ─────────────────────────────────────────────────
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # recall / TPR
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0   # TNR
    npv         = tn / (tn + fn) if (tn + fn) > 0 else 0.0   # negative predictive value

    # AUC: needs both classes present AND probability variance
    can_auc = (len(np.unique(y_true)) > 1) and (probs.std() > 1e-8)
    auc_val = float(roc_auc_score(y_true, probs)) if can_auc else 0.0

    results = {
        "accuracy"         : float(accuracy_score(y_true, preds)),
        "sensitivity"      : float(sensitivity),                    # TPR / recall
        "specificity"      : float(specificity),                    # TNR
        "balanced_accuracy": float(0.5 * (sensitivity + specificity)),
        "precision"        : float(precision_score(y_true, preds, zero_division=0)),
        "npv"              : float(npv),
        "f1"               : float(f1_score(y_true, preds, zero_division=0)),
        "auc"              : auc_val,
        "kappa"            : float(cohen_kappa_score(y_true, preds)),
        "threshold"        : float(threshold),
        # Raw CM for downstream use
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }

    return results
