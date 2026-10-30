"""Validation threshold selection and fixed-threshold reporting."""
import numpy as np
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score


def threshold_mcc(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores)
    if not len(labels) or not np.isfinite(scores).all() or set(labels) != {0, 1}:
        raise ValueError("Threshold selection needs finite scores and both validation classes.")
    order = np.argsort(scores, kind="stable")
    positives = np.r_[0, np.cumsum(labels[order])]
    candidates = np.unique(np.r_[0., scores, np.nextafter(scores.max(), np.inf)])
    below = np.searchsorted(scores[order], candidates, side="left")
    fn, tn = positives[below], below - positives[below]
    tp, fp = positives[-1] - fn, len(labels) - below - positives[-1] + fn
    denominator = np.sqrt((tp + fp).astype(float) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(tp * tn - fp * fn, denominator, out=np.zeros_like(denominator), where=denominator > 0)
    tied = np.flatnonzero(mcc == mcc.max())
    return float(candidates[tied[np.argmin(np.abs(candidates[tied] - .5))]])


def report(labels, scores, threshold):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if len(labels) != len(scores) or not len(labels) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Expected one finite probability per labelled record.")
    predicted, positive = scores >= threshold, labels == 1
    tp, tn = int((predicted & positive).sum()), int((~predicted & ~positive).sum())
    fp, fn = int((predicted & ~positive).sum()), int((~predicted & positive).sum())
    both = len(set(labels)) == 2
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    clipped = np.clip(scores, 1e-7, 1 - 1e-7)
    return {"n": len(labels), "positive_prevalence": float(positive.mean()),
            "auroc": float(roc_auc_score(labels, scores)) if both else None,
            "average_precision": float(average_precision_score(labels, scores)) if both else None,
            "mcc": float(matthews_corrcoef(labels, predicted)) if both else None,
            "precision": precision, "recall": recall, "specificity": specificity,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
            "balanced_accuracy": (recall + specificity) / 2 if both else None,
            "bce": float(-(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)).mean()),
            "brier": float(np.mean((scores - labels) ** 2)), "threshold": float(threshold),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}
