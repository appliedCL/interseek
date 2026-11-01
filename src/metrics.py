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
