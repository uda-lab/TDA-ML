from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import confusion_matrix, matthews_corrcoef, recall_score


def compute_recall_specificity_gmean_mcc(
    labels_gt: Sequence[int] | np.ndarray,
    labels_pred: Sequence[int] | np.ndarray,
) -> tuple[float, float, float, float]:
    """
    Return common binary metrics for 0=inlier and 1=outlier labels.

    Returns:
        recall, specificity, gmean, mcc
    """
    recall = recall_score(labels_gt, labels_pred, zero_division=0)
    tn, fp, _, _ = confusion_matrix(labels_gt, labels_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    gmean = float((recall * specificity) ** 0.5)
    mcc = float(matthews_corrcoef(labels_gt, labels_pred))
    return float(recall), float(specificity), gmean, mcc
