from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import confusion_matrix, matthews_corrcoef, recall_score

from tda_ml.persistence import compute_w_distance


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


def compute_recall_specificity_gmean_mcc_wdist(
    labels_gt: Sequence[int] | np.ndarray,
    labels_pred: Sequence[int] | np.ndarray,
    *,
    points: np.ndarray | None = None,
    gt_inliers: np.ndarray | None = None,
    empty_pred_wdist: float = 9.99,
) -> tuple[float, float, float, float, float]:
    """
    Return the four classification metrics plus W-Dist.

    W-Dist is computed only when ``points`` and ``gt_inliers`` are provided.
    If every predicted label is outlier (no predicted inliers), returns
    ``empty_pred_wdist`` (default ``9.99``), distinct from
    :func:`tda_ml.persistence.compute_w_distance` returning ``999.0`` for an empty point cloud.
    """
    recall, specificity, gmean, mcc = compute_recall_specificity_gmean_mcc(
        labels_gt, labels_pred
    )
    w_dist = 0.0
    if points is not None and gt_inliers is not None:
        pred_inliers = points[np.asarray(labels_pred) == 0]
        if len(pred_inliers) > 0:
            w_dist = float(compute_w_distance(pred_inliers, gt_inliers))
        else:
            w_dist = float(empty_pred_wdist)
    return recall, specificity, gmean, mcc, w_dist
