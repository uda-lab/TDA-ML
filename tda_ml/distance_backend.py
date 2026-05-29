"""
Distance-matrix backends: differentiable anisotropic Mahalanobis and ellphi tangency distance.

The differentiable ellphi path connects ``ellphi.grad`` functions
(``coef_from_cov_grad`` / ``pdist_tangency_grad``) through a PyTorch
``autograd.Function`` wrapper (upstream: https://github.com/t-uda/ellphi).
"""

from __future__ import annotations

import warnings

import numpy as np
import torch

from tda_ml.ellphi_torch import (
    _has_ellphi_grad_api,
    ellipse_params_to_centers_cov,
    pdist_tangency_matrix_differentiable,
)
from tda_ml.geometry import ellipse_params_to_centers_cov_numpy
from tda_ml.topology import compute_anisotropic_distance_matrix

try:
    import ellphi
except ImportError:
    ellphi = None  # type: ignore[misc, assignment]

try:
    from scipy.spatial.distance import squareform
except ImportError:
    squareform = None  # type: ignore[misc, assignment]

_ELLPHI_PROB_WARNED = False


def compute_ellphi_distance_matrix_np(points_np: np.ndarray, params_np: np.ndarray) -> np.ndarray:
    """
    Compute ellphi tangency distance matrix ``(N, N)`` in ``float64``.

    ``params`` has shape ``(N, 3) = [a, b, theta]`` at the points in ``points_np``.
    """
    if ellphi is None:
        raise ImportError("Distance backend 'ellphi' requires the ellphi package.")
    if squareform is None:
        raise ImportError("scipy is required to expand condensed pairwise distances.")

    centers, covs = ellipse_params_to_centers_cov_numpy(points_np, params_np)

    ec = ellphi.EllipseCloud.from_cov(
        np.ascontiguousarray(centers),
        np.ascontiguousarray(covs),
    )
    dist = ec.pdist_tangency()
    if dist.ndim == 1:
        dist = squareform(dist)
    return np.asarray(dist, dtype=np.float64)


def compute_distance_matrix_batch(
    points: torch.Tensor,
    params: torch.Tensor,
    *,
    probs: torch.Tensor | None,
    symmetrize: str,
    backend: str,
    ellphi_differentiable: bool = True,
) -> torch.Tensor:
    """
    Compute batched distance matrices with shape ``(B, N, N)``.

    backend:
      - ``mahalanobis``: ``compute_anisotropic_distance_matrix`` (differentiable)
      - ``ellphi``: tangency distance. If ``ellphi_differentiable=True`` and
        ``ellphi.grad`` is available, gradients flow to centers/covariances
        (therefore to ellipse parameters).

    For ``ellphi``, ``probs``-based weighting is currently unsupported and ignored.
    """
    b = backend.lower().strip()
    if b not in ("mahalanobis", "ellphi"):
        raise ValueError(f"Unknown distance backend: {backend!r}. Use 'mahalanobis' or 'ellphi'.")

    if b == "mahalanobis":
        return compute_anisotropic_distance_matrix(
            points, params, probs=probs, symmetrize=symmetrize
        )

    global _ELLPHI_PROB_WARNED
    if probs is not None and not _ELLPHI_PROB_WARNED:
        warnings.warn(
            "distance_backend='ellphi' ignores outlier-probability weighting because it is not implemented yet.",
            UserWarning,
            stacklevel=2,
        )
        _ELLPHI_PROB_WARNED = True

    use_torch = ellphi_differentiable and _has_ellphi_grad_api()
    if ellphi_differentiable and not use_torch:
        warnings.warn(
            "ellphi.grad (coef_from_cov_grad / pdist_tangency_grad) is unavailable; "
            "falling back to the NumPy non-differentiable path. Please update ellphi.",
            UserWarning,
            stacklevel=2,
        )

    batch_size = points.shape[0]
    mats: list[torch.Tensor] = []
    points_np = None
    params_np = None
    if not use_torch:
        # Convert once per batch to reduce repeated CPU/NumPy transfer overhead.
        points_np = points.detach().cpu().numpy()
        params_np = params.detach().cpu().numpy()

    for i in range(batch_size):
        if use_torch:
            c, cov = ellipse_params_to_centers_cov(points[i], params[i])
            mats.append(pdist_tangency_matrix_differentiable(c, cov))
        else:
            dm = compute_ellphi_distance_matrix_np(
                points_np[i],
                params_np[i],
            )
            mats.append(torch.from_numpy(dm).to(device=points.device, dtype=points.dtype))
    return torch.stack(mats, dim=0)
