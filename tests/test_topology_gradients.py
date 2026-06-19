"""Gradient health for anisotropic distance matrices (issue #61)."""

import torch

from tda_ml.topology import compute_anisotropic_distance_matrix


def test_anisotropic_distance_diagonal_is_exactly_zero():
    batch_size, num_points = 1, 5
    points = torch.randn(batch_size, num_points, 2, dtype=torch.float32)
    params = torch.tensor(
        [
            [
                [1.0, 0.5, 0.0],
                [1.0, 0.5, 0.0],
                [1.0, 0.5, 0.0],
                [1.0, 0.5, 0.0],
                [1.0, 0.5, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    probs = torch.full((batch_size, num_points), 0.2, dtype=torch.float32)

    dist = compute_anisotropic_distance_matrix(
        points, params, probs=probs, symmetrize="max"
    )
    diag = dist.diagonal(dim1=-2, dim2=-1)
    assert torch.all(diag == 0.0)
    assert torch.all(dist[..., ~torch.eye(num_points, dtype=torch.bool)] > 0)


def test_anisotropic_distance_backward_finite_at_zero_diagonal():
    """Self-distance diagonal is zero; backward must not emit inf/nan gradients."""
    batch_size, num_points = 1, 5
    points = torch.randn(batch_size, num_points, 2, dtype=torch.float32)
    params = torch.tensor(
        [
            [
                [1.0, 0.5, 0.0],
                [1.0, 0.5, 0.0],
                [1.0, 0.5, 0.0],
                [1.0, 0.5, 0.0],
                [1.0, 0.5, 0.0],
            ]
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    probs = torch.full((batch_size, num_points), 0.2, dtype=torch.float32)

    dist = compute_anisotropic_distance_matrix(
        points, params, probs=probs, symmetrize="max"
    )
    assert torch.all(dist.diagonal(dim1=-2, dim2=-1) == 0.0)

    dist.sum().backward()
    assert params.grad is not None
    assert torch.isfinite(params.grad).all()


def test_anisotropic_distance_backward_no_sqrt_anomaly_on_diagonal():
    with torch.autograd.detect_anomaly():
        points = torch.randn(1, 8, 2, dtype=torch.float32, requires_grad=True)
        params = torch.rand(1, 8, 3, dtype=torch.float32, requires_grad=True)
        probs = torch.full((1, 8), 0.3, dtype=torch.float32)
        dist = compute_anisotropic_distance_matrix(
            points, params, probs=probs, symmetrize="max"
        )
        dist.sum().backward()
    assert torch.isfinite(points.grad).all()
    assert torch.isfinite(params.grad).all()
