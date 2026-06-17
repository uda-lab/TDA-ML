import torch

from tda_ml.topology import compute_anisotropic_distance_matrix


def test_anisotropic_distance_backward_finite_at_zero_diagonal():
  """Diagonal distances are zero; backward must not emit inf/nan gradients."""
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
  assert torch.all(dist.diagonal(dim1=-2, dim2=-1) > 0)

  dist.sum().backward()
  assert params.grad is not None
  assert torch.isfinite(params.grad).all()
