"""Regression tests for training.topo_distance_mode (mahalanobis vs ellphi)."""

import unittest
import torch

from tda_ml.losses import (
    DISTANCE_MODE_ELLPHI,
    DISTANCE_MODE_MAHALANOBIS,
    compute_topo_distance_matrix,
    mahalanobis_distance_matrix_batched,
    normalize_topo_distance_mode,
)


class TestTopoDistanceMode(unittest.TestCase):
    def test_normalize_aliases(self):
        self.assertEqual(normalize_topo_distance_mode("Mahalanobis"), DISTANCE_MODE_MAHALANOBIS)
        self.assertEqual(normalize_topo_distance_mode("ellphi"), DISTANCE_MODE_ELLPHI)

    def test_mahalanobis_shape(self):
        torch.manual_seed(0)
        b, n = 2, 9
        pt = torch.randn(b, n, 2)
        par = torch.randn(b, n, 3)
        par[:, :, 0:2] = par[:, :, 0:2].abs() + 0.1
        d = compute_topo_distance_matrix(pt, par, distance_mode="mahalanobis")
        self.assertEqual(d.shape, (b, n, n))

    def test_ellphi_finite_and_grad(self):
        torch.manual_seed(0)
        b, n = 1, 7
        pt = torch.randn(b, n, 2, requires_grad=True)
        par = torch.randn(b, n, 3)
        par[:, :, 0:2] = par[:, :, 0:2].abs() + 0.15
        par.requires_grad_(True)
        d_e = compute_topo_distance_matrix(pt, par, distance_mode="ellphi", ellphi_backend="auto")
        d_m = mahalanobis_distance_matrix_batched(pt, par)
        self.assertEqual(d_e.shape, (b, n, n))
        self.assertTrue(torch.isfinite(d_e).all())
        loss = d_e.sum()
        loss.backward()
        self.assertIsNotNone(pt.grad)
        self.assertIsNotNone(par.grad)
        # Forward definitions differ; should not be identical in general
        self.assertGreater((d_e - d_m).abs().mean().item(), 1e-6)

    def test_normalize_invalid_mode_raises_value_error(self):
        with self.assertRaises(ValueError):
            normalize_topo_distance_mode("unknown-mode")

    def test_mahalanobis_extreme_params_remain_finite(self):
        """極小/極大軸長でも距離行列が非有限値にならないことを確認。"""
        b, n = 1, 5
        pt = torch.tensor(
            [[[1e-9, -1e-9], [1.0, 2.0], [3.0, -1.0], [0.5, 0.1], [-2.0, 1.5]]],
            dtype=torch.float64,
        )
        par = torch.tensor(
            [[[1e-8, 1e8, 0.0],
              [2e-8, 5e7, 0.3],
              [1e7, 2e-7, -0.4],
              [5e-8, 8e7, 0.7],
              [2e7, 3e-8, -1.0]]],
            dtype=torch.float64,
        )
        d = compute_topo_distance_matrix(pt, par, distance_mode="mahalanobis")
        self.assertEqual(d.shape, (b, n, n))
        self.assertTrue(torch.isfinite(d).all())


if __name__ == "__main__":
    unittest.main()
