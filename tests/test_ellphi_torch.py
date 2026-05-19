"""Unit tests for ellphi_torch helper utilities."""

import unittest

import numpy as np

from tda_ml.ellphi_torch import _condensed_gradient_from_full


class TestEllphiTorchHelpers(unittest.TestCase):
    def test_condensed_gradient_matches_expected_order(self):
        g = np.array(
            [
                [0.0, 1.0, 2.0, 3.0],
                [10.0, 0.0, 12.0, 13.0],
                [20.0, 21.0, 0.0, 23.0],
                [30.0, 31.0, 32.0, 0.0],
            ],
            dtype=np.float64,
        )
        # scipy condensed order for n=4: (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
        expected = np.array(
            [
                0.5 * (1.0 + 10.0),
                0.5 * (2.0 + 20.0),
                0.5 * (3.0 + 30.0),
                0.5 * (12.0 + 21.0),
                0.5 * (13.0 + 31.0),
                0.5 * (23.0 + 32.0),
            ],
            dtype=np.float64,
        )
        actual = _condensed_gradient_from_full(g)
        self.assertTrue(np.allclose(actual, expected))


if __name__ == "__main__":
    unittest.main()
