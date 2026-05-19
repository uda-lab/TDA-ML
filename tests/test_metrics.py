import unittest

import numpy as np

from tda_ml.metrics import compute_recall_specificity_gmean_mcc


class TestMetrics(unittest.TestCase):
    def test_compute_recall_specificity_gmean_mcc(self):
        y_true = [0, 0, 0, 1, 1, 1]
        y_pred = [0, 1, 0, 1, 0, 1]

        recall, specificity, gmean, mcc = compute_recall_specificity_gmean_mcc(
            y_true, y_pred
        )

        self.assertAlmostEqual(recall, 2.0 / 3.0, places=6)
        self.assertAlmostEqual(specificity, 2.0 / 3.0, places=6)
        self.assertAlmostEqual(gmean, 2.0 / 3.0, places=6)
        self.assertAlmostEqual(mcc, 1.0 / 3.0, places=6)

    def test_compute_metrics_all_inlier_zero_division_path(self):
        """positive class 不在時は recall=0 を返し、ゼロ除算を起こさない。"""
        y_true = np.array([0, 0, 0, 0], dtype=int)
        y_pred = np.array([0, 0, 0, 0], dtype=int)

        recall, specificity, gmean, mcc = compute_recall_specificity_gmean_mcc(
            y_true, y_pred
        )
        self.assertEqual(recall, 0.0)
        self.assertEqual(specificity, 1.0)
        self.assertEqual(gmean, 0.0)
        self.assertEqual(mcc, 0.0)

    def test_compute_metrics_nan_inf_labels_raise_value_error(self):
        """NaN/Inf を含むラベルは sklearn 側で ValueError として扱う。"""
        y_pred = np.array([0, 1, 0, 1], dtype=int)
        with self.assertRaises(ValueError):
            compute_recall_specificity_gmean_mcc(np.array([0, 1, np.nan, 0]), y_pred)
        with self.assertRaises(ValueError):
            compute_recall_specificity_gmean_mcc(np.array([0, 1, np.inf, 0]), y_pred)


if __name__ == "__main__":
    unittest.main()
