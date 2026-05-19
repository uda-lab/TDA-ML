import torch
import unittest
from tda_ml.models import AnisotropicOutlierClassifier

class TestAnisotropicOutlierClassifier(unittest.TestCase):
    def test_forward_pass(self):
        """
        Tests if the AnisotropicOutlierClassifier model can be initialized
        and can perform a forward pass with dummy data.
        """
        batch_size = 4
        num_points = 100
        point_dim = 2

        model = AnisotropicOutlierClassifier(point_dim=point_dim)

        dummy_input = torch.randn(batch_size, num_points, point_dim)

        outlier_logits, ellipse_params = model(dummy_input)

        self.assertEqual(outlier_logits.shape, (batch_size, num_points, 1))
        self.assertEqual(ellipse_params.shape, (batch_size, num_points, 3))

        self.assertIsInstance(outlier_logits, torch.Tensor)
        self.assertIsInstance(ellipse_params, torch.Tensor)

if __name__ == '__main__':
    unittest.main()
