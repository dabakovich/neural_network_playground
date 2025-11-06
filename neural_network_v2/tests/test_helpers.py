import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

# Add the parent directory to the path so we can import from neural_network_v2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_network_v2.helpers import build_layers, calculate_loss
from neural_network_v2.types import LayerConfig


class TestHelpers(unittest.TestCase):
    @patch("neural_network_v2.helpers.rg")
    def test_build_layers(self, mock_rg):
        """Test build_layers function"""

        # Mock random generator to return deterministic values based on shape
        def mock_random(shape):
            if shape == (2, 2):  # Layer 1: 2×2 weights
                return np.array([[0.1, 0.2], [0.3, 0.4]])
            elif shape == (1, 2):  # Layer 2: 1×2 weights
                return np.array([[0.5, 0.6]])
            else:
                return np.random.random(shape)

        mock_rg.random = MagicMock(side_effect=mock_random)

        layer_configs: list[LayerConfig] = [
            {"input_size": 2, "output_size": 2},
            {"input_size": 2, "output_size": 1},
        ]

        layers = build_layers(layer_configs)

        # Check we got the right number of layers
        self.assertEqual(len(layers), 2)

        # Check first layer structure (numpy array)
        first_layer = layers[0]
        self.assertIsInstance(first_layer, np.ndarray)
        self.assertEqual(first_layer.shape, (2, 3))  # (output_size, input_size + bias)

        # Check second layer structure
        second_layer = layers[1]
        self.assertIsInstance(second_layer, np.ndarray)
        self.assertEqual(second_layer.shape, (1, 3))  # (output_size, input_size + bias)

    def test_calculate_loss_same_vectors(self):
        """Test MSE calculation with identical vectors"""
        output = np.array([[1.0], [2.0], [3.0]])
        expected = np.array([[1.0], [2.0], [3.0]])

        result = calculate_loss(output, expected, "mse")

        self.assertEqual(result, 0.0)

    def test_calculate_loss_different_vectors(self):
        """Test MSE calculation with different vectors"""
        output = np.array([[1.0], [2.0], [3.0]])
        expected = np.array([[2.0], [3.0], [4.0]])

        result = calculate_loss(output, expected, "mse")

        # Expected MSE: ((1-2)² + (2-3)² + (3-4)²) = (1 + 1 + 1) = 3.0
        # Note: calculate_loss sums the squared differences, doesn't divide by length
        self.assertEqual(result, 3.0)

    def test_calculate_loss_with_vector_objects(self):
        """Test MSE calculation with numpy array vectors"""
        output = np.array([[1.0], [2.0]])
        expected = np.array([[3.0], [4.0]])

        result = calculate_loss(output, expected, "mse")

        # Expected MSE: ((1-3)² + (2-4)²) = (4 + 4) = 8.0
        self.assertEqual(result, 8.0)


if __name__ == "__main__":
    unittest.main()
