import os
import sys
import unittest
from unittest.mock import patch

# Add the parent directory to the path so we can import from neural_network_v2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_network_v2.helpers import build_layers, calculate_loss
from neural_network_v2.types import LayerConfig
from shared.matrix import Matrix
from shared.vector import Vector


class TestHelpers(unittest.TestCase):
    @patch("neural_network_v2.helpers.get_random")
    def test_build_layers(self, mock_get_random):
        """Test build_layers function"""
        # Mock random values to make test deterministic
        # Layer 1: 2 neurons × (2 weights + 1 bias) = 6 calls
        # Layer 2: 1 neuron × (2 weights + 1 bias) = 3 calls
        # Total: 9 calls
        mock_get_random.side_effect = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        layer_configs = [
            LayerConfig(input_size=2, output_size=2),
            LayerConfig(input_size=2, output_size=1),
        ]

        layers = build_layers(layer_configs)

        # Check we got the right number of layers
        self.assertEqual(len(layers), 2)

        # Check first layer structure
        first_layer = layers[0]
        self.assertIsInstance(first_layer, Matrix)
        self.assertEqual(len(first_layer.vectors), 2)  # output_size

        # Check each vector has input_size + 1 (for bias)
        for vector in first_layer.vectors:
            self.assertEqual(len(vector.values), 3)  # input_size + bias

        # Check second layer structure
        second_layer = layers[1]
        self.assertEqual(len(second_layer.vectors), 1)  # output_size
        self.assertEqual(len(second_layer.vectors[0].values), 3)  # input_size + bias

    def test_calculate_loss_same_vectors(self):
        """Test MSE calculation with identical vectors"""
        output = [1.0, 2.0, 3.0]
        expected = [1.0, 2.0, 3.0]

        result = calculate_loss(output, expected, "mse")

        self.assertEqual(result, 0.0)

    def test_calculate_loss_different_vectors(self):
        """Test MSE calculation with different vectors"""
        output = [1.0, 2.0, 3.0]
        expected = [2.0, 3.0, 4.0]

        result = calculate_loss(output, expected, "mse")

        # Expected MSE: ((1-2)² + (2-3)² + (3-4)²) / 3 = (1 + 1 + 1) / 3 = 1.0
        self.assertEqual(result, 1.0)

    def test_calculate_loss_with_vector_objects(self):
        """Test MSE calculation with Vector objects"""
        output = Vector([1.0, 2.0])
        expected = Vector([3.0, 4.0])

        result = calculate_loss(output, expected, "mse")

        # Expected MSE: ((1-3)² + (2-4)²) / 2 = (4 + 4) / 2 = 4.0
        self.assertEqual(result, 4.0)


if __name__ == "__main__":
    unittest.main()
