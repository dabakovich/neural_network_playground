import os
import sys
import unittest
from unittest.mock import patch

# Add the parent directory to the path so we can import from neural_network_v2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.helpers import get_vector
from shared.matrix import Matrix
from shared.vector import Vector


class TestHelpers(unittest.TestCase):
    def test_get_vector_with_list(self):
        """Test get_vector with a list input"""
        input_list = [1.0, 2.0, 3.0]
        result = get_vector(input_list)

        self.assertIsInstance(result, Vector)
        self.assertEqual(result.values, input_list)

    def test_get_vector_with_vector(self):
        """Test get_vector with a Vector input"""
        input_vector = Vector([1.0, 2.0, 3.0])
        result = get_vector(input_vector)

        self.assertIsInstance(result, Vector)
        self.assertEqual(result.values, input_vector.values)
        self.assertIs(result, input_vector)  # Should return the same object


if __name__ == "__main__":
    unittest.main()
