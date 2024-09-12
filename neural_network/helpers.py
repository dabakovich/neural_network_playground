import numpy as np

from neural_network.matrix import Matrix
from neural_network.vector import Vector


def get_random(min_value=0, max_value=1):
    return np.random.random() * (max_value - min_value) + min_value


def outer_product(vector1: Vector, vector2: Vector):
    return Matrix(
        [[vector1.values[i] * vector2.values[j] for j in range(len(vector2.values))] for i in
         range(len(vector1.values))]
    )
