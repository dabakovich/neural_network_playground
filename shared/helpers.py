import numpy as np

from .matrix import Matrix
from .types import InputVector
from .vector import Vector


def get_random(min_value=0, max_value=1):
    return np.random.random() * (max_value - min_value) + min_value


def outer_product(vector1: Vector, vector2: Vector):
    return Matrix(
        [
            [vector1.values[i] * vector2.values[j] for j in range(len(vector2.values))]
            for i in range(len(vector1.values))
        ]
    )


def get_vector(input: InputVector) -> Vector:
    if isinstance(input, Vector):
        return input

    if isinstance(input, list):
        return Vector(input)

    raise ValueError("Unknown vector value provided")


def get_matrix(input: Matrix | list[InputVector]) -> Matrix:
    if isinstance(input, Matrix):
        return input

    if isinstance(input, list):
        return Matrix([Vector(vector) for vector in input])

    raise ValueError("Unknown matrix value provided")
