from pandas import DataFrame

from .random import rng

from .matrix import Matrix
from .types import InputVector
from .vector import Vector


def get_random(min_value=0, max_value=1):
    return rng.random() * (max_value - min_value) + min_value


def split_dataset(
    dataset: DataFrame,
    train_percentage: float,
    validation_percentage: float,
    test_percentage: float,
):
    number_samples = len(dataset)

    if train_percentage + validation_percentage + test_percentage > 1:
        raise ValueError("Percentages should not be higher than 1")

    train_index = round(number_samples * train_percentage)
    validation_index = train_index + round(number_samples * validation_percentage)

    train_data: DataFrame = dataset.iloc[0:train_index]
    validation_data: DataFrame = dataset.iloc[train_index:validation_index]
    test_data: DataFrame = dataset.iloc[validation_index:]

    return (train_data, validation_data, test_data)


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
