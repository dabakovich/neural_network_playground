import math

from shared.helpers import get_random
from shared.matrix import Matrix
from shared.vector import Vector

from .types import Activator, InputVector, LayerConfig


def build_layers(layer_configs: list[LayerConfig]) -> list[Matrix]:
    """
    Builds neural network layers with biases
    """
    weights_list: list[Matrix] = []

    # Iterate over all layers
    for layer_config in layer_configs:
        weights = build_layer(layer_config)

        weights_list.append(weights)

    return weights_list


def build_layer(layer_config: LayerConfig) -> Matrix:
    weights = Matrix()

    # Iterate over output neurons
    for n_index in range(layer_config["output_size"]):
        weights_and_bias = Vector(
            [get_random(-1, 1) for _ in range(layer_config["input_size"])]
        )
        # Ad bias value
        weights_and_bias.values.append(get_random(-1, 1))

        weights.vectors.append(weights_and_bias)

    return weights


def get_vector(input: InputVector) -> Vector:
    if isinstance(input, list):
        return Vector(input)

    return input


def calculate_mse(
    actual_items: list[InputVector], predicted_items: list[InputVector]
) -> float:
    if len(actual_items) != len(predicted_items):
        raise ValueError("Length of arrays are not equal")

    sum = 0

    for index, actual_item in enumerate(actual_items):
        actual_item = get_vector(actual_item)
        predicted_item = get_vector(predicted_items[index])

        subtraction = actual_item - predicted_item

        sum += subtraction * subtraction

    return sum / len(actual_items)


def activate(input: Vector, activator: Activator) -> Vector:
    if activator == "linear":
        return input
    elif activator == "relu":
        return input.process(lambda value: value if value > 0 else -value)
    elif activator == "sigmoid":
        return input.process(lambda value: 1 / (1 + math.exp(-value)))

    raise ValueError("Unknown activator function")


def derivate(input: Vector, activator: Activator) -> Vector:
    if activator == "linear":
        return 1
    if activator == "relu":
        return input.process(lambda value: 1 if value > 0 else 0)
    if activator == "sigmoid":
        # dy/dx (1 / (1 + e^(-x))) = x * (1 - x)
        return input.process(lambda value: value * (1 - value))

    raise ValueError("Unknown activator function")


def calculate_mean_weight_slopes(
    batch_weight_slopes: list[list[Matrix]],
) -> list[Matrix]:
    sum: list[Matrix] = batch_weight_slopes[0]

    for nn_weight_slopes in batch_weight_slopes[1:]:
        sum = [
            sum[index] + layer_weight_slopes
            for (index, layer_weight_slopes) in enumerate(nn_weight_slopes)
        ]

    return [
        nn_weight_slopes.process(lambda value: value / len(batch_weight_slopes))
        for nn_weight_slopes in sum
    ]
