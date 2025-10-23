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


def calculate_mse(output: InputVector, expected_output: InputVector) -> float:
    sum = 0
    output = get_vector(output)
    expected_output = get_vector(expected_output)

    subtraction = output - expected_output

    sum += (subtraction * subtraction) / len(expected_output)

    return sum


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
