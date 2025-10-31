import math

from shared.helpers import get_random, get_vector
from shared.matrix import Matrix
from shared.types import InputVector
from shared.vector import Vector

from .types import Activator, LayerConfig, Loss


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
        min_weight = 0
        max_weight = 0.5
        weights_and_bias = Vector(
            [
                get_random(min_weight, max_weight)
                for _ in range(layer_config["input_size"])
            ]
        )
        # Ad bias value
        weights_and_bias.values.append(0)
        # weights_and_bias.values.append(get_random(-0.5, 0.5))
        # weights_and_bias.values.append(get_random(0, 2))
        # weights_and_bias.values.append(get_random(-1, 1))

        weights.vectors.append(weights_and_bias)

    return weights


def calculate_loss(
    actual_items: list[InputVector],
    predicted_items: list[InputVector],
    loss_name: Loss = "mse",
) -> float:
    if len(actual_items) != len(predicted_items):
        raise ValueError("Length of arrays are not equal")

    sum = 0

    for index, actual_y in enumerate[InputVector](actual_items):
        actual_y = get_vector(actual_y)
        predicted_y = get_vector(predicted_items[index])

        if loss_name == "mse":
            subtraction = actual_y - predicted_y

            sum += subtraction * subtraction
        elif loss_name == "log":
            sum += -actual_y * predicted_y.process(lambda value: math.log(value)) - (
                -actual_y + 1
            ) * predicted_y.process(lambda value: math.log(1 - value))
        else:
            raise ValueError("Unknown loss function name")

    return sum


def calculate_loss_derivative(
    predicted_output: Vector,
    actual_output: Vector,
    loss_name: Loss = "mse",
) -> Vector:
    if loss_name == "mse":
        return (predicted_output - actual_output) * 2
    elif loss_name == "log":
        return -actual_output.divide(predicted_output) + (1 - actual_output).divide(
            1 - predicted_output
        )

    raise ValueError("Unknown loss function name")


def activate(input: Vector, activator: Activator) -> Vector:
    if activator == "linear":
        return input
    elif activator == "relu":
        return input.process(lambda value: value if value > 0 else 0)
    elif activator == "sigmoid":
        return input.process(lambda value: 1 / (1 + math.exp(-value)))
    if activator == "tanh":
        return input.process(lambda value: math.tanh(value))

    raise ValueError("Unknown activator function")


def derivate(input: Vector, activator: Activator) -> Vector:
    if activator == "linear":
        return input
    if activator == "relu":
        return input.process(lambda value: 1 if value > 0 else 0)
    if activator == "sigmoid":
        # dy/dx (1 / (1 + e^(-x))) = x * (1 - x)
        return input.process(lambda value: value * (1 - value))
    if activator == "tanh":
        # dy/dx tanh(x) = 1 - tanh^2(x)
        return 1 - input**2

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
