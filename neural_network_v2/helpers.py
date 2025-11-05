import numpy as np

from .types import Activator, LayerConfig, Loss, Matrix, Vector

rg = np.random.default_rng(1)


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
    min_weight = 0
    max_weight = 0.5

    weights = (max_weight - min_weight) * rg.random(
        (layer_config["output_size"], layer_config["input_size"])
    ) + min_weight

    biases = np.zeros((layer_config["output_size"],))

    # Add biases as the last column in the weights matrix
    weights_and_biases = np.hstack((weights, biases[:, np.newaxis]))

    return weights_and_biases


def calculate_loss(
    pred_items: list[Vector],
    true_items: list[Vector],
    loss_name: Loss = "mse",
) -> float:
    if len(true_items) != len(pred_items):
        raise ValueError("Length of arrays are not equal")

    sum = 0

    for index, true_y in enumerate[Vector](true_items):
        pred_y = pred_items[index]

        if loss_name == "mse":
            subtraction = true_y - pred_y

            sum += subtraction.dot(subtraction)
        elif loss_name == "log":
            sum += -true_y.dot(np.log(pred_y)) - (-true_y + 1).dot(np.log(1 - pred_y))
        else:
            raise ValueError("Unknown loss function name")

    return sum


def calculate_loss_derivative(
    pred_y: Vector,
    true_y: Vector,
    loss_name: Loss = "mse",
) -> Vector:
    if loss_name == "mse":
        return (pred_y - true_y) * 2
    elif loss_name == "log":
        return -(true_y / pred_y) + (1 - true_y) / (1 - pred_y)

    raise ValueError("Unknown loss function name")


def activate(input: Vector, activator: Activator) -> Vector:
    if activator == "linear":
        return input
    elif activator == "relu":
        return np.where(input > 0, input, 0)
    elif activator == "sigmoid":
        return 1 / (1 + np.exp(-input))
    if activator == "tanh":
        return np.tanh(input)

    raise ValueError("Unknown activator function")


def derivate(input: Vector, activator: Activator) -> Vector:
    if activator == "linear":
        return input
    if activator == "relu":
        return np.where(input > 0, 1, 0)
    if activator == "sigmoid":
        # dy/dx (1 / (1 + e^(-x))) = x * (1 - x)
        return input * (1 - input)
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

    return [nn_weight_slopes / len(batch_weight_slopes) for nn_weight_slopes in sum]
