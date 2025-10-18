from shared.helpers import get_random
from shared.matrix import Matrix
from shared.vector import Vector
from .types import InputVector, Layer, LayerConfig


def build_layers(layer_configs: list[LayerConfig]) -> list[Layer]:
    """
    Builds neural network layers with biases
    """
    layers: list[Layer] = []

    # Iterate over all layers
    for l_index in range(len(layer_configs)):
        layer_config = layer_configs[l_index]
        layer = Matrix()

        # Iterate over output neurons
        for n_index in range(layer_config["output_size"]):
            weights_and_bias = Vector(
                [get_random(-1, 1) for _ in range(layer_config["input_size"])]
            )
            # Ad bias value
            weights_and_bias.values.append(get_random(-1, 1))

            layer.vectors.append(weights_and_bias)

        layers.append(layer)

    return layers


def get_vector(input: InputVector) -> Vector:
    if isinstance(input, list):
        return Vector(input)

    return input
