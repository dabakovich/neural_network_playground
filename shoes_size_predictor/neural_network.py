from shared.vector import Vector
from .helpers import build_layers, get_vector
from .types import InputVector, Layer, LayerConfig


class NeuralNetwork:
    layers: list[Layer]

    def __init__(self, layer_configs: list[LayerConfig], layers: list[Layer] = None):
        self.layer_configs = layer_configs

        if layers is None:
            layers = build_layers(layer_configs)
        self.layers = layers

    def forward(self, input: InputVector) -> list[Vector]:
        input = get_vector(input)

        next_input = input.clone()
        calculated_layers: list[Vector] = [next_input]

        for layer in self.layers:
            next_input = layer * Vector(next_input.values + [1])

            calculated_layers.append(next_input)

        return calculated_layers

    def calculate_output(self, input: InputVector) -> Vector:
        input = get_vector(input)

        return self.forward(input)[-1]

    def calculate_loss(data) -> float:
        return 0
