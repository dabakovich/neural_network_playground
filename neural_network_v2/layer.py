import numpy as np

from .helpers import activate, derivate
from .types import Activator, LayerConfig, Matrix, Vector


class Layer:
    layer_config: LayerConfig
    weights: Matrix
    biases: Vector
    calculated_layer: Vector | None = None

    def __init__(
        self,
        layer_config: LayerConfig,
        weights_and_biases: tuple[Matrix, Vector],
        learning_rate: float,
    ):
        self.layer_config = layer_config
        self.weights, self.biases = weights_and_biases
        self.learning_rate = learning_rate

    def forward(
        self,
        input: Vector,
        weights_and_biases_override: tuple[Matrix, Vector] | None = None,
    ) -> Vector:
        if weights_and_biases_override is not None:
            weights, biases = weights_and_biases_override
            signal = weights @ input + biases
        else:
            signal = self.weights @ input + self.biases

        activated_signal = activate(signal, self.get_activator_name())

        self.calculated_layer = activated_signal

        return activated_signal

    def backward(self, input: Vector, gradient: Vector):
        """
        Process backward gradient signal.

        This method calculate "next" gradient by multiplying gradient and weights matrices.

        Also, it calculates weights slopes and updates the layer weights.
        """
        if self.calculated_layer is None:
            raise ValueError("Layer was not calculated, call `forward` first")

        dy_dn = derivate(self.calculated_layer, self.get_activator_name())

        # Multiplying gradient by activation function derivate
        gradient *= dy_dn

        # Calculate gradient for "next" layer
        next_gradient = gradient @ self.weights

        # CALCULATE WEIGHT, BIASES SLOPES

        weight_slopes = gradient[:, np.newaxis] @ input[np.newaxis, :]
        biases_slopes = gradient.copy()

        return weight_slopes, biases_slopes, next_gradient

    def update_weights(self, weight_slopes: Matrix, biases_slopes: Vector):
        self.weights = self.weights - (weight_slopes * self.learning_rate)
        self.biases = self.biases - (biases_slopes * self.learning_rate)

    def get_activator_name(self) -> Activator:
        return self.layer_config.get("activation", "linear")

    def __str__(self):
        return f"Weights:\n{self.weights.__str__()}\nBiases:\n{self.biases.__str__()}\n"

    def __repr__(self):
        return (
            f"Weights:\n{self.weights.__repr__()}\nBiases:\n{self.biases.__repr__()}\n"
        )
