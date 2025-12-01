import numpy as np

from .helpers import activate, derivate, softmax_derivative
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

    def backward_softmax(self, input: Vector, gradient: Vector):
        if self.calculated_layer is None:
            raise ValueError("Layer was not calculated, call `forward` first")

        # Jacobian
        d_S_d_n = softmax_derivative(self.calculated_layer)

        d_loss_d_n = gradient[:, np.newaxis] * d_S_d_n

        # Duplicate input to make one input row for each output neuron
        # Number of columns – is count of input neurons
        # Number of rows – is count of output neurons
        input_matrix = input[np.newaxis, :]
        new_rows = np.tile(input_matrix[0], (len(gradient) - 1, 1))
        input_matrix = np.vstack((input_matrix, new_rows))

        # d_loss_d_w
        weight_slopes = d_loss_d_n.T @ input_matrix
        # d_loss_d_b
        biases_slopes = (d_loss_d_n.T @ np.ones((len(gradient), 1))).T[0]

        next_gradient_matrix = d_loss_d_n @ self.weights

        next_gradient = np.sum(next_gradient_matrix, axis=0)

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
