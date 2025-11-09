import numpy as np

from .helpers import activate, derivate
from .types import Activator, LayerConfig, Matrix, Vector


class Layer:
    layer_config: LayerConfig
    weights: Matrix
    calculated_layer: Vector | None = None

    def __init__(
        self,
        layer_config: LayerConfig,
        weights: Matrix,
        learning_rate: float,
    ):
        self.layer_config = layer_config
        self.weights = weights
        self.learning_rate = learning_rate

    def forward(self, input: Vector, weights_override: Matrix | None = None) -> Vector:
        if weights_override is not None:
            signal = weights_override @ np.append(input, 1)
        else:
            signal = self.weights @ np.append(input, 1)

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

        gradient *= dy_dn

        # print("gradient", gradient)

        # We need to transpose gradient to make it "vertical" matrix
        # We could use it for:
        # - Multiplying by input matrix to get weight slopes
        # - Multiplying by transpose weights matrix to get next gradient
        transposed_gradient = gradient[np.newaxis, :].T

        # We need transpose weights when moving backward, biases will be the last vector in the matrix
        transposed_weights = self.weights.transpose()

        # Remove last biases vector in the transposed_weights matrix, since biases doesn't depend on layer input
        transposed_weights = transposed_weights[:-1, :]

        # Calculate gradient for "next" layer
        next_gradient_matrix = transposed_weights @ transposed_gradient
        next_gradient_vector = next_gradient_matrix.T[0]

        # CALCULATE WEIGHT SLOPES AND UPDATE WEIGHTS

        # Add `1` for bias calculations and make vector as "horizontal" matrix
        input_matrix_with_bias = np.append(input, 1)[np.newaxis, :]

        # !!! NOT FINISHED HERE
        weight_slopes = transposed_gradient @ input_matrix_with_bias

        # print("weight_slopes", weight_slopes)

        return weight_slopes, next_gradient_vector

    def update_weights(self, weight_slopes: Matrix):
        self.weights = self.weights - (weight_slopes * self.learning_rate)

    def get_activator_name(self) -> Activator:
        return self.layer_config.get("activation", "linear")

    def __str__(self):
        return self.weights.__str__()

    def __repr__(self):
        return self.weights.__repr__()
