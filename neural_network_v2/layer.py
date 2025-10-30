from shared.helpers import get_matrix
from shared.matrix import Matrix
from shared.types import InputMatrix
from shared.vector import Vector

from .helpers import activate, derivate
from .types import Activator, LayerConfig


class Layer:
    layer_config: LayerConfig
    weights: Matrix
    calculated_layer: Vector | None = None

    def __init__(
        self,
        layer_config: LayerConfig,
        weights: InputMatrix,
        learning_rate: float,
    ):
        self.layer_config = layer_config
        self.weights = get_matrix(weights)
        self.learning_rate = learning_rate

    def forward(self, input: Vector) -> Vector:
        signal: Vector = self.weights * Vector(input.values + [1])

        activated_signal = activate(signal, self.get_activator_name())

        self.calculated_layer = activated_signal

        return activated_signal

    def backward(self, input: Vector, gradient: Vector):
        """
        Process backward gradient signal.

        This method calculate "next" gradient by multiplying gradient and weights matrices.

        Also, it calculates weights slopes and updates the layer weights.
        """
        dy_dn = derivate(self.calculated_layer, self.get_activator_name())

        gradient = gradient.multiply(dy_dn)

        # print("gradient", gradient)

        # We need to transpose gradient to make it "vertical" matrix
        # We could use it for:
        # - Multiplying by input matrix to get weight slopes
        # - Multiplying by transpose weights matrix to get next gradient
        transposed_gradient = Matrix([gradient]).transpose()

        # We need transpose weights when moving backward, biases will be the last vector in the matrix
        transposed_weights = self.weights.transpose()

        # Remove last biases vector in the transposed_weights matrix, since biases doesn't depend on layer input
        transposed_weights = Matrix(transposed_weights.vectors[:-1])

        # Calculate gradient for "next" layer
        next_gradient_matrix = transposed_weights * transposed_gradient
        next_gradient_vector: Vector = next_gradient_matrix.transpose().vectors[0]

        # CALCULATE WEIGHT SLOPES AND UPDATE WEIGHTS
        input_matrix_with_bias = Matrix([input.values + [1]])

        weight_slopes: Matrix = transposed_gradient * input_matrix_with_bias

        # print("weight_slopes", weight_slopes)

        return weight_slopes, next_gradient_vector

    def update_weights(self, weight_slopes: Matrix):
        self.weights = self.weights - (weight_slopes * self.learning_rate)

    def get_activator_name(self) -> Activator:
        return self.layer_config["activation"] or "linear"

    def __str__(self):
        return self.weights.__str__()

    def __repr__(self):
        return self.weights.__repr__()
