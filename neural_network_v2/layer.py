from shared.matrix import Matrix
from shared.vector import Vector

from .helpers import activate
from .types import Activator, LayerConfig


class Layer:
    layer_config: LayerConfig
    weights: Matrix

    def __init__(
        self, layer_config: LayerConfig, weights: Matrix, learning_rate: float
    ):
        self.layer_config = layer_config
        self.weights = weights
        self.learning_rate = learning_rate

    def forward(self, input: Vector) -> Vector:
        signal: Vector = self.weights * Vector(input.values + [1])

        activated_signal = signal.process(
            lambda value: activate(value, self.get_activator_name())
        )

        return activated_signal

    def backward(self, input: Vector, gradient: Vector) -> Vector:
        """
        Process backward gradient signal.

        This method calculate "next" gradient by multiplying gradient and weights matrices.

        Also, it calculates weights slopes and updates the layer weights.
        """
        # We need to transpose gradient to make it "vertical" matrix
        # We could use it for:
        # - Multiplying by input matrix to get weight slopes
        # - multiplying by transpose weights matrix to get next gradient
        transposed_gradient = Matrix([gradient]).transpose()

        # We need transpose weights when moving backward, biases will be the last vector in the matrix
        transposed_weights = self.weights.transpose()

        # Remove last biases vector in the transposed_weights matrix, since biases doesn't depend on layer input
        transposed_weights = Matrix(transposed_weights.vectors[:-1])

        # Calculate gradient for "next" layer
        next_gradient = transposed_weights * transposed_gradient

        # CALCULATE WEIGHT SLOPES AND UPDATE WEIGHTS
        input_matrix_with_bias = Matrix([input.values + [1]])

        weight_slopes = transposed_gradient * input_matrix_with_bias

        print("weight_slopes", weight_slopes)

        self.update_weights(weight_slopes)

        return next_gradient.transpose().vectors[0]

    def update_weights(self, weight_slopes: Matrix):
        self.weights = self.weights - (weight_slopes * self.learning_rate)

    def get_activator_name(self) -> Activator:
        return self.layer_config["activation"] or "linear"

    def __str__(self):
        return self.weights.__str__()

    def __repr__(self):
        return self.weights.__repr__()
