import math

from shared.helpers import get_random, outer_product
from shared.matrix import Matrix
from shared.vector import Vector


class Layer:
    def __init__(self, input_size, output_size, use_bias, activation_name):
        self.d_output = None
        self.d_weights = None
        self.input_vector = None
        self.input_size = input_size
        self.output_size = output_size
        self.use_bias = use_bias
        self.activation_name = activation_name
        weights = []

        for i in range(output_size):
            neuron_weights = Vector([get_random(-1, 1) for _ in range(input_size)])
            neuron_weights.values.append(get_random(-1, 1) if use_bias else 0)
            weights.append(neuron_weights)

        self.weights_matrix = Matrix(weights)

    def set_weights(self, weights: list[list[float]]):
        if len(weights) != self.output_size:
            raise ValueError(
                "The number of weight vectors must match the output size of the layer"
            )

        self.weights_matrix = Matrix(weights)

    def get_output(self, input_vector: Vector):
        # Add 1 for bias at the end of the input vector
        input_vector = Vector(input_vector.values + [1])

        self.input_vector = input_vector

        output = self.weights_matrix * input_vector

        if self.activation_name == "softmax":
            return self.calculate_softmax(output)

        return Vector([self.activation(x) for x in output.values])

    def calculate_gradients(self, output_vector: Vector, error: Vector):
        print("input_vector:", self.input_vector)
        print("self.weights_matrix:\n", self.weights_matrix)

        d_output_d_activation = Vector(
            [self.activation_derivative(x) for x in output_vector.values]
        )
        print("d_output_d_activation:", d_output_d_activation)

        self.d_output = d_output_d_activation.multiply(error)
        print("delta:", self.d_output)

        self.d_weights = outer_product(self.d_output, self.input_vector)
        print("d_weights:\n", self.d_weights)

        transposed_weights = self.weights_matrix.transpose()
        print("transposed_weights:\n", transposed_weights)

        prev_layer_error = transposed_weights * self.d_output

        return prev_layer_error

    def update_weights(self, learning_rate):
        print("weights_matrix before:\n", self.weights_matrix)

        self.weights_matrix = self.weights_matrix - self.d_weights.scalar_multiply(
            learning_rate
        )

        print("weights_matrix after:\n", self.weights_matrix)

    def activation(self, x):
        if self.activation_name == "linear":
            return x
        elif self.activation_name == "perceptron":
            return 1 if x > 0 else 0
        elif self.activation_name == "sigmoid":
            return 1 / (1 + math.exp(-x))
        elif self.activation_name == "relu":
            return max(0, x)
        elif self.activation_name == "tanh":
            return math.tanh(x)

    def activation_derivative(self, x):
        if self.activation_name == "linear":
            return 1
        elif self.activation_name == "perceptron":
            return 1
        elif self.activation_name == "sigmoid":
            return x * (1 - x)
        elif self.activation_name == "relu":
            return 1 if x > 0 else 0
        elif self.activation_name == "tanh":
            return 1 - x

    @staticmethod
    def calculate_softmax(v: Vector):
        total_sum = sum([math.exp(x) for x in v.values])
        return [math.exp(x) / total_sum for x in v.values]

    def __str__(self):
        return f"Layer: {self.activation_name} - {self.input_size} -> {self.output_size}, weights:\n{self.weights_matrix}"
