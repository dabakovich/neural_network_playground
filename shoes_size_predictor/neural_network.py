from shared.matrix import Matrix
from shared.vector import Vector
from .helpers import build_layers, calculate_mse, get_vector
from .types import DataItem, InputVector, Layer, LayerConfig


class NeuralNetwork:
    layers: list[Layer]

    def __init__(
        self,
        layer_configs: list[LayerConfig],
        layers: list[Layer] = None,
        learning_rate=0.01,
    ):
        self.layer_configs = layer_configs

        if layers is None:
            layers = build_layers(layer_configs)
        self.layers = layers

        self.learning_rate = learning_rate

    def forward(self, input: InputVector) -> list[Vector]:
        input = get_vector(input)

        next_input = input.clone()
        calculated_layers: list[Vector] = []

        for layer in self.layers:
            next_input = layer * Vector(next_input.values + [1])

            calculated_layers.append(next_input)

        return calculated_layers

    def calculate_output(self, input: InputVector) -> Vector:
        input = get_vector(input)

        return self.forward(input)[-1]

    def calculate_loss(self, data: list[DataItem]) -> float:
        sum = 0
        for item in data:
            output = self.calculate_output(item["input"])
            sum += calculate_mse(output, item["output"])

        return sum / len(data)

    def back_propagate(self, input: InputVector, expected_output: InputVector):
        """
        Back propagate neural network and update weights

        - Calculate initial (output) gradient.
        -- We're calculating derivative from loss function of NN output
        -- For MSE loss it's `2 * (output - input)`
        - After that we're iterating over each layer from last to the first one
        -- We're transposing output gradients matrix for multiplying with input matrix. It gives us weight slopes matrix with shape equal to the layer weights (countOfOutputNeurons, countOfInputNeurons).
        -- Calculate gradients for "next" layer output by multiplying weights matrix by "prev" gradients (like in forward signal propagation, but reverse)
        -- Update weights using the weight slopes matrix
        """
        input = get_vector(input)
        expected_output = get_vector(expected_output)

        calculated_layers = self.forward(input)
        output = calculated_layers[-1]

        initial_gradient = (output - expected_output) * 2

        print(initial_gradient)

        output_gradient = initial_gradient

        for layer_index in range(len(self.layers) - 1, -1, -1):
            # print(layer)
            input_with_bias: Matrix

            if layer_index == 0:
                input_with_bias = Matrix([input.values + [1]])
            else:
                input_with_bias = Matrix([calculated_layers[layer_index].values + [1]])

            transposed_output_gradient = Matrix([output_gradient]).transpose()

            print("transposed_output_gradient", transposed_output_gradient)
            print("input_with_bias", input_with_bias)

            # Calculate weight slopes to updates weights later
            weight_slopes = transposed_output_gradient * input_with_bias
            print("weight_slopes", weight_slopes)

            transposed_weights = self.layers[layer_index].transpose()

            # Remove last vector in the transposed_weights matrix
            transposed_weights = Matrix(transposed_weights.vectors[:-1])

            # Calculate output gradient for "next" layer
            output_gradient = transposed_weights * output_gradient

            # Update weights
            self.update_weights(layer_index, weight_slopes)

        print("New weights", self.layers)
        return

    def update_weights(self, layer_index: int, weight_slopes: Matrix):
        self.layers[layer_index] = self.layers[layer_index] - (
            weight_slopes * self.learning_rate
        )
