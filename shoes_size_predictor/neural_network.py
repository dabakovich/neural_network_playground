import matplotlib.pyplot as plt

from shared.helpers import get_random
from shared.matrix import Matrix
from shared.vector import Vector
from shoes_size_predictor.visual import cleanup_plot, init_plot, render_plot

from .helpers import build_layers, calculate_mse, get_vector
from .layer import Layer
from .types import DataItem, InputVector, LayerConfig


class NeuralNetwork:
    layers: list[Layer]

    def __init__(
        self,
        layer_configs: list[LayerConfig],
        weights_list: list[Matrix] = None,
        learning_rate=0.01,
    ):
        self.layer_configs = layer_configs

        if weights_list is None:
            weights_list = build_layers(layer_configs)

        layers = []
        for index, layer_config in enumerate(layer_configs):
            layer = Layer(layer_config, weights_list[index], learning_rate)
            layers.append(layer)
        self.layers = layers

        self.learning_rate = learning_rate

    def forward(self, input: InputVector) -> list[Vector]:
        input = get_vector(input)

        next_input = input.clone()
        calculated_layers: list[Vector] = []

        for layer in self.layers:
            next_input = layer.forward(next_input)

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

        print("initial weights", self.layers)

        calculated_layers = self.forward(input)
        output = calculated_layers[-1]

        initial_gradient = (output - expected_output) * 2

        print("initial_gradient", initial_gradient)

        output_gradient = initial_gradient

        for index, layer in enumerate(reversed(self.layers)):
            # print(layer)
            next_input: Vector

            if index == 0:
                next_input = input
            else:
                next_input = calculated_layers[index]

            # Calculate new output gradient that will be used in the "next" layer
            output_gradient = layer.backward(next_input, output_gradient)

        print("New weights", self.layers)
        return

    def train(self, data: list[DataItem], epochs: int):
        data_tuples = [(item["input"], item["output"]) for item in data]
        losses = []

        print("initial loss", self.calculate_loss(data))

        # Initialize the plot for real-time updates
        init_plot()

        for iteration in range(epochs):
            print("-" * 20)
            print("iteration", iteration)

            index = round(get_random(0, len(data) - 1))
            print("index", index)

            data_item = data[index]
            print("data_item", data_item)

            self.back_propagate(data_item["input"], data_item["output"])

            new_loss = self.calculate_loss(data)
            losses.append(new_loss)
            print("new loss", new_loss)

            # Render plot
            render_plot(data_tuples, losses)

            plt.pause(1)  # Use matplotlib's pause for better integration

        cleanup_plot()
