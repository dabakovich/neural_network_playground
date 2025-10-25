import matplotlib.pyplot as plt

from shared.helpers import get_random
from shared.matrix import Matrix
from shared.vector import Vector

from .helpers import (
    build_layers,
    calculate_mean_weight_slopes,
    calculate_mse,
    get_vector,
)
from .layer import Layer
from .types import DataItem, InputVector, LayerConfig
from .visual import cleanup_plot, init_plot, render_plot


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
        calculated_layers: list[Vector] = [next_input]

        for layer in self.layers:
            next_input = layer.forward(next_input)

            calculated_layers.append(next_input)

        return calculated_layers

    def calculate_output(self, input: InputVector) -> Vector:
        input = get_vector(input)

        return self.forward(input)[-1]

    def calculate_loss(self, batch: list[DataItem]) -> float:
        actual_items = [item["output"] for item in batch]
        predicted_items = []

        for item in batch:
            predicted_items.append(self.calculate_output(item["input"]))

        loss = calculate_mse(actual_items, predicted_items)

        return loss

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

        print("calculated_layers", calculated_layers)

        output = calculated_layers[-1]

        initial_gradient = (output - expected_output) * 2

        print("initial_gradient", initial_gradient)

        output_gradient = initial_gradient

        for index in range(len(self.layers) - 1, -1, -1):
            print(f"back propagate layer index {index}")
            print("output_gradient", output_gradient)

            layer = self.layers[index]
            next_input = calculated_layers[index]

            print("next_input", next_input)

            # Calculate new output gradient that will be used in the "next" layer
            weight_slopes, output_gradient = layer.backward(next_input, output_gradient)

            layer.update_weights(weight_slopes)

        print("New weights", self.layers)
        return

    def back_propagate_batch(self, batch: list[DataItem]):
        print("Starting back propagate for batch")
        predictions = [self.forward(item["input"]) for item in batch]

        all_batch_weight_slopes: list[list[Matrix]] = []

        for item_index, item in enumerate(batch):
            actual_output = get_vector(item["output"])
            calculated_layers = predictions[item_index]
            predicted_output = calculated_layers[-1]

            initial_gradient = (predicted_output - actual_output) * 2
            output_gradient = initial_gradient

            nn_weight_slopes: list[Matrix] = []

            for layer_index in range(len(self.layers) - 1, -1, -1):
                layer = self.layers[layer_index]
                next_input = calculated_layers[layer_index]

                # Calculate new output gradient that will be used in the "next" layer
                layer_weight_slopes, output_gradient = layer.backward(
                    next_input, output_gradient
                )
                # Add layer weight slopes to the whole NN weight slopes
                nn_weight_slopes.insert(0, layer_weight_slopes)

            # Append neural network example weight slopes to the whole batch weight slopes list
            all_batch_weight_slopes.append(nn_weight_slopes)

        # Calculate mean weight slopes for all examples in the batch
        mean_batch_weight_slopes = calculate_mean_weight_slopes(all_batch_weight_slopes)

        print("mean_batch_weight_slopes", mean_batch_weight_slopes)

        # Update whole NN weights using batch mean slopes
        for index, weight_slopes in enumerate(mean_batch_weight_slopes):
            self.layers[index].update_weights(weight_slopes)

    def train_sgd(self, data: list[DataItem], epochs: int):
        """
        Train SGD (stochastic gradient descent)
        It takes random item from the all dataset list and makes back propagate for one example
        """
        data_tuples = [(item["input"], item["output"]) for item in data]
        losses = []

        print("initial loss", self.calculate_loss(data))

        # Initialize the plot for real-time updates
        init_plot()

        for iteration in range(epochs):
            print("-" * 20)
            print("iteration", iteration)

            date_item_index = round(get_random(0, len(data) - 1))
            print("random data item index", date_item_index)

            data_item = data[date_item_index]
            print("data_item", data_item)

            self.back_propagate(data_item["input"], data_item["output"])

            new_loss = self.calculate_loss(data)
            losses.append(new_loss)
            print("new loss", new_loss)

            # Update plot

            # Works only with one size input for now
            render_plot(data_tuples, lambda x: self.forward(x), losses)

            plt.pause(0.5)  # Use matplotlib's pause for better integration

        cleanup_plot()

    def train_batch(self, data: list[DataItem], epochs: int):
        """
        Train NN over batch examples.
        It calculates mean weight slopes and updates weights once per epoch
        """
        data_tuples = [(item["input"], item["output"]) for item in data]
        losses = []

        print("initial loss", self.calculate_loss(data))

        # Initialize the plot for real-time updates
        init_plot()

        for iteration in range(epochs):
            print("-" * 20)
            print("iteration", iteration)

            self.back_propagate_batch(data)

            new_loss = self.calculate_loss(data)
            losses.append(new_loss)
            print("new loss", new_loss)

            # Update plot

            # Works only with one size input for now
            render_plot(data_tuples, lambda x: self.forward(x), losses)

            plt.pause(0.5)  # Use matplotlib's pause for better integration

        cleanup_plot()
