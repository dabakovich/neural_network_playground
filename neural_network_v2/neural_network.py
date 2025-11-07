import math
import random

import numpy as np

from shared.helpers import get_random

from .helpers import (
    build_layers,
    calculate_loss,
    calculate_loss_derivative,
    calculate_mean_weight_slopes,
)
from .layer import Layer
from .types import DataItem, LayerConfig, Loss, Matrix, Vector
from .visual import (
    cleanup_plot,
    init_plot,
    render_losses,
    render_nn_output_for_two_inputs,
    # render_nn_output,
)


class NeuralNetwork:
    layers: list[Layer]
    loss_name: Loss

    def __init__(
        self,
        layer_configs: list[LayerConfig],
        weights_list_parameter: list[Matrix] | None = None,
        learning_rate=0.01,
        loss_name: Loss = "mse",
    ):
        self.layer_configs = layer_configs

        if weights_list_parameter is None:
            weights_list = build_layers(layer_configs)
        else:
            weights_list = weights_list_parameter

        layers = []
        for index, layer_config in enumerate[LayerConfig](layer_configs):
            layer = Layer(layer_config, weights_list[index], learning_rate)
            layers.append(layer)
        self.layers = layers

        self.learning_rate = learning_rate
        self.loss_name = loss_name

    def forward(self, input: Vector) -> list[Vector]:
        next_input = input.copy()
        calculated_layers: list[Vector] = [next_input]

        for layer in self.layers:
            next_input = layer.forward(next_input)

            calculated_layers.append(next_input)

        return calculated_layers

    def calculate_output(self, input: Vector) -> Vector:
        return self.forward(input)[-1]

    def calculate_loss(self, x_list: np.ndarray, y_list: np.ndarray) -> float:
        pred_items = np.array([self.calculate_output(x) for x in x_list])

        loss = calculate_loss(pred_items, y_list, self.loss_name)

        return loss

    def back_propagate(self, input: Vector, true_output: Vector):
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
        # print("initial weights", self.layers)

        calculated_layers = self.forward(input)

        # print("calculated_layers", calculated_layers)

        pred_output = calculated_layers[-1]

        d_loss_d_y = calculate_loss_derivative(pred_output, true_output, self.loss_name)

        # print("d_loss_d_y", d_loss_d_y)

        output_gradient = d_loss_d_y

        nn_weight_slopes: list[Matrix] = []

        for index in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[index]
            next_input = calculated_layers[index]

            # print("next_input", next_input)

            # Calculate new output gradient that will be used in the "next" layer
            weight_slopes, output_gradient = layer.backward(next_input, output_gradient)

            nn_weight_slopes.insert(0, weight_slopes)

            layer.update_weights(weight_slopes)

        # print("nn_weight_slopes", nn_weight_slopes)
        # print("New weights", self.layers)
        return

    def back_propagate_batch(self, x_batch: np.ndarray, y_batch: np.ndarray):
        # print("Starting back propagate for batch")
        predictions = [self.forward(x) for x in x_batch]

        all_batch_weight_slopes: list[list[Matrix]] = []

        for index, x in enumerate(x_batch):
            # print(f"item_index {item_index}")
            true_y = y_batch[index]
            calculated_layers = predictions[index]
            pred_y = calculated_layers[-1]

            # print(f"predicted_output {predicted_output}")
            # print(f"actual_output {actual_output}")

            d_loss_d_y = calculate_loss_derivative(pred_y, true_y, self.loss_name)
            output_gradient = d_loss_d_y

            nn_weight_slopes: list[Matrix] = []

            for layer_index in range(len(self.layers) - 1, -1, -1):
                # print(f"layer_index {layer_index}")
                layer = self.layers[layer_index]
                next_input = calculated_layers[layer_index]

                # print(f"output_gradient {output_gradient}")

                # Calculate new output gradient that will be used in the "next" layer
                layer_weight_slopes, output_gradient = layer.backward(
                    next_input, output_gradient
                )
                # Add layer weight slopes to the whole NN weight slopes
                nn_weight_slopes.insert(0, layer_weight_slopes)

            # Append neural network example weight slopes to the whole batch weight slopes list
            all_batch_weight_slopes.append(nn_weight_slopes)

        # print("all_batch_weight_slopes", all_batch_weight_slopes)
        # Calculate mean weight slopes for all examples in the batch
        mean_batch_weight_slopes = calculate_mean_weight_slopes(all_batch_weight_slopes)

        # print("mean_batch_weight_slopes", mean_batch_weight_slopes)

        # Update whole NN weights using batch mean slopes
        for index, weight_slopes in enumerate(mean_batch_weight_slopes):
            self.layers[index].update_weights(weight_slopes)

        # print("new weights", self.layers)

    def train(
        self,
        x_list: np.ndarray,
        y_list: np.ndarray,
        epochs: int,
        batch_size: int = 1,
        stop_on_loss: float | None = None,
        render_every=1000,
    ):
        losses = []

        print("initial loss", self.calculate_loss(x_list, y_list))

        # Initialize the plot for real-time updates
        init_plot()

        for iteration in range(epochs):
            # SGD method takes random item from the all dataset list and makes back propagate for one example
            if batch_size == 1:
                for _ in range(len(x_list)):
                    index = math.floor(get_random(0, len(x_list)))
                    # print("random data item index", date_item_index)

                    self.back_propagate(x_list[index], y_list[index])

            # Batch method calculates mean weight slopes and updates weights once per epoch
            else:
                randomize = np.arange(len(x_list))
                split_indices = np.arange(1, len(x_list), 1)

                # Shuffle the data for each epoch to introduce stochasticity and split into batches
                np.random.shuffle(randomize)

                x_batches = np.array_split(x_list[randomize], split_indices)
                y_batches = np.array_split(y_list[randomize], split_indices)

                for index, x in enumerate(x_batches):
                    self.back_propagate_batch(x, y_batches[index])

            new_loss = self.calculate_loss(x_list, y_list)
            losses.append(new_loss)

            is_stop = False

            # Stop training if loss is less than stop_on_loss
            if stop_on_loss is not None:
                if new_loss < stop_on_loss:
                    is_stop = True

            if iteration % render_every == 0 or iteration == 1 or is_stop:
                print("-" * 20)
                print("iteration", iteration)

                print("new loss", new_loss)
                print("new weights", self.layers)

                # Update plot
                render_nn_output_for_two_inputs(
                    x_list, y_list, lambda x: self.calculate_output(x)
                )

                render_losses(losses)

            # Stop training if loss is less than stop_on_loss
            if is_stop:
                break

        # Print final NN outputs
        print([(item, self.calculate_output(item)) for item in x_list])

        cleanup_plot()
