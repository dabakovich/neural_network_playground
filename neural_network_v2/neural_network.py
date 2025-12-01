import math
import time

import numpy as np

from shared.helpers import get_random

from .helpers import (
    build_layers,
    calculate_loss,
    calculate_loss_derivative,
    calculate_mean_biases_slopes,
    calculate_mean_weight_slopes,
)
from .layer import Layer
from .types import LayerConfig, Loss, Matrix, Vector
from .visual import (
    cleanup_plot,
    init_plot,
    render_losses,
    render_nn_output_for_two_inputs,
    render_weight_loss_plot_3d,
    # render_nn_output,
)


np.set_printoptions(precision=3)


class NeuralNetwork:
    layers: list[Layer]
    loss_name: Loss

    def __init__(
        self,
        layer_configs: list[LayerConfig],
        weights_list_parameter: tuple[list[Matrix], list[Vector]] | None = None,
        learning_rate=0.01,
        loss_name: Loss = "mse",
    ):
        self.layer_configs = layer_configs

        if weights_list_parameter is None:
            weights_list, biases_list = build_layers(layer_configs)
        else:
            weights_list, biases_list = weights_list_parameter

        layers = []
        for index, layer_config in enumerate[LayerConfig](layer_configs):
            layer = Layer(
                layer_config, (weights_list[index], biases_list[index]), learning_rate
            )
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

    def calculate_accuracy(
        self,
        x_list: np.ndarray,
        y_list: np.ndarray,
        threshold: float,
    ):
        pred_items = np.array(
            [1 if self.calculate_output(x) >= threshold else 0 for x in x_list]
        )

        return (y_list.flatten() == pred_items).sum() / len(y_list)

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

        # Calculate loss derivate with respect to activation output
        d_loss_d_y = calculate_loss_derivative(pred_output, true_output, self.loss_name)

        # print("d_loss_d_y", d_loss_d_y)

        # The loss derivate will be the initial gradient for the back forward
        output_gradient = d_loss_d_y

        nn_weight_slopes: list[Matrix] = []

        for index in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[index]
            next_input = calculated_layers[index]

            # print("next_input", next_input)

            # ToDo: need to be tested
            if layer.get_activator_name() == "softmax":
                # Calculate new output gradient that will be used in the "next" layer
                weight_slopes, biases_slopes, output_gradient = layer.backward_softmax(
                    next_input, output_gradient
                )
            else:
                # Calculate new output gradient that will be used in the "next" layer
                weight_slopes, biases_slopes, output_gradient = layer.backward(
                    next_input, output_gradient
                )

            nn_weight_slopes.insert(0, weight_slopes)

            layer.update_weights(weight_slopes, biases_slopes)

        # print("nn_weight_slopes", nn_weight_slopes)
        # print("New weights", self.layers)
        return

    def back_propagate_batch(self, x_batch: np.ndarray, y_batch: np.ndarray):
        predictions = [self.forward(x) for x in x_batch]

        all_batch_weight_slopes: list[list[Matrix]] = []
        all_batch_biases_slopes: list[list[Vector]] = []

        for index, x in enumerate(x_batch):
            true_y = y_batch[index]
            calculated_layers = predictions[index]
            pred_y = calculated_layers[-1]

            # print(f"predicted_output {predicted_output}")
            # print(f"actual_output {actual_output}")

            d_loss_d_y = calculate_loss_derivative(pred_y, true_y, self.loss_name)
            output_gradient = d_loss_d_y

            nn_weight_slopes: list[Matrix] = []
            nn_biases_slopes: list[Vector] = []

            for layer_index in range(len(self.layers) - 1, -1, -1):
                layer = self.layers[layer_index]
                next_input = calculated_layers[layer_index]

                # Calculate new output gradient that will be used in the "next" layer
                if layer.get_activator_name() == "softmax":
                    layer_weight_slopes, layer_biases_slopes, output_gradient = (
                        layer.backward_softmax(next_input, output_gradient)
                    )
                else:
                    layer_weight_slopes, layer_biases_slopes, output_gradient = (
                        layer.backward(next_input, output_gradient)
                    )

                # Add layer weight slopes to the whole NN weight slopes
                nn_weight_slopes.insert(0, layer_weight_slopes)
                nn_biases_slopes.insert(0, layer_biases_slopes)

            # Append neural network example weight slopes to the whole batch weight slopes list
            all_batch_weight_slopes.append(nn_weight_slopes)
            all_batch_biases_slopes.append(nn_biases_slopes)

        # print("all_batch_weight_slopes", all_batch_weight_slopes)
        # Calculate mean weight slopes for all examples in the batch
        mean_batch_weight_slopes = calculate_mean_weight_slopes(all_batch_weight_slopes)
        mean_batch_biases_slopes = calculate_mean_biases_slopes(all_batch_biases_slopes)

        # print("mean_batch_weight_slopes", mean_batch_weight_slopes)

        # Update whole NN weights using batch mean slopes
        for index in range(len(mean_batch_weight_slopes)):
            self.layers[index].update_weights(
                mean_batch_weight_slopes[index], mean_batch_biases_slopes[index]
            )

        # print("new weights", self.layers)

    def train(
        self,
        x_list: np.ndarray,
        y_list: np.ndarray,
        epochs: int,
        batch_size: int = 1,
        stop_on_loss: float | None = None,
        render_every: int | None = None,
        threshold: float | None = None,
    ):
        losses: list[float] = []

        print(f"Initial loss: {self.calculate_loss(x_list, y_list):.3f}")

        if threshold is not None:
            print(
                f"Initial accuracy: {self.calculate_accuracy(x_list, y_list, threshold):.3f}"
            )

        if render_every:
            # Initialize the plot for real-time updates
            init_plot()

        for epoch in range(epochs):
            time_per_step_ms = 0
            # SGD method takes random item from the all dataset list and makes back propagate for one example
            if batch_size == 1:
                for _ in range(len(x_list)):
                    index = math.floor(get_random(0, len(x_list)))
                    # print("random data item index", date_item_index)

                    self.back_propagate(x_list[index], y_list[index])

            # Batch method calculates mean weight slopes and updates weights once per epoch
            else:
                randomize = np.arange(len(x_list))
                split_indices = np.arange(1, len(x_list), batch_size)

                # Shuffle the data for each epoch to introduce stochasticity and split into batches
                np.random.shuffle(randomize)

                x_batches = np.array_split(x_list[randomize], split_indices)
                y_batches = np.array_split(y_list[randomize], split_indices)

                for index, x in enumerate(x_batches):
                    start_time = time.time()
                    self.back_propagate_batch(x, y_batches[index])
                    end_time = time.time()

                    time_per_step_ms = (end_time - start_time) * 1000

            new_loss = self.calculate_loss(x_list, y_list)
            losses.append(new_loss)

            is_stop = False

            # Stop training if loss is less than stop_on_loss
            if stop_on_loss is not None:
                if new_loss < stop_on_loss:
                    is_stop = True

            print(f"Epoch {epoch + 1}/{epochs}")

            metrics = [
                f"{time_per_step_ms:.3f}ms/step",
                f"loss: {new_loss:.3f}",
            ]

            if threshold is not None:
                metrics.append(
                    f"acc: {self.calculate_accuracy(x_list, y_list, threshold):.3f}"
                )

            print(" - ".join(metrics))

            # Update plot
            # render_nn_output_for_two_inputs(
            #     x_batches[0], y_batches[0], lambda x: self.calculate_output(x)
            # )

            if render_every and (epoch % render_every == 0 or epoch == 0 or is_stop):
                render_losses(losses)

            # Stop training if loss is less than stop_on_loss
            if is_stop:
                break

        # Print sample NN outputs
        randomize = np.arange(len(x_list))
        np.random.shuffle(randomize)
        indices = randomize[0:4]

        print(
            np.array(
                [(self.calculate_output(item)) for item in x_list[indices]]
            ).flatten()
        )
        print(y_list[indices].flatten())

        print(f"FINAL LAYERS\n{self.__str__()}")

        cleanup_plot()

        return losses

    def __str__(self):
        return "\n".join(
            [
                f"Layer{index + 1}:\n{layer.__str__()}"
                for index, layer in enumerate(self.layers)
            ]
        )
