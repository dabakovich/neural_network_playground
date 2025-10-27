from functools import reduce
import math
from neural_network_v2.datasets import (
    not_dataset,
    shoes_dataset,
    and_dataset,
    xor_dataset,
)
from neural_network_v2.helpers import calculate_loss_derivative
from neural_network_v2.neural_network import NeuralNetwork
from shared.matrix import Matrix
from shared.vector import Vector


nn_one_layer = NeuralNetwork(
    [
        {"input_size": 1, "output_size": 1, "activation": "linear"},
    ],
    learning_rate=0.01,
)

nn_one_layer_sigmoid = NeuralNetwork(
    [
        {"input_size": 1, "output_size": 1, "activation": "sigmoid"},
    ],
    # [[[-3, 1.3]]],
    learning_rate=1,
    loss_name="log",
)

nn_one_layer_two_inputs_sigmoid = NeuralNetwork(
    [
        {"input_size": 2, "output_size": 1, "activation": "sigmoid"},
    ],
    # [[[-3, 1.3]]],
    learning_rate=1,
    loss_name="log",
)

nn_two_layers_two_inputs_sigmoid = NeuralNetwork(
    [
        {"input_size": 2, "output_size": 2, "activation": "linear"},
        {"input_size": 2, "output_size": 1, "activation": "sigmoid"},
    ],
    # [[[-3, 1.3]]],
    learning_rate=1,
    loss_name="log",
)

nn_two_layers = NeuralNetwork(
    [
        {"input_size": 1, "output_size": 2, "activation": "linear"},
        {"input_size": 2, "output_size": 1, "activation": "linear"},
    ],
    learning_rate=0.001,
)

nn_one_layer_two_inputs = NeuralNetwork(
    [
        {"input_size": 2, "output_size": 1, "activation": "linear"},
    ],
    learning_rate=0.05,
)


# print(nn_two_layers_two_inputs_sigmoid.layers)
# nn_one_layer_sigmoid.train_batch(not_dataset, 500)
nn_one_layer_two_inputs_sigmoid.train_batch(and_dataset, 500)
# nn_two_layers_two_inputs_sigmoid.train_batch(and_dataset, 500)
# nn_two_layers_two_inputs_sigmoid.train_sgd(xor_dataset, 500)
# nn_one_layer.train_batch(shoes_dataset, 50)
# nn_two_layers.train_batch(shoes_dataset, 50)
# nn_one_layer_two_inputs.train(and_dataset, 50)
