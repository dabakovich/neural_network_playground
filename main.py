from neural_network_v2.datasets import shoes_dataset, and_dataset
from neural_network_v2.neural_network import NeuralNetwork


nn_one_layer = NeuralNetwork(
    [
        {"input_size": 1, "output_size": 1, "activation": "linear"},
    ],
    learning_rate=0.01,
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


nn_two_layers.train(shoes_dataset, 50)
# nn_one_layer_two_inputs.train(and_dataset, 50)
