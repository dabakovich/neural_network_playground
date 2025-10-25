from neural_network_v2.datasets import shoes_dataset
from neural_network_v2.neural_network import NeuralNetwork


nn = NeuralNetwork(
    [{"input_size": 1, "output_size": 1, "activation": "linear"}]
    # [{"input_size": 1, "output_size": 2}, {"input_size": 2, "output_size": 1}]
)


nn.train(shoes_dataset, 50)
