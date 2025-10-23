from shoes_size_predictor.datasets import shoes_dataset
from shoes_size_predictor.neural_network import NeuralNetwork


nn = NeuralNetwork(
    [{"input_size": 1, "output_size": 1, "activation": "linear"}]
    # [{"input_size": 1, "output_size": 2}, {"input_size": 2, "output_size": 1}]
)


nn.train(shoes_dataset, 50)
