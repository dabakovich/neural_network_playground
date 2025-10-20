from shoes_size_predictor.datasets import shoes_dataset
from shoes_size_predictor.helpers import calculate_mse
from shoes_size_predictor.neural_network import NeuralNetwork

nn = NeuralNetwork(
    [{"input_size": 1, "output_size": 1}]
    # [{"input_size": 1, "output_size": 2}, {"input_size": 2, "output_size": 1}]
)


print(
    f"Output: {nn.back_propagate(shoes_dataset[0]['input'], shoes_dataset[0]['output'])}"
)
