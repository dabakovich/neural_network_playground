from shoes_size_predictor.neural_network import NeuralNetwork

nn = NeuralNetwork(
    [{"input_size": 1, "output_size": 2}, {"input_size": 2, "output_size": 1}]
)

print(f"layers:\n{'\n---\n'.join([str(layer) for layer in nn.layers])}")

calculated_layers = nn.forward([1])

print(
    f"calculated layers:\n{'\n---\n'.join([str(layer) for layer in calculated_layers])}"
)

print(f"Output: {nn.calculate_output([1])}")
