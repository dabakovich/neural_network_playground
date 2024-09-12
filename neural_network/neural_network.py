from neural_network.layer import Layer
from neural_network.matrix import Matrix
from neural_network.vector import Vector

my_learning_rate = 0.1


class NeuralNetwork:
    def __init__(self, layer_configs):
        self.layers = []
        self.learning_history = []

        for i in range(len(layer_configs)):
            layer_config = layer_configs[i]
            input_size = layer_config['input_size']
            output_size = layer_config['output_size']
            use_bias = layer_config.get('use_bias', True)
            activation = layer_config['activation']

            self.layers.append(Layer(input_size, output_size, use_bias, activation))

    # Load weights from a list of matrices, accepts a list of Matrix objects
    def load_weights(self, layer_matrices: list[list[list[float]]]):
        for i in range(len(layer_matrices)):
            self.layers[i].weights_matrix = Matrix(layer_matrices[i])

    def forward(self, input_vector: list[float] or Vector):
        if isinstance(input_vector, list):
            input_vector = Vector(input_vector)

        output = input_vector

        for layer in self.layers:
            output = layer.get_output(output)

        return output

    def train(self, data: list[tuple[list[float], list[float]]], epochs: int):
        for epoch in range(epochs):
            total_loss = 0

            for data_item in data:
                output = self.forward(data_item[0])

                print('Input:', data_item[0])
                print('Output:', output)
                print('Target:', data_item[1])

                loss = self.get_mse_loss(output, data_item[1])
                total_loss += loss

                self.backward(output, data_item[1])
                self.update_weights(my_learning_rate)

                print('-' * 25)

            mean_loss = total_loss / len(data)

            print(f"Epoch {epoch + 1}, Loss: {mean_loss}")
            print('-' * 50)

            self.learning_history.append(mean_loss)

        return self.learning_history

    def backward(self, output: Vector, target: list[float]):
        error = output - Vector(target)
        print('Error:', error)

        for layer in reversed(self.layers):
            error = layer.calculate_gradients(output, error)

    def update_weights(self, learning_rate):
        for layer in self.layers:
            layer.update_weights(learning_rate)

    def get_total_loss(self, data: list[tuple[list[float], list[float]]]):
        total_loss = 0

        for data_item in data:
            output = self.forward(data_item[0])
            loss = self.get_mse_loss(output, data_item[1])
            total_loss += loss

        return total_loss

    @staticmethod
    def get_mse_loss(output: Vector, target: list[float]):
        loss = 0

        loss += sum([(output.values[i] - target[i]) ** 2 for i in range(len(output.values))])

        return loss

    def __str__(self):
        return '\n---\n'.join([str(layer) for layer in self.layers])
