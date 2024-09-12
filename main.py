import matplotlib.pyplot as plt
import numpy as np

from neural_network import get_random
from neural_network import NeuralNetwork

data = []

nn = NeuralNetwork([
    {'input_size': 1, 'output_size': 1, 'activation': 'linear'},
    # {'input_size': 1, 'output_size': 1, 'activation': 'linear'},
])

# nn.load_weights([
#     [[0.5, 0]]
# ])

print(nn)


def fill_data():
    plt.axis((-10, 10, -10, 10))

    for i in range(-10, 11):
        x = i
        y_base = function_y(x)
        y = distribute_y(y_base)

        data.append(([x], [y]))


# Draw a straight line of Neural Network prediction for x in [-10, 10]
def draw_nn_line():
    x = np.linspace(-10, 10, 10)
    y = []

    for i in range(len(x)):
        output = nn.forward([x[i]])
        y.append(output.values[0])

    plt.plot(x, y, '-b', label='NN prediction')


def draw_plot():
    fill_data()
    for x, y in data:
        plt.scatter(x[0], y[0])

    # print('Loss:', nn.get_mse_loss(data))

    draw_nn_line()

    plt.show()


def function_y(x):
    return x * 0.5


def distribute_y(y):
    return y + get_random(-3, 3)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    draw_plot()
