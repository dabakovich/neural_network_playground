import matplotlib.pyplot as plt
from visual import render_plot, init_plot, cleanup_plot

data = [
    ([1.8], [42]),
    ([1.6], [39]),
    ([1.85], [43.5]),
    ([1.85], [44]),
    ([1.7], [40]),
    ([1.65], [40]),
    ([1.9], [44]),
    ([1.75], [41]),
]

learning_rate = 0.005


class NeuralNetwork:
    def __init__(self, layers_config):
        self.layers = []


def get_output(input: list[int], w: int, b: int) -> list[int]:
    return input * w + b


def calculate_loss(data: list[tuple[int, int]], w: int, b: int):
    n = len(data)
    sum = 0

    for i in range(n):
        input_i = data[i][0]
        output_i = get_output(input_i, w, b)
        expected_output_i = data[i][1]

        sum += (output_i - expected_output_i) ** 2

    return sum / n


def calculate_slopes(data: list[tuple[int, int]], w: int, b: int):
    n = len(data)
    w_sum = 0
    b_sum = 0

    for i in range(n):
        input_i = data[i][0]
        output_i = get_output(input_i, w, b)
        expected_output_i = data[i][1]

        w_sum += 2 * (output_i - expected_output_i) * input_i
        b_sum += 2 * (output_i - expected_output_i)

    return w_sum, b_sum


def train(data: list[tuple[int, int]], epochs: int, threshold: int):
    losses = []
    w = 5
    b = 2
    current_loss = threshold

    print("Starting train...")
    print(f"Initial w: {w}, b: {b}")
    print(f"Initial loss: {calculate_loss(data, w, b)}")

    # Initialize the plot for real-time updates
    init_plot()

    while current_loss >= threshold:
        print(f"Epoch {len(losses) + 1}")
        slopes = calculate_slopes(data, w, b)
        print(f"Slopes {slopes}")

        w = w - learning_rate * slopes[0]
        b = b - learning_rate * slopes[1]

        current_loss = calculate_loss(data, w, b)
        losses.append(current_loss)

        print(f"New w: {w}, b: {b}")
        print(f"New loss: {current_loss}")

        render_plot(data, w, b, losses)

        plt.pause(1)  # Use matplotlib's pause for better integration

    # Clean up the plot when training is complete
    cleanup_plot()


train(data, 20, 0.2)
