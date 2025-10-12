from ast import Tuple


data = [
    (1.8, 42),
    (1.6, 39),
    (1.85, 43.5),
    (1.85, 44),
    (1.7, 40),
    (1.65, 40),
    (1.9, 44),
    (1.75, 41),
]

learning_rate = 0.01


def get_output(input: int, w: int, b: int):
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


def train(data: list[tuple[int, int]], epochs: int):
    w = 5
    b = 2

    print("Starting train...")
    print(f"Initial w: {w}, b: {b}")
    print(f"Initial loss: {calculate_loss(data, w, b)}")

    for i in range(epochs):
        print(f"Epoch {i + 1}")
        slopes = calculate_slopes(data, w, b)
        print(f"Slopes {slopes}")

        w = w - learning_rate * slopes[0]
        b = b - learning_rate * slopes[1]

        print(f"New w: {w}, b: {b}")
        print(f"New loss: {calculate_loss(data, w, b)}")


train(data, 20)
