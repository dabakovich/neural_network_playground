import math

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from neural_network_v2.neural_network import NeuralNetwork
from neural_network_v2.types import LayerConfig
from shared.matrix import Matrix

np.set_printoptions(precision=3, floatmode="maxprec")

file_path = "datasets/shoes.csv"
# file_path = "datasets/xor.csv"

ds = pd.read_csv(file_path)


y_list = np.array(
    [
        [1, 0],
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1],
    ]
)

predicted_y_list = np.array(
    [
        [0.9, 0],
        [0.9, 0],
        [0.9, 0],
        [0.9, 0],
        [0, 0.9],
    ]
)

a = np.array([0.5, 1, 3, 1, 0.1])

s = np.exp(a) / np.exp(a).sum()

# s = np.array([0.8, 0.2])
s = np.array([0.7, 0.2, 0.1])

# y = np.array([1, 0])
y = np.array([1, 0, 0])

loss = (s - y).dot(s - y)

print(f"loss\n{loss}")

d_loss_d_S = (s - y) * 2
print(f"d_loss_d_S[:, np.newaxis]\n{d_loss_d_S[:, np.newaxis]}")

n = len(s)
identity = np.eye(n)

d_S_d_a = s[:, np.newaxis] * (identity - s[np.newaxis, :])
print(f"d_S_d_a\n{d_S_d_a}")

d_loss_d_a = d_loss_d_S[:, np.newaxis] * d_S_d_a

print(f"d_loss_d_a\n{d_loss_d_a}")
print(f"d_loss_d_a.T\n{d_loss_d_a.T}")


# input = np.array([1, 2, 3])
input = np.array([1, 2, 3, 4])

input_matrix = input[np.newaxis, :]
new_rows = np.tile(input_matrix[0], (len(y) - 1, 1))
input_matrix = np.vstack((input_matrix, new_rows))

print(f"input_matrix\n{input_matrix}")

d_loss_d_w = d_loss_d_a.T @ input_matrix

print(f"d_loss_d_w\n{d_loss_d_w}")

print(f"np.ones((len(y), 1))\n{np.ones((len(y), 1))}")

d_loss_d_b = d_loss_d_a.T @ np.ones((len(y), 1))

print(f"d_loss_d_b\n{d_loss_d_b}")

weights = np.array(
    [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [4, 3, 2, 1],
    ]
)

print(f"weights\n{weights}")


next_gradient_matrix = d_loss_d_a @ weights

print(f"next_gradient_matrix\n{next_gradient_matrix}")

next_gradient = np.sum(next_gradient_matrix, axis=0)

print(f"next_gradient\n{next_gradient}")
