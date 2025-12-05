import logging
import numpy as np

from experiments.reinforcement_learning.constants import REWARD_SHIFTS_LIST
from neural_network_v2.neural_network import NeuralNetwork
from shared.random import rng

logging.basicConfig(level=logging.DEBUG)


nn = NeuralNetwork(
    [
        {"input_size": 9, "output_size": 18, "activation": "relu"},
        {"input_size": 18, "output_size": 18, "activation": "relu"},
        {"input_size": 18, "output_size": 9, "activation": "softmax"},
    ],
    learning_rate=0.01,
    loss_name="log",
)

board_1 = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0])
board_2 = np.array([1, 0, 1, 0, 1, 0, 0, 1, 0])
board_3 = np.array([1, 0, 1, 0, 1, 0, 0, 1, 0])

boards = np.array([board_1, board_2, board_3])
ideal_y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 2.5])

grad = REWARD_SHIFTS_LIST[-3:]

print(grad)
