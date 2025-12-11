import logging
import numpy as np

from experiments.reinforcement_learning.constants import REWARD_SHIFTS_LIST
from neural_network_v2.neural_network import NeuralNetwork
from shared.random import rng

logging.basicConfig(level=logging.DEBUG)

arr = np.arange(0, 10)

print(arr)
print(arr[-5:])
