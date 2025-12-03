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
