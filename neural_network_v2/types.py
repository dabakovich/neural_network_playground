from typing import Literal, TypedDict

from shared.vector import Vector

type NeuronBiasAndWeights = Vector


type Loss = Literal["mse", "log"]

type Activator = Literal["linear", "relu", "sigmoid", "tanh"]


class LayerConfig(TypedDict):
    input_size: int
    output_size: int
    activation: Activator


class DataItem(TypedDict):
    input: list[int or float]
    output: list[int or float]
