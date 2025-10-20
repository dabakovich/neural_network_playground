from typing import TypedDict

from shared.matrix import Matrix
from shared.vector import Vector


type NeuronBiasAndWeights = Vector

type Layer = Matrix

type InputVector = Vector | list[float | int]


class LayerConfig(TypedDict):
    input_size: int
    output_size: int


class DataItem(TypedDict):
    input: list[int or float]
    output: list[int or float]
