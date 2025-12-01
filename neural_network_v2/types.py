from typing import Literal, NotRequired, TypedDict

import numpy.typing as npt

type Loss = Literal["mse", "log"]

type Activator = Literal["linear", "relu", "sigmoid", "tanh", "softmax"]

# type Vector = np.ndarray[np.float64, np._1DShapeT]
type Vector = npt.NDArray
# type Vector = np.ndarray[np._1DShapeT]

type Matrix = npt.NDArray
# type Matrix = np.ndarray[np._2DShapeT_co]

type NeuronBiasAndWeights = Vector


class LayerConfig(TypedDict):
    input_size: int
    output_size: int
    activation: NotRequired[Activator]
