from .matrix import Matrix
from .vector import Vector

type InputVector = Vector | list[float | int]

type InputMatrix = Matrix | list[InputVector]
