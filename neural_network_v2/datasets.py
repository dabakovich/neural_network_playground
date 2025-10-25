from .types import DataItem

shoes_dataset: list[DataItem] = [
    {"input": [1.8], "output": [42]},
    {"input": [1.6], "output": [39]},
    {"input": [1.85], "output": [43.5]},
    {"input": [1.85], "output": [44]},
    {"input": [1.7], "output": [40]},
    {"input": [1.65], "output": [40]},
    {"input": [1.9], "output": [44]},
    {"input": [1.75], "output": [41]},
]

and_dataset: list[DataItem] = [
    {"input": [0, 0], "output": [0]},
    {"input": [0, 1], "output": [0]},
    {"input": [1, 0], "output": [0]},
    {"input": [1, 1], "output": [1]},
]
