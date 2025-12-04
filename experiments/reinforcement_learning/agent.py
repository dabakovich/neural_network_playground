import numpy as np

from experiments.reinforcement_learning.constants import EPOCHS_PER_LEARNING
from experiments.reinforcement_learning.history import History
from neural_network_v2.neural_network import NeuralNetwork
from shared.random import rng


class TicTacToeAgent:
    name: str
    nn: NeuralNetwork
    history: History

    def __init__(self, name: str, nn: NeuralNetwork) -> None:
        self.name = name
        self.nn = nn
        self.history = History()

    def select_spot_index(self, board: np.ndarray):
        """
        Options for choosing busy cell
        - end game as regular lose
        - end game with very big punish
        - just not be able to select busy cells
        """
        output = self.nn.calculate_output(board.flatten())

        indices = np.arange(9)

        selected: int = rng.choice(indices, p=output)

        return selected

    def reset(self):
        self.history = History()

    def reinforce_learn(self):
        if self.history.reward is None:
            raise ValueError("Add reward to the history")

        return self.nn.train(
            self.history.x_list,
            self.history.y_list,
            epochs=EPOCHS_PER_LEARNING,
            batch_size=10,
        )
