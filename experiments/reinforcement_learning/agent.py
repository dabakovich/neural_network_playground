import logging
import numpy as np

from experiments.reinforcement_learning.constants import (
    BATCH_SIZE,
    EPOCHS_PER_LEARNING,
    RANDOM_VALUE_PROBABILITY,
    RL_ONE_MOVE_EPOCHS_PER_LEARNING,
    TOP_K,
)
from experiments.reinforcement_learning.history import History
from neural_network_v2.neural_network import NeuralNetwork
from neural_network_v2.types import Vector
from shared.random import rng


class TicTacToeAgent:
    name: str
    nn: NeuralNetwork
    history: History

    def __init__(self, name: str, nn: NeuralNetwork) -> None:
        self.name = name
        self.nn = nn
        self.history = History()

    def select_spot_index(
        self,
        board: np.ndarray,
        top_k=TOP_K,
        random_value_p=RANDOM_VALUE_PROBABILITY,
    ) -> int:
        """
        Options for choosing busy cell
        - end game as regular lose
        - end game with very big punish
        - just not be able to select busy cells
        """
        output = self.nn.calculate_output(board.flatten())

        logging.debug(f"[TicTacToeAgent] output: {output}")

        if random_value_p > 0:
            random_v = rng.random()
            if random_v < random_value_p:
                selected = int(rng.integers(low=0, high=8))
                logging.debug(f"[TicTacToeAgent] random selected: {selected + 1}")
                return selected

        idx = np.argpartition(output, -top_k)[-top_k:]

        values = output[idx]
        probabilities = values / values.sum()

        selected = rng.choice(idx, p=probabilities)

        logging.debug(f"[TicTacToeAgent] choice selected: {selected + 1}")

        return selected

    def reset(self):
        self.history = History()

    def reinforce_learn(self):
        if self.history.reward is None:
            raise ValueError("Add reward to the history")

        logging.debug(
            f"[TicTacToeAgent] RL for {self.name}, reward: {self.history.reward}"
        )
        logging.debug(f"[TicTacToeAgent] inputs:\n{self.history.x_list}")

        y_list = np.apply_along_axis(self.nn.calculate_output, 1, self.history.x_list)
        logging.debug(f"[TicTacToeAgent] outputs:\n{y_list}")

        y_list += self.history.reward_shifts_list
        logging.debug(f"[TicTacToeAgent] shifted outputs:\n{y_list}")

        losses = self.nn.train(
            self.history.x_list,
            y_list,
            epochs=EPOCHS_PER_LEARNING,
            batch_size=BATCH_SIZE,
        )

        after_outputs = np.apply_along_axis(
            self.nn.calculate_output, 1, self.history.x_list
        )
        logging.debug(f"[TicTacToeAgent] after train outputs:\n{after_outputs}")

        return losses

    def reinforce_learn_one_move(
        self,
        x: Vector,
        y_index: int,
        reward_shift: float,
        epochs=RL_ONE_MOVE_EPOCHS_PER_LEARNING,
    ):
        model_output = self.nn.calculate_output(x)
        logging.debug(f"[TicTacToeAgent] model_output:\n{model_output}")

        # model_output -= reward_shift
        # model_output[y_index] += reward_shift * 2
        model_output[y_index] += reward_shift
        logging.debug(f"[TicTacToeAgent] model_output with shift:\n{model_output}")

        losses = self.nn.train(
            x[np.newaxis, :],
            model_output[np.newaxis, :],
            epochs=epochs,
            batch_size=BATCH_SIZE,
        )

        after_output = self.nn.calculate_output(x)

        logging.debug(f"[TicTacToeAgent] after train outputs:\n{after_output}")

        return losses
