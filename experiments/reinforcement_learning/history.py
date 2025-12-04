import logging
import numpy as np

from experiments.reinforcement_learning.constants import (
    REWARDS_LIST,
    WRONG_SPOT_REWARD_LIST,
)


class History:
    x_steps_list: list[np.ndarray]  # list of 1d vectors
    indices: list[int]  # list of 1d vectors

    # 2d matrix, ready for training
    x_list: np.ndarray

    # 2d matrix, with zeros and rewards on the relevant indices
    reward_shifts_list: np.ndarray
    reward: int | None

    def __init__(self) -> None:
        self.x_steps_list = []
        self.indices = []

    def add_step(self, board: np.ndarray, selected_spot_index: int):
        self.x_steps_list.append(board.copy())
        self.indices.append(selected_spot_index)

    def finish_game(self, reward: int, is_wrong_spot=False):
        self.reward = reward

        self.x_list = np.array(self.x_steps_list)

        reward_gradient = (
            WRONG_SPOT_REWARD_LIST if is_wrong_spot else REWARDS_LIST
        ) * reward

        reward_gradient = reward_gradient[-len(self.indices) :]

        y_list = np.zeros((len(self.indices), 9))

        for index in range(len(self.indices)):
            y_list[index][self.indices[index]] = reward_gradient[index]

        self.reward_shifts_list = y_list

        logging.debug(f"[History] reward_shifts_list:\n{self.reward_shifts_list}")

    def __str__(self) -> str:
        steps_str = "\n".join(
            [
                f"{self.x_steps_list[i]} - {self.indices[i]}"
                for i in range(len(self.x_steps_list))
            ]
        )
        return f"Steps:\n{steps_str}"
