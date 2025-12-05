import numpy as np

from experiments.reinforcement_learning.constants import GameResult


class Statistics:
    def __init__(self) -> None:
        self.games = np.array([])
        self.wrong_spots_count_list = np.array([])

    def append_game_result(self, game_result: GameResult | str):
        self.games = np.append(self.games, game_result)

    def append_wrong_spots_count(self, wrong_spots_count: int):
        self.wrong_spots_count_list = np.append(
            self.wrong_spots_count_list, wrong_spots_count
        )
