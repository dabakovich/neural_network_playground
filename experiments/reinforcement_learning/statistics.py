import numpy as np

from experiments.reinforcement_learning.constants import GameResult


class Statistics:
    def __init__(self) -> None:
        self.games = np.array([])
        self.agent_1_wrong_spots_count = np.array([])
        self.agent_2_wrong_spots_count = np.array([])

    def append_game_result(self, game_result: GameResult | str):
        self.games = np.append(self.games, game_result)

    def append_agent_1_wrong_spots_count(self, wrong_spots_count: int):
        self.agent_1_wrong_spots_count = np.append(
            self.agent_1_wrong_spots_count, wrong_spots_count
        )

    def append_agent_2_wrong_spots_count(self, wrong_spots_count: int):
        self.agent_2_wrong_spots_count = np.append(
            self.agent_2_wrong_spots_count, wrong_spots_count
        )
