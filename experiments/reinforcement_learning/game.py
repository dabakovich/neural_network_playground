from typing import Literal

import numpy as np

from experiments.reinforcement_learning.error import GameOverError, SpotTakenError
from experiments.reinforcement_learning.game_helpers import (
    check_tie,
    check_win,
    init_board,
)


str_mapper = np.vectorize(lambda x: "x" if x == 1 else "o" if x == -1 else " ")
invert_mapper = np.vectorize(lambda x: -x)


class Game:
    board: np.ndarray

    def __init__(self) -> None:
        self.board = init_board()

    @property
    def board_2d(self):
        return self.board.reshape((3, -1))

    @property
    def inverted_board(self):
        return invert_mapper(self.board)

    def next_move(self, index: int, value: Literal[1, -1]):
        """
        index - cell index
        value - `1` for "X", `-1` for "O"
        """
        if self.board[index] != 0:
            raise SpotTakenError(index=index)

        self.board[index] = value

        self.check_end_game()

    def check_end_game(self):
        """
        Returns tuple[bool, int]

        bool is status of the game (True if game ended, False if not)
        int shows who's won, or tie
        `1` - "X" won
        `-1` - "O" won
        `0` - tie
        """
        winner = check_win(self.board_2d)

        if winner is not None:
            raise GameOverError(winner=winner)

        if check_tie(self.board):
            raise GameOverError()

    def __str__(self):
        return "\n".join([row.__str__() for row in str_mapper(self.board_2d)])
