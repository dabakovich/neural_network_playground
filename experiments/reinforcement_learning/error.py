from experiments.reinforcement_learning.constants import BoardValue


class TicTacToeError(Exception):
    """Custom exception for a TicTacToe game."""

    pass


class SpotTakenError(TicTacToeError):
    """Raised when a player attempts to place a mark on an already taken spot."""

    def __init__(self, message="This spot is already taken.", index=None):
        super().__init__(message)
        self.index = index


class GameOverError(TicTacToeError):
    """Raised when an action is attempted after the game has ended."""

    def __init__(
        self, message="The game is already over.", winner: BoardValue | None = None
    ):
        super().__init__(message)
        self.winner = winner
