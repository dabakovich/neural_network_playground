import numpy as np

from experiments.reinforcement_learning.constants import BoardValue


def init_board():
    board = np.empty(9)
    board.fill(BoardValue.EMPTY)
    return board


def check_win(board: np.ndarray) -> BoardValue | None:
    """
    Checks if the specified player has won the Tic-Tac-Toe game.

    Args:
        board (np.ndarray): A 3x3 NumPy array representing the Tic-Tac-Toe board.

    Returns:
        bool: True if the player has won, False otherwise.
    """
    # Check rows
    for row in board:
        if np.all(row == BoardValue.X):
            return BoardValue.X
        if np.all(row == BoardValue.O):
            return BoardValue.O

    # Check columns
    for col_idx in range(3):
        if np.all(board[:, col_idx] == BoardValue.X):
            return BoardValue.X
        if np.all(board[:, col_idx] == BoardValue.O):
            return BoardValue.O

    # Check main diagonal
    if np.all(np.diag(board) == BoardValue.X):
        return BoardValue.X
    if np.all(np.diag(board) == BoardValue.O):
        return BoardValue.O

    # Check anti-diagonal
    if np.all(np.diag(np.fliplr(board)) == BoardValue.X):
        return BoardValue.X
    if np.all(np.diag(np.fliplr(board)) == BoardValue.O):
        return BoardValue.O

    return None


def check_tie(board):
    return np.all(board != BoardValue.EMPTY)


def get_int_input(
    prompt_message: str,
    min_value: int | None = None,
    max_value: int | None = None,
    default_value: int | None = None,
):
    """
    Waits for the user to input a valid number and returns it.
    Keeps prompting until a valid number is entered.
    """
    while True:
        user_input = input(
            f"{prompt_message}{f' ({default_value})' if default_value else ''}: "
        )
        try:
            if not user_input and default_value:
                return default_value

            number = int(user_input)  # Attempt to convert input to a int

            if min_value:
                if number < min_value:
                    raise ValueError(
                        f"Number should be higher than or equal {min_value}"
                    )

            if max_value:
                if number > max_value:
                    raise ValueError(f"Number should be less than or equal {max_value}")

            return number  # Return the number if conversion is successful
        except ValueError:
            print("Invalid input. Please enter a valid number.")


def user_pause():
    input("Pause...")
