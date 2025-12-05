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


def get_int_input(prompt_message="Please enter a number from 1 to 9: "):
    """
    Waits for the user to input a valid number and returns it.
    Keeps prompting until a valid number is entered.
    """
    while True:
        user_input = input(prompt_message)
        try:
            number = int(user_input)  # Attempt to convert input to a int
            if number < 1 or number > 9:
                raise ValueError("Number should be between 1 and 9")

            return number  # Return the number if conversion is successful
        except ValueError:
            print("Invalid input. Please enter a valid number.")


def user_pause():
    input("Pause...")
