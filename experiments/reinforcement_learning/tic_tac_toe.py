import matplotlib.pyplot as plt
import numpy as np

from experiments.reinforcement_learning.error import GameOverError, SpotTakenError
from experiments.reinforcement_learning.game import Game
from experiments.reinforcement_learning.game_helpers import get_int_input
from neural_network_v2.neural_network import NeuralNetwork


def render_losses(losses: list[float]):
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning history")
    plt.show()


class TicTacToeAgent:
    nn: NeuralNetwork

    def __init__(self, nn: NeuralNetwork) -> None:
        self.nn = nn

    def get_output(self, state: np.ndarray):
        """
        Options for choosing busy cell
        - end game as regular lose
        - end game with very big punish
        - just not be able to select busy cells
        """
        output = self.nn.calculate_output(state.flatten())

        indices = np.arange(9)

        selected = np.random.choice(indices, p=output)

        return selected


nn_1 = NeuralNetwork(
    [
        {"input_size": 9, "output_size": 9, "activation": "softmax"},
    ],
    learning_rate=0.01,
    loss_name="log",
)

nn_2 = NeuralNetwork(
    [
        {"input_size": 9, "output_size": 9, "activation": "softmax"},
    ],
    learning_rate=0.01,
    loss_name="log",
)


def run_game():
    game = Game()
    x_turn = True
    print(game)

    while True:
        try:
            next_input = (
                get_int_input(f"Enter index for `{'x' if x_turn else 'o'}`: ") - 1
            )
            game.next_move(next_input, 1 if x_turn else -1)
            print(game)

            x_turn = not x_turn
        except SpotTakenError as e:
            print(f"Wrong spot {e.index + 1 if e.index is not None else ''}, try again")
        except GameOverError as e:
            if e.winner is None:
                print("Game over, it's tie")
            else:
                print(f"Game over, winner is {'x' if e.winner == 1 else 'o'}")
            return


run_game()
