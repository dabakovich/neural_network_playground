import matplotlib.pyplot as plt
import numpy as np

from experiments.reinforcement_learning.agent import TicTacToeAgent
from experiments.reinforcement_learning.constants import (
    AGENT_1,
    AGENT_2,
    LOOSER_REWARD,
    TIE,
    TIE_REWARD,
    WINNER_REWARD,
    WRONG_SPOT,
    WRONG_SPOT_REWARD,
)
from experiments.reinforcement_learning.error import GameOverError, SpotTakenError
from experiments.reinforcement_learning.game import Game
from experiments.reinforcement_learning.game_helpers import user_pause
from neural_network_v2.neural_network import NeuralNetwork


def render_losses(losses: list[float]):
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning history")
    plt.show()


nn_1 = NeuralNetwork(
    [
        {"input_size": 9, "output_size": 9, "activation": "relu"},
        {"input_size": 9, "output_size": 9, "activation": "softmax"},
    ],
    learning_rate=0.001,
    loss_name="log",
)

nn_2 = NeuralNetwork(
    [
        {"input_size": 9, "output_size": 9, "activation": "relu"},
        {"input_size": 9, "output_size": 9, "activation": "softmax"},
    ],
    learning_rate=0.001,
    loss_name="log",
)

NUMBER_OF_GAMES = 10000

games = np.array([])


def run_game(agents: list[TicTacToeAgent]):
    global games
    game = Game()
    x_turn = True
    print(game)

    while True:
        agent = agents[0 if x_turn else 1]
        opponent = agents[1 if x_turn else 0]
        board = game.board if x_turn else game.inverted_board
        move = 1 if x_turn else -1

        try:
            # board = game.board if x_turn else game.inverted_board

            # print(f"board: {board}, {'x' if x_turn else 'o'} turn")

            # X TURN
            spot_index = agent.select_spot_index(board)
            print(f"{agent.name} selected {spot_index + 1}")
            agent.history.add_step(board, spot_index)
            game.next_move(spot_index, move)

            print(game)

            x_turn = not x_turn
        except SpotTakenError as e:
            print(
                f"Wrong spot {e.index + 1 if e.index is not None else ''}, ending game with bigger punish"
            )

            agent.history.finish_game(WRONG_SPOT_REWARD)
            opponent.history.finish_game(TIE_REWARD)

            games = np.append(games, WRONG_SPOT)

            return
        except GameOverError as e:
            if e.winner is None:
                print("Game over, it's tie")
                agent.history.finish_game(TIE_REWARD)
                opponent.history.finish_game(TIE_REWARD)
                games = np.append(games, TIE)

            else:
                print(f"Game over, winner is {'x' if e.winner == 1 else 'o'}")
                is_agent_winner = e.winner == move
                agent.history.finish_game(
                    WINNER_REWARD if is_agent_winner else LOOSER_REWARD
                )
                opponent.history.finish_game(
                    WINNER_REWARD if not is_agent_winner else LOOSER_REWARD
                )

                games = np.append(
                    games, agent.name if is_agent_winner else opponent.name
                )
            return


def run_games():
    agent_1 = TicTacToeAgent(AGENT_1, nn_1)
    agent_2 = TicTacToeAgent(AGENT_2, nn_2)
    agents = [agent_1, agent_2]

    for i in range(NUMBER_OF_GAMES):
        print(f"---GAME {i + 1}---")
        print(agent_1.history)
        print(agent_2.history)

        run_game(agents)

        for agent in agents:
            if agent.history.reward != 0:
                print(f"AGENT {agent.name} LEARN")
                agent.reinforce_learn()

            agent.reset()

        agents.reverse()

        if i != 0 and i % 500 == 0:
            unique_elements, counts = np.unique(games, return_counts=True)
            element_counts = dict(zip(unique_elements.tolist(), counts.tolist()))

            print(element_counts)
            user_pause()


run_games()
