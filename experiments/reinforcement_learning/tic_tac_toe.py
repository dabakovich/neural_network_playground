import logging
import matplotlib.pyplot as plt
import numpy as np

from experiments.reinforcement_learning.agent import TicTacToeAgent
from experiments.reinforcement_learning.constants import (
    AGENT_1,
    AGENT_2,
    IS_END_GAME_ON_WRONG_SPOT,
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

logging.basicConfig(level=logging.DEBUG)


def render_losses(losses: list[float]):
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning history")
    plt.show()


nn_1 = NeuralNetwork(
    [
        {"input_size": 9, "output_size": 9, "activation": "tanh"},
        {"input_size": 9, "output_size": 9, "activation": "softmax"},
    ],
    learning_rate=0.0001,
    loss_name="log",
)

nn_2 = NeuralNetwork(
    [
        {"input_size": 9, "output_size": 18, "activation": "tanh"},
        {"input_size": 18, "output_size": 18, "activation": "tanh"},
        {"input_size": 18, "output_size": 9, "activation": "softmax"},
    ],
    learning_rate=0.001,
    loss_name="log",
)

NUMBER_OF_GAMES = 50000

games = np.array([])
wrong_spots_count_list = np.array([])


def run_game(agents: list[TicTacToeAgent]):
    global games, wrong_spots_count_list
    wrong_spots_count = 0
    game = Game()
    x_turn = True
    logging.debug(f"[run_game] Initial board:\n{game}")

    while True:
        agent = agents[0 if x_turn else 1]
        opponent = agents[1 if x_turn else 0]
        board = game.board if x_turn else game.inverted_board
        move = 1 if x_turn else -1

        try:
            spot_index = agent.select_spot_index(board)
            logging.debug(
                f"[run_game] {agent.name} selected '{'x' if x_turn else 'o'}' for {spot_index + 1}"
            )

            # user_pause()

            if not game.can_make_move(spot_index):
                wrong_spots_count += 1
                if IS_END_GAME_ON_WRONG_SPOT:
                    # Add last step before exception
                    agent.history.add_step(board, spot_index)

                    raise SpotTakenError(index=spot_index)
                else:
                    agent.reinforce_learn_one_move(
                        board,
                        spot_index,
                        -0.5,
                    )
                    logging.debug(f"[run_game] Board after RL one move:\n{game}")
                    continue

            agent.history.add_step(board, spot_index)
            game.next_move(spot_index, move)

            logging.debug(f"[run_game] Board after move:\n{game}")

            x_turn = not x_turn
        except SpotTakenError as e:
            print(f"Wrong spot {e.index + 1 if e.index is not None else ''}")
            print("Ending game...")

            agent.history.finish_game(WRONG_SPOT_REWARD, True)
            opponent.history.finish_game(TIE_REWARD)

            games = np.append(games, WRONG_SPOT)
            wrong_spots_count_list = np.append(
                wrong_spots_count_list, wrong_spots_count
            )

            return

        except GameOverError as e:
            if e.winner is None:
                print("Game over, it's tie")
                agent.history.finish_game(TIE_REWARD)
                opponent.history.finish_game(TIE_REWARD)
                games = np.append(games, TIE)

            else:
                logging.info(f"Game over, winner is {'x' if e.winner == 1 else 'o'}")
                logging.debug(f"[run_game] Board after end game:\n{game}")

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

            wrong_spots_count_list = np.append(
                wrong_spots_count_list, wrong_spots_count
            )
            return


def run_games(render_every=1000):
    agent_1 = TicTacToeAgent(AGENT_1, nn_1)
    agent_2 = TicTacToeAgent(AGENT_2, nn_2)
    agents = [agent_1, agent_2]

    print("Initial agents")
    print("agent_1")
    print(agent_1.nn)
    print("agent_2")
    print(agent_2.nn)
    user_pause()

    for i in range(NUMBER_OF_GAMES):
        print(f"{'-' * 50}GAME {i + 1}{'-' * 50}")
        print(agent_1.history)
        print(agent_2.history)

        run_game(agents)

        for agent in agents:
            if agent.history.reward != 0:
                print(f"AGENT {agent.name} LEARN")
                agent.reinforce_learn()

            agent.reset()

        agents.reverse()

        # user_pause()

        if i != 0 and i % render_every == 0:
            print("agent_1")
            # print(agent_1.nn)
            print("agent_2")
            # print(agent_2.nn)

            unique_elements, counts = np.unique(games, return_counts=True)
            element_counts = dict(zip(unique_elements.tolist(), counts.tolist()))

            print(element_counts)
            print(
                f"[global] element_counts: {element_counts}, wrong spots: {wrong_spots_count_list.sum()}"
            )

            last_unique_elements, last_counts = np.unique(
                games[-render_every:], return_counts=True
            )
            last_element_counts = dict(
                zip(last_unique_elements.tolist(), last_counts.tolist())
            )

            print(
                f"[last {render_every}] element_counts: {last_element_counts}, wrong spots: {wrong_spots_count_list[-render_every:].sum()}"
            )
            user_pause()


run_games()
