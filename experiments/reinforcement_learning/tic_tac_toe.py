import logging

import matplotlib.pyplot as plt
import numpy as np

from experiments.reinforcement_learning.agent import TicTacToeAgent
from experiments.reinforcement_learning.constants import GameResult
from experiments.reinforcement_learning.game_helpers import user_pause
from experiments.reinforcement_learning.game_runner import run_game
from experiments.reinforcement_learning.statistics import Statistics
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


def run_games(render_every=1000):
    statistics = Statistics()
    agent_1 = TicTacToeAgent(GameResult.AGENT_1, nn_1)
    agent_2 = TicTacToeAgent(GameResult.AGENT_2, nn_2)
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

        run_game(agents, statistics)

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

            unique_elements, counts = np.unique(statistics.games, return_counts=True)
            element_counts = dict(zip(unique_elements.tolist(), counts.tolist()))

            print(element_counts)
            print(
                f"[global] element_counts: {element_counts}, wrong spots: {statistics.wrong_spots_count_list.sum()}"
            )

            last_unique_elements, last_counts = np.unique(
                statistics.games[-render_every:], return_counts=True
            )
            last_element_counts = dict(
                zip(last_unique_elements.tolist(), last_counts.tolist())
            )

            print(
                f"[last {render_every}] element_counts: {last_element_counts}, wrong spots: {statistics.wrong_spots_count_list[-render_every:].sum()}"
            )
            user_pause()


run_games()
