import logging

import matplotlib.pyplot as plt
import numpy as np

from experiments.reinforcement_learning.agent import TicTacToeAgent
from experiments.reinforcement_learning.constants import AgentName
from experiments.reinforcement_learning.game_helpers import get_int_input, user_pause
from experiments.reinforcement_learning.game_runner import run_game
from experiments.reinforcement_learning.statistics import Statistics
from experiments.reinforcement_learning.visual import init_plt, render_statistics
from neural_network_v2.neural_network import NeuralNetwork

logging.basicConfig(level=logging.DEBUG)


def render_losses(losses: list[float]):
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning history")
    plt.show()


# + Higher LR
# + Try batch size 1!!!
# + Add count of game
# + !!! Implement safe softmax. It turned on that each time I was getting NaN error, the model actually was starting becoming smarter
# We need to reward more quick game end
# + Skip count of games
# Show statistics in percentage
# + Render statistics in graph
# Test top 1 with higher LR

nn_1 = NeuralNetwork(
    [
        {"input_size": 9, "output_size": 18, "activation": "tanh"},
        {"input_size": 18, "output_size": 18, "activation": "tanh"},
        {"input_size": 18, "output_size": 9, "activation": "softmax"},
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
    learning_rate=0.0001,
    loss_name="log",
)

NUMBER_OF_GAMES = 100000
DEFAULT_GAMES_TO_SKIP = 1000
STATISTICS_BATCH_SIZE = 1000


def run_games():
    init_plt()
    games_to_skip = DEFAULT_GAMES_TO_SKIP
    statistics = Statistics()
    agent_1 = TicTacToeAgent(AgentName.AGENT_1, nn_1)
    agent_2 = TicTacToeAgent(AgentName.AGENT_2, nn_2)
    agents = [agent_1, agent_2]

    print("Initial agents")
    print("agent_1")
    print(agent_1.nn)
    print("agent_2")
    print(agent_2.nn)

    games_to_skip = get_int_input(
        "Enter count of games to skip",
        max_value=NUMBER_OF_GAMES,
        default_value=DEFAULT_GAMES_TO_SKIP,
    )

    for i in range(NUMBER_OF_GAMES):
        print(f"{'-' * 50}GAME {i + 1}{'-' * 50}")
        print(agent_1.history)
        print(agent_2.history)

        if i != 0 and i % 1000 == 0:
            render_statistics(statistics, 100, 5000)

        run_game(agents, statistics)

        for agent in agents:
            if agent.history.reward != 0:
                print(f"AGENT {agent.name} LEARN")
                agent.reinforce_learn()

            agent.reset()

        agents.reverse()

        # user_pause()

        games_to_skip -= 1

        if games_to_skip == 0 or i == NUMBER_OF_GAMES:
            print(f"Game {i + 1} results")
            print("agent_1")
            # print(agent_1.nn)
            print("agent_2")
            # print(agent_2.nn)

            unique_elements, counts = np.unique(statistics.games, return_counts=True)
            element_counts = dict(zip(unique_elements.tolist(), counts.tolist()))

            print(element_counts)
            print(f"[global] element_counts: {element_counts}")
            print(
                f"[global] wrong spots: agent 1 - {statistics.agent_1_wrong_spots_count.sum()}; agent 2 - {statistics.agent_2_wrong_spots_count.sum()}"
            )

            last_unique_elements, last_counts = np.unique(
                statistics.games[-STATISTICS_BATCH_SIZE:], return_counts=True
            )
            last_element_counts = dict(
                zip(last_unique_elements.tolist(), last_counts.tolist())
            )

            print(
                f"[last {STATISTICS_BATCH_SIZE}] element_counts: {last_element_counts}"
            )
            print(
                f"[last {STATISTICS_BATCH_SIZE}] wrong spots: agent 1 -  {statistics.agent_1_wrong_spots_count[-STATISTICS_BATCH_SIZE:].sum()}; agent 2 - {statistics.agent_2_wrong_spots_count[-STATISTICS_BATCH_SIZE:].sum()}"
            )

            games_to_skip = get_int_input(
                "Enter count of games to skip",
                max_value=NUMBER_OF_GAMES - i,
                default_value=DEFAULT_GAMES_TO_SKIP,
            )


run_games()
