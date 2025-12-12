import matplotlib.pyplot as plt
import numpy as np

from experiments.reinforcement_learning.constants import GameResult
from experiments.reinforcement_learning.statistics import Statistics

fig = None
game_results_axes = None
wrong_spots_axes = None

result_color_map = {
    GameResult.AGENT_1_WIN: "red",
    GameResult.AGENT_2_WIN: "blue",
    GameResult.TIE: "green",
    GameResult.AGENT_1_WRONG_SPOT: None,
    GameResult.AGENT_2_WRONG_SPOT: None,
}


def init_plt():
    global fig, game_results_axes, wrong_spots_axes

    plt.ion()

    fig = plt.figure(figsize=(16, 10))
    game_results_axes = fig.add_subplot(4, 1, (1, 3))
    wrong_spots_axes = fig.add_subplot(4, 1, 4)


def render_statistics(
    statistics: Statistics,
    batch_size=100,
    show_last: int | None = None,
):
    global fig, game_results_axes, wrong_spots_axes

    if not fig or not game_results_axes or not wrong_spots_axes:
        raise ValueError("Plot was not initialized")

    game_results_axes.clear()
    wrong_spots_axes.clear()

    games = statistics.games
    agent_1_wrong_spots = statistics.agent_1_wrong_spots_count
    agent_2_wrong_spots = statistics.agent_2_wrong_spots_count

    if show_last:
        games = games[-show_last:]
        agent_1_wrong_spots = agent_1_wrong_spots[-show_last:]
        agent_2_wrong_spots = agent_2_wrong_spots[-show_last:]

    batches_count = int(len(games) / batch_size)
    shift = (
        int((len(statistics.games) - show_last) / batch_size)
        if show_last and len(statistics.games) > show_last
        else 0
    )
    x = np.arange(shift, batches_count + shift)

    batched_games = games.reshape((-1, batch_size))
    batched_agent_1_wrong_spots = agent_1_wrong_spots.reshape((-1, batch_size))
    batched_agent_2_wrong_spots = agent_2_wrong_spots.reshape((-1, batch_size))

    for result in list(GameResult):
        y = (batched_games == result).sum(axis=1)
        game_results_axes.plot(x, y, label=result, color=result_color_map[result])

    batched_agent_1_wrong_spots_y = batched_agent_1_wrong_spots.sum(axis=1)
    batched_agent_2_wrong_spots_y = batched_agent_2_wrong_spots.sum(axis=1)

    wrong_spots_axes.plot(
        x, batched_agent_1_wrong_spots_y, label="Agent 1", color="red"
    )
    wrong_spots_axes.plot(
        x, batched_agent_2_wrong_spots_y, label="Agent 2", color="blue"
    )

    # plt.xlim((0, 1000))

    game_results_axes.set_ylabel(f"Game results/{batch_size} games")
    game_results_axes.set_title(f"Statistics ({len(statistics.games)} games)")

    wrong_spots_axes.set_xlabel(f"Count of {batch_size} games")
    wrong_spots_axes.set_ylabel(f"Wrong spots/{batch_size} games")

    game_results_axes.legend()
    wrong_spots_axes.legend()

    # plt.show()
    fig.canvas.draw()
    fig.canvas.flush_events()
