import matplotlib.pyplot as plt
import numpy as np

from experiments.reinforcement_learning.constants import GameResult
from experiments.reinforcement_learning.statistics import Statistics

fig = None
statistics_axes = None


def init_plt():
    global fig, statistics_axes

    plt.ion()

    fig = plt.figure(figsize=(16, 8))
    statistics_axes = fig.add_subplot(1, 1, 1)


def render_statistics(
    statistics: Statistics,
    batch_size=100,
    show_last: int | None = None,
):
    global fig, statistics_axes

    if not fig or not statistics_axes:
        raise ValueError("Plot was not initialized")

    statistics_axes.clear()

    games = statistics.games

    if show_last:
        games = games[-show_last:]

    batches_count = int(len(games) / batch_size)
    shift = (
        int((len(statistics.games) - show_last) / batch_size)
        if show_last and len(statistics.games) > show_last
        else 0
    )
    # x = np.arange(0, batches_count)
    x = np.arange(shift, batches_count + shift)

    batched_games = games.reshape((batches_count, -1))

    for result in list(GameResult):
        y = (batched_games == result).sum(axis=1)
        statistics_axes.plot(x, y, label=result)

    # plt.xlim((0, 1000))

    statistics_axes.set_xlabel(f"Count of {batch_size} games")
    statistics_axes.set_ylabel(f"Game results count per {batch_size} games")
    statistics_axes.set_title(f"Statistics ({len(statistics.games)} games)")

    statistics_axes.legend()

    # plt.show()
    fig.canvas.draw()
    fig.canvas.flush_events()
