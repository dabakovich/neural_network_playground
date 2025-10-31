from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from neural_network_v2.types import DataItem
from shared.types import InputVector
from shared.vector import Vector

# Global variables to persist figure and axes
fig = None
ax = None
bx = None


def set_labels():
    # Set labels (these don't change)
    ax.set_xlabel("X_1")
    ax.set_ylabel("X_2")
    ax.set_zlabel("Y")

    bx.set_xlabel("Iterations")
    bx.set_ylabel("Loss")


def init_plot():
    """Initialize the plot for real-time updates"""
    global fig, ax, bx

    # Enable interactive mode
    plt.ion()

    # Create figure and axes once
    fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(1, 2, 1)
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    bx = fig.add_subplot(1, 2, 2)

    print("Plot initialized for real-time updates...")


def render_plot(
    data: list[tuple[int, int]],
    get_nn_output: Callable[[InputVector], list[Vector]],
    losses: list[int],
):
    """Update the existing plot with new data"""
    global fig, ax

    if fig is None or ax is None:
        print("Warning: Plot not initialized. Call init_plot() first.")
        return

    print("Updating plot...")

    # Clear the axes but keep the labels
    ax.clear()
    set_labels()

    # Add points
    points = np.array(data)
    ax.scatter(points[:, 0], points[:, 1], s=100, marker="o", label="Points")

    min_x = min(points[:, 0])
    max_x = max(points[:, 0])

    # Add line
    X = np.linspace(min_x, max_x)
    Y = []
    for x in X:
        output = get_nn_output([x])
        Y.append(output[-1].values[0])

    ax.plot(X, Y, label="Neural network prediction")
    ax.legend()

    render_losses(losses)

    # Draw the updates
    fig.canvas.draw()
    fig.canvas.flush_events()


# Renders 3D plot of neural network output where x and y are inputs and z is output
def render_nn_output(
    data: list[DataItem],
    get_nn_output: Callable[[InputVector], Vector],
):
    global fig, ax

    if fig is None or ax is None:
        print("Warning: Plot not initialized. Call init_plot() first.")
        return

    ax.clear()
    set_labels()

    # Extract dataset points
    if data and len(data) > 0:
        data_x = [item["input"][0] for item in data]
        data_y = [item["input"][1] for item in data]
        data_z = [item["output"][0] for item in data]

    # Plot dataset points
    ax.scatter(
        data_x, data_y, data_z, color="red", marker="o", s=100, label="Dataset points"
    )

    min = -0.5
    max = 1.5
    X = np.linspace(min, max, 100)
    Y = np.linspace(min, max, 100)

    X, Y = np.meshgrid(X, Y)

    Z = [get_nn_output([x, y]).values[0] for x, y in zip(X.flatten(), Y.flatten())]

    Z = np.array(Z).reshape(X.shape)

    ax.plot_surface(X, Y, Z)

    # Draw the updates
    fig.canvas.draw()
    fig.canvas.flush_events()


def render_losses(losses: list[int]):
    """Update the existing plot with new data"""
    global fig, bx

    if fig is None:
        print("Warning: Plot not initialized. Call init_plot() first.")
        return

    bx.clear()
    set_labels()

    # Render loss graph
    loss_points = np.array(losses)
    loss_X = range(0, len(losses))
    loss_Y = loss_points
    bx.plot(loss_X, loss_Y, label=f"Current loss: {losses[-1]:.4f}")
    bx.legend()

    # Draw the updates
    fig.canvas.draw()
    fig.canvas.flush_events()


def cleanup_plot():
    """Clean up the plot when done"""
    global fig, ax, bx

    plt.ioff()  # Turn off interactive mode
    if fig is not None:
        plt.pause(60)
        plt.close(fig)

    fig = None
    ax = None
    bx = None
    print("Plot cleanup completed.")
