from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from neural_network_v2.types import Vector

# Global variables to persist figure and axes
fig = None
ax_3d = None
bx = None


def set_labels():
    global fig, ax_3d, bx

    if ax_3d is None or bx is None:
        raise ValueError("Init ax_3d and bx first")

    # Set labels (these don't change)
    ax_3d.set_xlabel("X_1")
    ax_3d.set_ylabel("X_2")
    ax_3d.set_zlabel("Y")

    bx.set_xlabel("Iterations")
    bx.set_ylabel("Loss")


def init_plot():
    """Initialize the plot for real-time updates"""
    global fig, ax_3d, bx

    # Enable interactive mode
    plt.ion()

    # Create figure and axes once
    fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(1, 2, 1)
    ax_3d = fig.add_subplot(1, 2, 1, projection="3d")
    bx = fig.add_subplot(1, 2, 2)

    print("Plot initialized for real-time updates...")


# Renders 3D plot of neural network output where x and y are inputs and z is output
def render_nn_output_for_two_inputs(
    x: np.ndarray,
    y: np.ndarray,
    get_nn_output: Callable[[Vector], Vector],
):
    global fig, ax_3d

    if fig is None or ax_3d is None:
        print("Warning: Plot not initialized. Call init_plot() first.")
        return

    ax_3d.clear()
    set_labels()

    data_x = x[:, 0]
    data_y = x[:, 1]
    data_z = y[:, 0]

    # Plot dataset points
    ax_3d.scatter(
        xs=data_x,
        ys=data_y,
        zs=data_z,  # type: ignore
        color="red",
        marker="o",
        s=100,
        label="Dataset points",
    )

    min = -0.5
    max = 1.5
    X = np.linspace(min, max, 100)
    Y = np.linspace(min, max, 100)

    X, Y = np.meshgrid(X, Y)

    grid_points = np.stack([X.flatten(), Y.flatten()], axis=1)

    Z = np.apply_along_axis(lambda point: get_nn_output(point), 1, grid_points)

    Z = Z.reshape(X.shape)

    ax_3d.plot_surface(X, Y, Z)

    # Draw the updates
    fig.canvas.draw()
    fig.canvas.flush_events()


def render_losses(losses: list[int]):
    """Update the existing plot with new data"""
    global fig, bx

    if fig is None or bx is None:
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
    global fig, ax_3d, bx

    plt.ioff()  # Turn off interactive mode
    if fig is not None:
        plt.pause(60)
        plt.close(fig)

    fig = None
    ax_3d = None
    bx = None
    print("Plot cleanup completed.")
