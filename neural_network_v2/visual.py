from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from neural_network_v2.types import Vector

# Global variables to persist figure and axes
fig = None
nn_output_axes_3d = None
weight_loss_axes_3d = None
loss_axes = None


def set_labels():
    global fig, nn_output_axes_3d, loss_axes

    if nn_output_axes_3d is None or weight_loss_axes_3d is None or loss_axes is None:
        raise ValueError("Init ax_3d and bx first")

    # Set labels (these don't change)
    nn_output_axes_3d.set_xlabel("X_1")
    nn_output_axes_3d.set_ylabel("X_2")
    nn_output_axes_3d.set_zlabel("Y")

    # Set labels (these don't change)
    weight_loss_axes_3d.set_xlabel("W_1")
    weight_loss_axes_3d.set_ylabel("W_2")
    weight_loss_axes_3d.set_zlabel("Loss")

    loss_axes.set_xlabel("Iterations")
    loss_axes.set_ylabel("Loss")


def init_plot():
    """Initialize the plot for real-time updates"""
    global fig, nn_output_axes_3d, weight_loss_axes_3d, loss_axes

    # Enable interactive mode
    plt.ion()

    # Create figure and axes once
    fig = plt.figure(figsize=(10, 8))
    nn_output_axes_3d = fig.add_subplot(2, 2, 1, projection="3d")
    loss_axes = fig.add_subplot(2, 2, 2)
    weight_loss_axes_3d = fig.add_subplot(2, 2, 3, projection="3d")

    print("Plot initialized for real-time updates...")


# Renders 3D plot of neural network output where x and y are inputs and z is output
def render_nn_output_for_two_inputs(
    x: np.ndarray,
    y: np.ndarray,
    get_nn_output: Callable[[Vector], Vector],
):
    global fig, nn_output_axes_3d

    if fig is None or nn_output_axes_3d is None:
        print("Warning: Plot not initialized. Call init_plot() first.")
        return

    nn_output_axes_3d.clear()
    set_labels()

    data_x = x[:, 0]
    data_y = x[:, 1]
    data_z = y[:, 0]

    # Plot dataset points
    nn_output_axes_3d.scatter(
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

    nn_output_axes_3d.plot_surface(X, Y, Z)

    # Draw the updates
    fig.canvas.draw()
    fig.canvas.flush_events()


def render_weight_loss_plot_3d(
    weight_history: list[tuple[float, float]],
    get_loss: Callable[[Vector], float],
):
    global fig, weight_loss_axes_3d

    if fig is None or weight_loss_axes_3d is None:
        print("Warning: Plot not initialized. Call init_plot() first.")
        return

    weight_loss_axes_3d.clear()
    set_labels()

    # first_point = weight_history[0]
    # print(first_point)
    # print([first_point[0], first_point[1], get_loss(np.array(first_point))])

    points = np.array(
        [[point[0], point[1], get_loss(np.array(point))] for point in weight_history]
    ).T

    # Plot dataset points
    weight_loss_axes_3d.plot(
        xs=points[0],
        ys=points[1],
        zs=points[2],  # type: ignore
        color="green",
        marker="x",
        # s=10,
        label="Weights point",
    )

    # Plot dataset points
    weight_loss_axes_3d.scatter(
        *points.T[-1],  # type: ignore
        color="red",
        marker="o",
        s=10,
        label="Weights point",
    )

    X = np.linspace(np.min(points[0]) - 1, np.max(points[0]) + 1, 100)
    Y = np.linspace(np.min(points[1]) - 1, np.max(points[1]) + 1, 100)

    X, Y = np.meshgrid(X, Y)

    grid_points = np.stack([X.flatten(), Y.flatten()], axis=1)

    Z = np.apply_along_axis(lambda point: get_loss(point), 1, grid_points)

    Z = Z.reshape(X.shape)

    weight_loss_axes_3d.plot_surface(X, Y, Z)

    fig.canvas.draw()
    fig.canvas.flush_events()


def render_losses(losses: list[int]):
    """Update the existing plot with new data"""
    global fig, loss_axes

    if fig is None or loss_axes is None:
        print("Warning: Plot not initialized. Call init_plot() first.")
        return

    loss_axes.clear()
    set_labels()

    # Render loss graph
    loss_points = np.array(losses)
    loss_X = range(0, len(losses))
    loss_Y = loss_points
    loss_axes.plot(loss_X, loss_Y, label=f"Current loss: {losses[-1]:.4f}")
    loss_axes.legend()

    # Draw the updates
    fig.canvas.draw()
    fig.canvas.flush_events()


def cleanup_plot():
    """Clean up the plot when done"""
    global fig, nn_output_axes_3d, loss_axes

    plt.ioff()  # Turn off interactive mode
    if fig is not None:
        plt.pause(60)
        plt.close(fig)

    fig = None
    nn_output_axes_3d = None
    loss_axes = None
    print("Plot cleanup completed.")
