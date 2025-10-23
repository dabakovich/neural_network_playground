import matplotlib.pyplot as plt
import numpy as np

# Global variables to persist figure and axes
fig = None
ax = None
bx = None


def set_labels():
    # Set labels (these don't change)
    ax.set_xlabel("Height")
    ax.set_ylabel("Size")

    bx.set_xlabel("Height")
    bx.set_ylabel("Size")


def init_plot():
    """Initialize the plot for real-time updates"""
    global fig, ax, bx

    # Enable interactive mode
    plt.ion()

    # Create figure and axes once
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 2, 1)
    bx = fig.add_subplot(1, 2, 2)

    print("Plot initialized for real-time updates...")


def render_plot(
    data: list[tuple[int, int]],
    # X: list[float],
    # Y: list[float],
    losses: list[int],
):
    """Update the existing plot with new data"""
    global fig, ax, bx

    if fig is None or ax is None:
        print("Warning: Plot not initialized. Call init_plot() first.")
        return

    print("Updating plot...")

    # Clear the axes but keep the labels
    ax.clear()
    bx.clear()
    set_labels()

    # Add points
    points = np.array(data)
    ax.scatter(points[:, 0], points[:, 1], s=100, marker="o", label="Points")

    # min_x = min(points[:, 0])
    # max_x = max(points[:, 0])

    # Add line
    # X = np.linspace(min_x, max_x)
    # Y = X * w + b

    # ax.plot(X, Y, label="Neural network")
    ax.legend()

    # Render loss graph
    loss_points = np.array(losses)
    loss_X = range(0, len(losses))
    loss_Y = loss_points
    bx.plot(loss_X, loss_Y, label=f"Current loss: {losses[-1]}")
    bx.legend()

    # Draw the updates
    fig.canvas.draw()
    fig.canvas.flush_events()


def cleanup_plot():
    """Clean up the plot when done"""
    global fig, ax, bx

    plt.ioff()  # Turn off interactive mode
    if fig is not None:
        plt.pause(10)
        plt.close(fig)

    fig = None
    ax = None
    bx = None
    print("Plot cleanup completed.")
