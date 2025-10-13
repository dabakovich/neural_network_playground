import matplotlib.pyplot as plt
import numpy as np

# Global variables to persist figure and axes
fig = None
ax = None


def init_plot():
    """Initialize the plot for real-time updates"""
    global fig, ax

    # Enable interactive mode
    plt.ion()

    # Create figure and axes once
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)

    # Set labels (these don't change)
    ax.set_xlabel("Height")
    ax.set_ylabel("Size")

    print("Plot initialized for real-time updates...")


def render_plot(data: list[tuple[int, int]], w: int, b: int):
    """Update the existing plot with new data"""
    global fig, ax

    if fig is None or ax is None:
        print("Warning: Plot not initialized. Call init_plot() first.")
        return

    print("Updating plot...")

    # Clear the axes but keep the labels
    ax.clear()
    ax.set_xlabel("Height")
    ax.set_ylabel("Size")

    # Add points
    points = np.array(data)
    ax.scatter(points[:, 0], points[:, 1], s=100, marker="o", label="Points")

    min_x = min(points[:, 0])
    max_x = max(points[:, 0])

    # Add line
    X = np.linspace(min_x, max_x)
    Y = X * w + b

    ax.plot(X, Y, label=f"y = x * {w:.3f} + {b:.3f}")
    ax.legend()

    # Draw the updates
    fig.canvas.draw()
    fig.canvas.flush_events()


def cleanup_plot():
    """Clean up the plot when done"""
    global fig, ax

    plt.ioff()  # Turn off interactive mode
    if fig is not None:
        plt.close(fig)

    fig = None
    ax = None
    print("Plot cleanup completed.")
