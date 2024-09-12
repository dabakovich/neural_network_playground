import matplotlib.pyplot as plt


def draw_learning_history(learning_history: list[float]):
    plt.plot(learning_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning history')
    plt.show()
