import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from neural_network_v2.neural_network import NeuralNetwork

pd.options.display.float_format = "{:.3f}".format

rice_dataset_path = "datasets/Rice_Cammeo_Osmancik.csv"

rice_dataset = pd.read_csv(rice_dataset_path)

print(rice_dataset.describe())


# sns.pairplot(data=rice_dataset, hue="Class")

rice_dataset_2 = rice_dataset.copy()
rice_dataset_2["Class"] = rice_dataset_2["Class"].map({"Cammeo": 1, "Osmancik": 0})  # pyright: ignore[reportArgumentType]

x_labels = [
    "Area",
    "Perimeter",
    "Major_Axis_Length",
    "Eccentricity",
]
y_labels = [
    "Class",
]

rice_dataset_mean = rice_dataset_2[x_labels].mean()
rice_dataset_std = rice_dataset_2[x_labels].std()

normalized_x_dataset = (rice_dataset_2[x_labels] - rice_dataset_mean) / rice_dataset_std

# rice_dataset_2[x_labels] = rice_dataset_2[x_labels] / rice_dataset[x_labels].mean()


print(normalized_x_dataset.head())
print(normalized_x_dataset.describe())


def show_correlation():
    correlation = rice_dataset_2.corr("spearman")
    sns.heatmap(correlation, vmin=-1, center=0, vmax=1, annot=True)
    plt.show()


# show_correlation()

nn = NeuralNetwork(
    [
        {"input_size": 4, "output_size": 1, "activation": "sigmoid"},
    ],
    learning_rate=0.001,
    loss_name="log",
)

x_list = normalized_x_dataset.to_numpy()
y_list = rice_dataset_2[y_labels].to_numpy()


def train_nn():
    nn.train(
        x_list=x_list,
        y_list=y_list,
        epochs=100,
        batch_size=100,
        # stop_on_loss=720,
        render_every=10,
        threshold=0.5,
    )


train_nn()
