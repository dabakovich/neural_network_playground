import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from neural_network_v2.neural_network import NeuralNetwork
from shared.helpers import split_dataset

pd.options.display.float_format = "{:.3f}".format

rice_dataset_path = "datasets/Rice_Cammeo_Osmancik.csv"

raw_rice_dataset = pd.read_csv(rice_dataset_path)


# sns.pairplot(data=rice_dataset, hue="Class")

rice_dataset = raw_rice_dataset.copy()
rice_dataset["Class_1"] = raw_rice_dataset["Class"].map({"Cammeo": 1, "Osmancik": 0})  # pyright: ignore[reportArgumentType]
rice_dataset["Class_2"] = raw_rice_dataset["Class"].map({"Cammeo": 0, "Osmancik": 1})  # pyright: ignore[reportArgumentType]

print(rice_dataset.head())


rice_x_labels = [
    "Area",
    # "Perimeter",
    "Major_Axis_Length",
    # "Minor_Axis_Length",
    "Eccentricity",
    "Convex_Area",
    # "Extent",
]
rice_y_labels = [
    "Class_1",
    "Class_2",
]


train_rice_dataset, validate_rise_dataset, test_rice_dataset = split_dataset(
    rice_dataset, 0.8, 0.2, 0
)

threshold = 0.5
dataset_mean = train_rice_dataset[rice_x_labels].mean()
dataset_std = train_rice_dataset[rice_x_labels].std()


print(len(train_rice_dataset))
print(len(validate_rise_dataset))
print(len(test_rice_dataset))


def get_normalized_x_y_datasets(
    dataset: pd.DataFrame, x_labels: list[str], y_labels: list[str]
):
    normalized_x_dataset: pd.DataFrame = (
        dataset[x_labels] - dataset_mean
    ) / dataset_std

    y_dataset = dataset[y_labels]

    return normalized_x_dataset, y_dataset


normalized_train_x_dataset, train_y_dataset = get_normalized_x_y_datasets(
    train_rice_dataset, rice_x_labels, rice_y_labels
)


train_x_list = normalized_train_x_dataset.to_numpy()
train_y_list = train_y_dataset.to_numpy()


def show_correlation():
    correlation = rice_dataset.corr("spearman")
    sns.heatmap(correlation, vmin=-1, center=0, vmax=1, annot=True)
    plt.show()


# show_correlation()


def render_losses(losses: list[float]):
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning history")
    plt.show()


nn = NeuralNetwork(
    [
        {"input_size": len(rice_x_labels), "output_size": 2, "activation": "softmax"},
    ],
    learning_rate=0.01,
    loss_name="log",
)


def train_nn():
    return nn.train(
        x_list=train_x_list,
        y_list=train_y_list,
        epochs=100,
        batch_size=100,
        # stop_on_loss=0.2,
        # render_every=10,
        # threshold=threshold,
    )


def validate_nn():
    normalized_validate_x_dataset, validate_y_dataset = get_normalized_x_y_datasets(
        validate_rise_dataset, rice_x_labels, rice_y_labels
    )

    x_list = normalized_validate_x_dataset.to_numpy()
    y_list = validate_y_dataset.to_numpy()

    loss = nn.calculate_loss(x_list, y_list)
    # acc = nn.calculate_accuracy(x_list, y_list, threshold)
    acc = 0

    print(f"Validate loss: {loss:.3f} -  acc: {acc:.3f}")


def test_nn():
    normalized_test_x_dataset, test_y_dataset = get_normalized_x_y_datasets(
        test_rice_dataset, rice_x_labels, rice_y_labels
    )

    x_list = normalized_test_x_dataset.to_numpy()
    y_list = test_y_dataset.to_numpy()

    loss = nn.calculate_loss(x_list, y_list)
    # acc = nn.calculate_accuracy(x_list, y_list, threshold)
    acc = 0

    print(f"Test loss: {loss:.3f} -  acc: {acc:.3f}")


losses = train_nn()

validate_nn()

# test_nn()

render_losses(losses)
