import pandas as pd

from neural_network_v2.neural_network import NeuralNetwork

xor_dataset_path = "datasets/xor.csv"


xor_df = pd.read_csv(xor_dataset_path)

xor_df["Class_1"] = xor_df["y1"].map({0: 0, 1: 1})  # pyright: ignore[reportArgumentType]
xor_df["Class_2"] = xor_df["y1"].map({1: 0, 0: 1})  # pyright: ignore[reportArgumentType]

print(xor_df)


xor_x_list = xor_df[["x1", "x2"]].to_numpy()
xor_y_list = xor_df[["Class_1", "Class_2"]].to_numpy()

nn_two_layers_two_inputs_sigmoid = NeuralNetwork(
    [
        {"input_size": 2, "output_size": 2, "activation": "tanh"},
        {"input_size": 2, "output_size": 2, "activation": "softmax"},
    ],
    # [[[-1.2, 2.1, -0.6], [1.2, -0.2, 0.4]], [[-0.1, 0.2, 0.5]]],
    learning_rate=0.01,
    loss_name="log",
)

nn_two_layers_two_inputs_sigmoid.train(
    x_list=xor_x_list,
    y_list=xor_y_list,
    epochs=40000,
    batch_size=2,
    # stop_on_loss=0.01,
    stop_on_loss=0.05,
    # render_every=100,
    render_every=1000,
)
