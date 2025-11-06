import pandas as pd

from neural_network_v2.neural_network import NeuralNetwork

and_dataset_path = "datasets/and.csv"
not_dataset_path = "datasets/not.csv"
shoes_dataset_path = "datasets/shoes.csv"
xor_dataset_path = "datasets/xor.csv"

nn_one_layer = NeuralNetwork(
    [
        {"input_size": 1, "output_size": 1, "activation": "linear"},
    ],
    learning_rate=0.01,
)

nn_one_layer_sigmoid = NeuralNetwork(
    [
        {"input_size": 1, "output_size": 1, "activation": "sigmoid"},
    ],
    # [[[-3, 1.3]]],
    learning_rate=1,
    loss_name="log",
)

nn_one_layer_two_inputs_sigmoid = NeuralNetwork(
    [
        {"input_size": 2, "output_size": 1, "activation": "sigmoid"},
    ],
    # [[[-3, 1.3]]],
    learning_rate=1,
    loss_name="log",
)


nn_two_layers_two_inputs_sigmoid = NeuralNetwork(
    [
        {"input_size": 2, "output_size": 2, "activation": "tanh"},
        {"input_size": 2, "output_size": 1, "activation": "sigmoid"},
    ],
    # [[[-1.2, 2.1, -0.6], [1.2, -0.2, 0.4]], [[-0.1, 0.2, 0.5]]],
    learning_rate=0.01,
    loss_name="log",
)

nn_two_layers = NeuralNetwork(
    [
        {"input_size": 1, "output_size": 2, "activation": "linear"},
        {"input_size": 2, "output_size": 1, "activation": "linear"},
    ],
    learning_rate=0.001,
)

nn_one_layer_two_inputs = NeuralNetwork(
    [
        {"input_size": 2, "output_size": 1, "activation": "linear"},
    ],
    learning_rate=0.05,
)


# print(nn_two_layers_two_inputs_sigmoid.layers)
# nn_one_layer_sigmoid.train_batch(not_dataset, 500)

# Working example for AND
# nn_one_layer_two_inputs_sigmoid.train_batch(and_dataset, 10000, 10)
# nn_two_layers_two_inputs_sigmoid.train_batch(and_dataset, 500)


xor_df = pd.read_csv(xor_dataset_path)

xor_x_list = xor_df[["x1", "x2"]].to_numpy()
xor_y_list = xor_df[["y1"]].to_numpy()

"""
Most effective XOR:
- tanh activation in the hidden layer
-- (relu also work, but less effective; also relu breaks with log loss since it couldn't divide by 0)
- asymmetric random weights (0 to 0.5), zeros for bias
- fast learning rate (0.1, 0.2)
"""
# Working example for XOR
# nn_one_layer_two_inputs_sigmoid.train(
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

# Not stable example for XOR
# nn_two_layers_two_inputs_sigmoid.train(xor_dataset, 40000, 1000, method="batch")

# nn_one_layer.train_batch(shoes_dataset, 50)
# nn_two_layers.train_batch(shoes_dataset, 50)
# nn_one_layer_two_inputs.train(and_dataset, 50)

# test parabola dataset
nn_two_layer_one_input_parabola = NeuralNetwork(
    [
        {"input_size": 1, "output_size": 2, "activation": "linear"},
        {"input_size": 2, "output_size": 1, "activation": "sigmoid"},
    ],
    # [[[-3, 1.3]]],
    learning_rate=0.1,
    # loss_name="log",
    loss_name="mse",
)

# nn_two_layer_one_input_parabola.train_sgd(test_2d_parabola_dataset, 10000)
