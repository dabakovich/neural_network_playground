# Neural Network

A from-scratch neural network implementation in Python using NumPy.

## Core Implementation (`neural_network_v2/`)

Custom neural network built entirely from scratch with:

- **Pure NumPy-based** implementation
- **Multiple activation functions**: tanh, sigmoid, softmax
- **Configurable layer architecture** with flexible network depth
- **Multiple loss functions**: MSE, log loss
- **Batch training** with mini-batch support
- **Built-in visualization tools** for training progress and decision boundaries

## Experiments

### Classic NN Training

- **[XOR Problem](experiments/xor.py)** - Demonstrates solving the classic non-linear XOR problem with a 2-layer network
- **[Rice Classification](experiments/rice.py)** - Binary classification of rice grain varieties (Cammeo vs Osmancik) using morphological features

### Reinforcement Learning

- **[Tic-Tac-Toe RL Agent](experiments/reinforcement_learning/)** - Neural network agent that learns to play Tic-Tac-Toe through self-play reinforcement learning

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run XOR experiment
python experiments/xor.py

# Run rice classification
python experiments/rice.py

# Train RL agent
python experiments/reinforcement_learning/tic_tac_toe.py
```
