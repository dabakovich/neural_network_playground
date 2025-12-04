import numpy as np

WRONG_SPOT_REWARD = -2
TIE_REWARD = 0
WINNER_REWARD = 1
LOOSER_REWARD = -1

# HYPERPARAMETERS
TOP_K = 2
EPOCHS_PER_LEARNING = 1
RL_ONE_MOVE_EPOCHS_PER_LEARNING = 1

# It's a whole tic tac toe dataset in one batch (maximum 5 moves could be done)
BATCH_SIZE = 10

# Reward list, graduated by number of move. Last moves are the most serious ones, especially at the beginning
# REWARDS_LIST = np.array([1, 0.5, 0.3, 0.25, 0.2])
# REWARDS_LIST = np.array([0.2, 0.25, 0.3, 0.5, 1])
# REWARDS_LIST = np.array([0.2, 0.25, 0.4, 0.6, 1])
REWARDS_LIST = np.array([0.3, 0.35, 0.4, 0.6, 1])
# REWARDS_LIST = np.array([0.4, 0.5, 0.6, 0.75, 1])

WRONG_SPOT_REWARD_LIST = np.array([0, 0, 0, 0, 1])

IS_END_GAME_ON_WRONG_SPOT = True


# End reasons
WRONG_SPOT = "wrong_spot"
TIE = "tie"
AGENT_1 = "agent_1"
AGENT_2 = "agent_2"
