import numpy as np

WRONG_SPOT_REWARD = -2
TIE_REWARD = 0
WINNER_REWARD = 1
LOOSER_REWARD = -1

EPOCHS_PER_LEARNING = 5

REWARDS_LIST = np.array([1, 0.5, 0.3, 0.25, 0.2])


# End reasons
WRONG_SPOT = "wrong_spot"
TIE = "tie"
AGENT_1 = "agent_1"
AGENT_2 = "agent_2"
