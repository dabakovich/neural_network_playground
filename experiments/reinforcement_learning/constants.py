from enum import Enum, IntEnum, StrEnum

import numpy as np


class BoardStringValue(StrEnum):
    X = "x"
    O = "o"
    EMPTY = " "


class BoardValue(IntEnum):
    X = 1
    O = 2
    # O = -1
    EMPTY = 0


BoardValueMap = {
    BoardValue.X: BoardStringValue.X,
    BoardValue.O: BoardStringValue.O,
    BoardValue.EMPTY: BoardStringValue.EMPTY,
}


class Reward(float, Enum):
    NOTHING = 0

    # For enums we need to have unique values
    WRONG_SPOT = -3
    TIE = 0.1
    WIN = 1
    LOSE = -1


# HYPERPARAMETERS

TOP_K = 5

EPOCHS_PER_LEARNING = 5
RL_ONE_MOVE_EPOCHS_PER_LEARNING = 1

RL_ONE_MOVE_WRONG_SPOT_REWARD_SHIFT = -1

# It's a whole tic tac toe dataset in one batch (maximum 5 moves could be done)
BATCH_SIZE = 2

# Reward list, graduated by number of move. Last moves are the most serious ones, especially at the beginning
# REWARD_SHIFTS_LIST = np.array([1, 0.5, 0.3, 0.25, 0.2])
# REWARD_SHIFTS_LIST = np.array([0.2, 0.25, 0.3, 0.5, 1])
# REWARD_SHIFTS_LIST = np.array([0.2, 0.25, 0.4, 0.6, 1])
REWARD_SHIFTS_LIST = np.array([0.3, 0.35, 0.4, 0.6, 1])
# REWARD_SHIFTS_LIST = np.array([0.4, 0.5, 0.6, 0.75, 1])

WRONG_SPOT_REWARD_SHIFTS_LIST = np.array([0, 0, 0, 0, 1])

IS_END_GAME_ON_WRONG_SPOT = False
# IS_END_GAME_ON_WRONG_SPOT = True


class AgentName(StrEnum):
    AGENT_1 = "agent_1"
    AGENT_2 = "agent_2"


class GameResult(StrEnum):
    TIE = "tie"

    AGENT_1_WRONG_SPOT = "agent_1_wrong_spot"
    AGENT_2_WRONG_SPOT = "agent_2_wrong_spot"

    AGENT_1_WIN = "agent_1_win"
    AGENT_2_WIN = "agent_2_win"
