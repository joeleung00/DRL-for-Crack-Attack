import pickle
import sys
import numpy as np
sys.path.insert(1, '../game_simulation')
from parameters import Parameter

NUM_OF_COLOR = Parameter.NUM_OF_COLOR
ROW_DIM = Parameter.ROW_DIM
COLUMN_DIM = Parameter.COLUMN_DIM
GAMEOVER_ROUND = Parameter.GAMEOVER_ROUND
def flatten_action(action):
    return action[0] * COLUMN_DIM + action[1]

def deflatten_action(action):
    return action // COLUMN_DIM, action % COLUMN_DIM


def save_train_data(data, path):
    with open(path, "wb") as fd:
        pickle.dump(data, fd)


def load_train_data(path):
    with open(path, "rb") as fd:
        return pickle.load(fd)

def to_one_hot(board):
    if type(board) == list:
        board = np.array(board, dtype=np.int)
    onehot = np.zeros((NUM_OF_COLOR, ROW_DIM, COLUMN_DIM))
    for row in range(ROW_DIM):
        for col in range(COLUMN_DIM):
            color = board[row, col]
            if color == -1:
                continue
            onehot[color, row, col] = 1

    return onehot

def to_one_hot_batch(board_list):
    return list(map(to_one_hot, board_list))

def flatten_action_batch(action_list):
    return list(map(flatten_action, action_list))