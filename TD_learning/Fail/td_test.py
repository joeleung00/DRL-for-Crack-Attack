## create a game
## load the net
## from the game give the state to net
## predict the next move
## play the game
## get the next state from game


import sys
sys.path.insert(1, '../game_simulation')
sys.path.insert(1, "../cnn")
from GameBoard import GameBoard
from GameCLI import Game

from cnn_regression import Net
import torch
import numpy as np

from parameters import Parameter


NUM_OF_COLOR = Parameter.NUM_OF_COLOR
ROW_DIM = Parameter.ROW_DIM
COLUMN_DIM = Parameter.COLUMN_DIM
ACTION_SIZE = ROW_DIM * COLUMN_DIM

network_path = "/Users/joeleung/Documents/CUHK/yr4_term1/csfyp/csfyp/TD_learning/network/"
filepath = "/Users/joeleung/Documents/CUHK/yr4_term1/csfyp/csfyp/TD_learning/initboard"
net = Net()

def deflatten_action(action):
    return action // COLUMN_DIM, action % COLUMN_DIM

def load_net(number):
    net.load_state_dict(torch.load(network_path + "net" + number + ".pth"))

def pre_process_features(raw_board):

    onehot = np.zeros((NUM_OF_COLOR + 1, ROW_DIM, COLUMN_DIM))
    for row in range(ROW_DIM):
        for col in range(COLUMN_DIM):
            color = raw_board[row][col]
            onehot[int(color), row, col] = 1

    return onehot

def get_action(current_state):
    oneho_current_state = torch.from_numpy(onehot_current_state).type(torch.float32)
    actionst_current_state = pre_process_features(current_state)
    onehot_mask = torch.from_numpy(np.ones((1, ACTION_SIZE))).type(torch.float32)
    Q_values = net(onehot_current_state.unsqueeze(0), actions_mask) ## output is a qvalue tensor for all actionss(size of  72)
    value, index = torch.max(Q_values[0], 0)
    print(value)
    return index.item()

def play(mode):

    game = Game(filepath)
    load_net(mode)
    state = game.gameboard.board
    while not game.termination():
        choice = get_action(state)
        choice2d = deflatten_action(choice)
        state, reward = game.input_pos(choice2d[0], choice2d[1])

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("enter your network number:")
        exit(0)
    mode = sys.argv[1]
    if not mode.isdigit():
        print("Undefined number!!")
        exit(0)
    play(mode)
