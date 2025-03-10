import sys
sys.path.insert(1, '../game_simulation')
sys.path.insert(1, "../cnn")
from GameBoard import GameBoard
from GameCLI import Game

from policy_value_net import Net
import torch
import numpy as np

from parameters import Parameter


NUM_OF_COLOR = Parameter.NUM_OF_COLOR
ROW_DIM = Parameter.ROW_DIM
COLUMN_DIM = Parameter.COLUMN_DIM
ACTION_SIZE = ROW_DIM * COLUMN_DIM

network_path = "./network/"

net = Net()

def deflatten_action(action):
    return action // COLUMN_DIM, action % COLUMN_DIM

def load_net(number):
    net.load_state_dict(torch.load(network_path + "network" + number + ".pth"))

def pre_process_features(raw_board):

    onehot = np.zeros((NUM_OF_COLOR, ROW_DIM, COLUMN_DIM))
    for row in range(ROW_DIM):
        for col in range(COLUMN_DIM):
            color = raw_board[row][col]
            onehot[int(color), row, col] = 1

    return onehot

def net_index2action_index(index):
    offset = index // (COLUMN_DIM - 1)
    return index +  offset

def get_action(current_state):
    onehot_current_state = pre_process_features(current_state)
    onehot_current_state = torch.from_numpy(onehot_current_state).type(torch.float32)
    probi, _ = net(onehot_current_state.unsqueeze(0)) ## output is a qvalue tensor for all actionss(size of  72)
    probi = probi[0]
    #index = np.random.choice(range(len(probi)), p=probi.detach().numpy())
    value, index = torch.max(probi, 0)
    print(value)
    return net_index2action_index(index)

def play(mode, filename, number):
    load_net(mode)
    fd = open("./output/" + filename, 'w')
    for i in range(number):
        game = Game(show=False)
        state = game.gameboard.board
        game.gameboard.print_board()
        while not game.termination():
            choice = get_action(state)
            choice2d = deflatten_action(choice)
            state, reward = game.input_pos(choice2d[0], choice2d[1])
        fd.write(str(game.gameboard.score) + "\n")
    fd.close()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("enter your network number:")
        exit(0)
    mode = sys.argv[1]
    if not mode.isdigit():
        print("Undefined number!!")
        exit(0)

    filename = sys.argv[2]
    n = sys.argv[3]
    play(mode, filename, int(n))
