import sys
sys.path.insert(1, '../game_simulation')
from GameBoard import GameBoard
from GameCLI import Game

from reward_network import Net
import torch
import numpy as np

from parameters import Parameter


NUM_OF_COLOR = Parameter.NUM_OF_COLOR
ROW_DIM = Parameter.ROW_DIM
COLUMN_DIM = Parameter.COLUMN_DIM
ACTION_SIZE = ROW_DIM * COLUMN_DIM

network_path = "./network/n2.pth"

net = Net()

def deflatten_action(action):
    return action // COLUMN_DIM, action % COLUMN_DIM

def load_net(cuda=True):
    if cuda:
        net.load_state_dict(torch.load(network_path))
    else:
        net.load_state_dict(torch.load(network_path, map_location=torch.device('cpu'))) 

def pre_process_features(raw_board):

    onehot = np.zeros((NUM_OF_COLOR, ROW_DIM, COLUMN_DIM))
    for row in range(ROW_DIM):
        for col in range(COLUMN_DIM):
            color = raw_board[row][col]
            if color < 0 :
                continue
            onehot[int(color), row, col] = 1

    return onehot


def get_action(current_state):
    onehot_current_state = pre_process_features(current_state)
    onehot_current_state = torch.from_numpy(onehot_current_state).type(torch.float32)
    with torch.no_grad():
        Q_values = net(onehot_current_state.unsqueeze(0)) ## output is a qvalue tensor for all actionss(size of  72)
    value = Q_values[0]
    #index = np.random.choice(range(len(probi)), p=probi.detach().numpy())
    value, index = torch.max(value, 0)
    #print(value)
    return index

def play():
    
    game = Game(show=False)
    state = game.gameboard.board
    #game.gameboard.print_board()
    while not game.termination():
        choice = get_action(state)
        choice2d = deflatten_action(choice)
        ### check the action is available?
        state, reward = game.input_pos(choice2d[0], choice2d[1])
    return game.gameboard.score

def multiple_games():
    load_net(cuda=False)
    total_score = 0
    for i in range(100):
        score = play()
        total_score += score
        
    print(total_score/100)
    

if __name__ == "__main__":

    multiple_games()