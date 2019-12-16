import sys
sys.path.insert(1, '../game_simulation')
sys.path.insert(1, "../cnn")
from GameBoard import GameBoard
from GameCLI import Game

from cnn_regression import Net
from collections import deque
import random
import torch
import numpy as np
from cnn_regression import train
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import time
import pickle
from parameters import Parameter

train_data_path = "/Users/joeleung/Documents/CUHK/yr4_term1/csfyp/csfyp/TD_learning/train_data/"
filepath = "/Users/joeleung/Documents/CUHK/yr4_term1/csfyp/csfyp/TD_learning/initboard"
network_path = "/Users/joeleung/Documents/CUHK/yr4_term1/csfyp/csfyp/TD_learning/network/"

NUM_OF_COLOR = Parameter.NUM_OF_COLOR
ROW_DIM = Parameter.ROW_DIM
COLUMN_DIM = Parameter.COLUMN_DIM

ACTION_SIZE = ROW_DIM * COLUMN_DIM

epsilon = 1.0
final_epsilon = 0.1
epsilon_step_num = 100
epsilon_decay = (1.0 - final_epsilon) / epsilon_step_num
gamma_rate = 1
n_step = 5
batch_size = 100
observations_steps = 100
nEpisode = 10000
target_model_period = 100
save_model_period = 200
epoch_per_batch = 2
lambda_rate = 0.5

net = Net()
target_net = Net()
criterion = nn.SmoothL1Loss()
#criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.0005, momentum=0.7)

## action is a flat arrary
## (current_state, action, reward, next_state)
replay_memory = deque(maxlen = 100000)
## run for a while to get observations


## network has 72 output for each possible actions
## top left be the index 0, bottom right is ROW * COLUMN - 1
def flatten_action(action):
    return action[0] * COLUMN_DIM + action[1]

def deflatten_action(action):
    return action // COLUMN_DIM, action % COLUMN_DIM
## raw board to numpy arrary
def pre_process_features(raw_board):

    onehot = np.zeros((NUM_OF_COLOR + 1, ROW_DIM, COLUMN_DIM))
    for row in range(ROW_DIM):
        for col in range(COLUMN_DIM):
            color = raw_board[row][col]
            onehot[int(color), row, col] = 1

    return onehot

def get_batch_from_memory():
    ## min_batch are all python data type
    min_batch = random.sample(replay_memory, batch_size)

    ## they are batch states
    current_states = np.zeros((batch_size, ROW_DIM, COLUMN_DIM))
    next_states = np.zeros((batch_size, ROW_DIM, COLUMN_DIM))
    actions = []
    rewards = []
    for i, value in enumerate(min_batch):
        current_states[i] = value[0]
        actions.append(value[1])
        rewards.append(value[2])
        next_states[i] = value[3]

    ## return data are all ten
    return (current_states, actions, rewards, next_states)


## epsilon greedy policy
def greedy_policy(current_state, episode, possible_actions, net, best=False):
    if (np.random.rand() <= epsilon or episode < observations_steps) and not best:
        ## pick random action
        choice = random.randrange(len(possible_actions) - 1)
        return flatten_action(possible_actions[choice])
    else:
        ## pick best action
        ## current state is not tensor
        onehot_current_state = pre_process_features(current_state)
        onehot_current_state = torch.from_numpy(onehot_current_state).type(torch.float32)
        actions_mask = torch.from_numpy(np.ones((1, ACTION_SIZE))).type(torch.float32)
        Q_values = net(onehot_current_state.unsqueeze(0), actions_mask) ## output is a qvalue tensor for all actionss(size of  72)
        _, index = torch.max(Q_values[0], 0)
        return index.item()

## batch learning
def batch_learning():
    ## actions is a flatten_action
    current_state_batch , actions, rewards, next_state_batch = get_batch_from_memory()
    #print(list(map(pre_process_features, next_state_batch)))
    next_state_batch = torch.FloatTensor(list(map(pre_process_features, next_state_batch)))

    actions_mask = torch.from_numpy(np.ones((batch_size,ACTION_SIZE))).type(torch.float32)

    next_Q_values = target_net(next_state_batch, actions_mask) ## 2D array (batch size, 72)

    #print(next_Q_values)
    filtered_Q_values_target = torch.zeros((batch_size, ACTION_SIZE))

    for i, row in enumerate(next_Q_values):
        max, index = torch.max(row, 0)
        filtered_Q_values_target[i][index] = rewards[i] + gamma_rate * max



    one_hot_actions = torch.from_numpy(np.eye(ACTION_SIZE)[np.array(actions).reshape(-1)]).type(torch.float32)
    ## convert current_state to pytorch
    current_state_batch = torch.FloatTensor(list(map(pre_process_features, current_state_batch)))

    train(epoch_per_batch, current_state_batch, one_hot_actions, filtered_Q_values_target, net, criterion, optimizer)

def save_train_data(episode):
    fullpathname = train_data_path + "data" + str(episode)
    fd = open(fullpathname, 'wb')
    pickle.dump(replay_memory, fd)


def load_train_data(episode):
    global replay_memory
    fullpathname = train_data_path + "data" + str(episode)
    fd = open(fullpathname, 'rb')
    replay_memory = pickle.load(fd)

def td_learning(mode):
    global epsilon
    max_score = 0
    if mode != "new":
        start_episode = int(mode)
        net.load_state_dict(torch.load(network_path + "net" + mode + ".pth"))
        target_net.load_state_dict(torch.load(network_path + "net" + mode + ".pth"))
        load_train_data(mode)
        global epsilon
        epsilon = epsilon - epsilon_decay * start_episode
        print("load the network and train data " + mode)

    else:
        start_episode = 0
    for episode in range(start_episode, nEpisode):
        ## init state
        game = Game(filepath)
        round = 0
        while not game.termination():
            ## pick an action
            possible_actions = game.gameboard.get_available_choices()
            ## choice is a flatten action
            current_state = game.gameboard.board
            choice = greedy_policy(current_state, episode, possible_actions, net)
            choice2d = deflatten_action(choice)
            next_state, reward = game.input_pos(choice2d[0], choice2d[1])

            if epsilon > final_epsilon and episode > observations_steps:
                epsilon -= epsilon_decay

            replay_memory.append((current_state, choice, reward, next_state))

            if episode > observations_steps:
                if round == 0:
                    batch_learning()
                if episode % target_model_period == 0 and game.gameboard.round_index == 1:
                    target_net.load_state_dict(net.state_dict())
                    print("Update the target net")


            if game.gameboard.score > max_score and episode > observations_steps:
                if game.gameboard.round_index == 1 and episode == observations_steps + 1:
                    print("Finish observations")
                max_score = game.gameboard.score
                print("max score is %d in episode %d" % (max_score, episode))

            round = (round + 1) % (batch_size // 4)

        if episode % save_model_period == 0:
            print("save model in episode %d" % (episode))
            save_net(net, episode)
            save_train_data(episode)

def save_net(net, episode):
    net_name = "net" + str(episode) + ".pth"
    torch.save(net.state_dict(), network_path + net_name)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("enter your mode:")
        print("new or continue(number)")
        exit(0)
    mode = sys.argv[1]
    if mode != "new" and not mode.isdigit():
        print("Undefined mode!!")
        exit(0)
    td_learning(mode)
