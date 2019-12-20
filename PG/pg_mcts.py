import random
import sys
import math
from copy import copy, deepcopy
import sys
sys.path.insert(1, '../game_simulation')
sys.path.insert(1, '../cnn')
from GameBoard import GameBoard
from GameCLI import Game
from parameters import Parameter
from policy_value_net import Net
import pickle
from collections import deque
import numpy as np
import torch
import multiprocessing


network_path = "./network/"
train_data_path = './train_data/'

NUM_OF_COLOR = Parameter.NUM_OF_COLOR
ROW_DIM = Parameter.ROW_DIM
COLUMN_DIM = Parameter.COLUMN_DIM

C = 10
MAX_ROUND_NUMBER = Parameter.MAX_ROUND_NUMBER
TAU = 0.2 ## cannot be zero
MAX_ROLLOUT_ROUND_NUMBER = 3
GAMMA_RATE = 1
## train an episode per iteration
TRAIN_ITERATION = 10000
EPISODE_PER_ITERATION = 10
SAVE_MODEL_PERIOD = 1000
DATA_SIZE_PER_TRAIN = 100
NUM_OF_PROCESSES = Parameter.NUM_OF_PROCESSES


replay_memory = deque(maxlen = 50000)
net = Net()


class Node:
    def __init__(self, state, parent = None):
        self.parent = parent
        self.child = {}
        self.visited_times = 0
        ## Q-value is the expected reward of the next actions, not counting reward come from this state
        self.quality_value = 0.0
        self.state = state
        self.policy_probi = {}


    def is_all_expanded(self):
        return len(self.child) == self.state.num_available_choices

    def add_child(self, node, child_id):
        self.child[child_id] = node


class State:
    def __init__(self, board, round_index, cumulative_choices, num_available_choices = 0, action_reward = 0):
        self.current_board = deepcopy(board)
        self.current_round_index = round_index
        self.cumulative_choices = deepcopy(cumulative_choices)
        self.num_available_choices = num_available_choices
        self.action_reward = action_reward

    def is_terminal(self, rollout = False):
        ## Add one more case - check is there any possible move?
        if not rollout:
            max_round = MAX_ROUND_NUMBER
        else:
            max_round = MAX_ROLLOUT_ROUND_NUMBER

        if self.current_round_index == max_round:
            return True
        elif self.num_available_choices == 0:
            return True
        else:
            return False


    def compute_reward(self, simulation_board):
        return simulation_board.score


    def get_next_state_with_random_choice(self, simulation_board, exclude=None):
        ## AVAILABLE_CHOICES is a double integer tupple list
        available_choices = simulation_board.get_available_choices()
        random_choice = random.choice(available_choices)
        child_id = flatten_action(random_choice)

        if exclude != None:
            while flatten_action(random_choice) in exclude:
                random_choice = random.choice(available_choices)
                child_id = flatten_action(random_choice)

        ## going to create new state
        action_reward = simulation_board.proceed_next_state(random_choice[0], random_choice[1])
        available_choices = simulation_board.get_available_choices()
        next_state =  State(simulation_board.board, self.current_round_index + 1,
            self.cumulative_choices + [random_choice], len(available_choices), action_reward)

        return next_state, child_id

    def get_next_best_state(self, simulation_board):
        available_choices = simulation_board.get_available_choices()
        best_choice = available_choices[0]
        best_score = 0
        for choice in available_choices:
            tmp_board = GameBoard(simulation_board.board, simulation = True)
            score = tmp_board.proceed_next_state(choice[0], choice[1])
            if score > best_score:
                best_score = score
                best_choice = choice
        action_reward = simulation_board.proceed_next_state(best_choice[0], best_choice[1])
        available_choices = simulation_board.get_available_choices()
        next_state =  State(simulation_board.board, self.current_round_index + 1,
            self.cumulative_choices + [best_choice], len(available_choices), action_reward)

        return next_state

    def get_choice(self):
        return self.cumulative_choices[-1]


def net_index2action_index(index):
    offset = index // (COLUMN_DIM - 1)
    return index +  offset

def action_index2net_index(index):
    offset = index // COLUMN_DIM
    return index - offset

def pre_process_features(raw_board):

    onehot = np.zeros((NUM_OF_COLOR, ROW_DIM, COLUMN_DIM))
    for row in range(ROW_DIM):
        for col in range(COLUMN_DIM):
            color = raw_board[row][col]
            onehot[int(color), row, col] = 1

    return onehot


def flatten_action(action):
    return action[0] * COLUMN_DIM + action[1]

def deflatten_action(action):
    return action // COLUMN_DIM, action % COLUMN_DIM

def pre_process_features(raw_board):

    onehot = np.zeros((NUM_OF_COLOR, ROW_DIM, COLUMN_DIM))
    for row in range(ROW_DIM):
        for col in range(COLUMN_DIM):
            color = raw_board[row][col]
            onehot[int(color), row, col] = 1

    return onehot


def tree_policy(node):
    # Check if the current node is the leaf node
    while node.state.is_terminal() == False:

        if node.is_all_expanded():
            node, _ = best_child(node, True)
        else:
            # Return the new sub node
            sub_node = expand(node)
            #print(node.state.current_board)
            return sub_node

    # Return the leaf node
    return node



def default_policy(node):

    # Get the state of the game
    current_state = node.state

    ## pre_process_features:
    onehot_board = pre_process_features(current_state.current_board)
    onehot_current_state = torch.from_numpy(onehot_board).type(torch.float32)
    probi, v = net(onehot_current_state.unsqueeze(0))
    probi = probi[0]
    for i, value in enumerate(probi):
        action_index = net_index2action_index(i)
        node.policy_probi[action_index] = value.item()

    return v.item()



def expand(node):
    """
    输入一个节点，在该节点上拓展一个新的节点，使用random方法执行Action，返回新增的节点。注意，需要保证新增的节点与其他节点Action不同。
    """
    child_node_state_set = set()
    for key in node.child:
        child_node_state_set.add(key)

    simulation_board = GameBoard(node.state.current_board)

    new_state, child_id = node.state.get_next_state_with_random_choice(simulation_board, exclude=child_node_state_set)

    # Check until get the new state which has the different action from others
    # while new_state in tried_sub_node_states:
    #     new_state = node.state.get_next_state_with_random_choice()

    sub_node = Node(parent=node, state=new_state)
    node.add_child(sub_node, child_id)

    return sub_node


def best_child(node, is_exploration):
    """
    使用UCB算法，权衡exploration和exploitation后选择得分最高的子节点，注意如果是预测阶段直接选择当前Q值得分最高的。
    """

    # TODO: Use the min float value
    best_score = -sys.maxsize
    best_sub_node = None
    probi = np.zeros(len(node.child))
    child_list_index = np.zeros(len(node.child))
    sum = 0

    # Travel all sub nodes to find the best one
    for i, key in enumerate(node.child):
        sub_node = node.child[key]
        child_list_index[i] = key
        # Ignore exploration for sinference
        if is_exploration:
            # old: UCB = quality / times + C * sqrt(2 * ln(total_times) / times)
            # C = 1 / math.sqrt(2.0)
            # left = sub_node.quality_value / sub_node.visited_times
            # right = 2.0 * math.log(node.visited_times) / sub_node.visited_times
            # score = left + C * math.sqrt(right)
            ## new: a = argmax(a) quality + C * P(a|s) / (1 + N(a|s))

            left = sub_node.quality_value
            right = C * node.policy_probi[key] / (1 + sub_node.visited_times)
            score = left + right


            if score > best_score:
                best_sub_node = sub_node
                best_score = score
                #print(best_score)
                #print("left: " + str(left))
                #print("right: " + str(right))

        else:
            #score = sub_node.visited_times ** (1/TAU)
            probi[i] = sub_node.visited_times ** (1/TAU)
            sum += probi[i]

    # if not is_exploration:
    #     probi /= sum
    #     cu_sum = np.cumsum(probi)
    #     rand = random.random()
    #     for i, value in enumerate(cu_sum):
    #         if rand <= value:
    #             key = child_list_index[i]
    #             best_sub_node = node.child[key]
    #             best_i = i
    #             break

    # if not is_exploration:
    #     best_sub_node = get_best_child(node)
    if is_exploration:
        return best_sub_node, None
    else:

        probi /= sum
        best_i = np.random.choice(range(len(probi)), p=probi)
        key = child_list_index[best_i]
        best_sub_node = node.child[key]

        return best_sub_node, probi[best_i]


def backup(node, reward):

    node.quality_value = reward + node.state.action_reward
    node.visited_times = 1

    ## this reward in the respective of parent node
    reward = GAMMA_RATE * node.quality_value
    node = node.parent
    # Update util the root node
    while node != None:
        #  Update the visit times
        node.visited_times += 1
        # Update the quality value

        node.quality_value += (1/node.visited_times) * (reward + node.state.action_reward - node.quality_value)
        ## this reward in the respective of parent node
        reward = GAMMA_RATE * reward

        # Change the node to the parent node
        node = node.parent

# def backup(node, reward):
#
#     node.quality_value = reward + node.state.action_reward
#     node.visited_times = 1
#     reward = node.quality_value
#     node = node.parent
#
#     # Update util the root node
#     while node != None:
#         #  Update the visit times
#         node.visited_times += 1
#
#         # Update the quality value
#         new_quality = reward + node.state.action_reward
#         if new_quality > node.quality_value:
#             node.quality_value = new_quality
#
#         reward = GAMMA_RATE * node.quality_value
#         # Change the node to the parent node
#         node = node.parent


def monte_carlo_tree_search(node):

    computation_budget = 200

    # Run as much as possible under the computation budget
    for i in range(computation_budget):

        # 1. Find the best node to expand
        expand_node = tree_policy(node)
        # 2. Random run to add node and get reward
        reward = default_policy(expand_node)

        # 3. Update all passing nodes with reward

        backup(expand_node, reward)

    # N. Get the best next node
    best_next_node, pi = best_child(node, False)
    #print("my quality_value :" + str(node.quality_value))

    return best_next_node, pi

def get_best_child(node):
    best_quality_value = -sys.maxsize
    best_child = None
    for key in node.child:
        if node.child[key].quality_value > best_quality_value:
            best_quality_value = node.child[key].quality_value
            best_child = node.child[key]
    return best_child

def load_net(number):
    net.load_state_dict(torch.load(network_path + "net" + number + ".pth"))

def save_net(net, number):
    net_name = "net" + str(number) + ".pth"
    torch.save(net.state_dict(), network_path + net_name)

def save_train_data(train_data, number):
    fullpathname = train_data_path + "data" + str(number)
    fd = open(fullpathname, 'wb')
    pickle.dump(train_data, fd)


def load_train_data(number):
    global replay_memory
    fullpathname = train_data_path + "data" + str(number)
    fd = open(fullpathname, 'rb')
    return pickle.load(fd)

def get_batch_from_memory():
    ## min_batch are all python data type (state, pi, reward)
    train_data = random.sample(replay_memory, DATA_SIZE_PER_TRAIN)

    ## they are batch states
    states = np.zeros((DATA_SIZE_PER_TRAIN, ROW_DIM, COLUMN_DIM)).astype(int)
    actions = []
    pis = []
    rewards = []
    for i, value in enumerate(train_data):
        states[i] = value[0]
        actions.append(value[1])
        pis.append(value[2])
        rewards.append(value[3])


    ## return data are all ten
    return (states, actions, pis, rewards)

def init_first_node(gameboard):
    num_available_choices = len(gameboard.get_available_choices())
    init_state = State(gameboard.board, 0, [], num_available_choices)
    root_node = Node(state=init_state)
    ## init visited_times, quality_value, policy_probi

    onehot_board = pre_process_features(gameboard.board)
    onehot_current_state = torch.from_numpy(onehot_board).type(torch.float32)
    probi, v = net(onehot_current_state.unsqueeze(0))
    probi = probi[0]
    for i, value in enumerate(probi):
        action_index = net_index2action_index(i)
        root_node.policy_probi[action_index] = value.item()

    root_node.visited_times = 1
    root_node.quality_value = v.item()
    return root_node



def policy_iteration(start_iteration=0):
    ## list of [state, action, pi, reward]
    pool = multiprocessing.Pool(processes = NUM_OF_PROCESSES)
    for i in range(start_iteration, TRAIN_ITERATION):
        # for j in range(EPISODE_PER_ITERATION):
        #     train_data = run_episode()
        train_data = pool.map(thread_thunk, range(NUM_OF_PROCESSES))
        for value in train_data:
            replay_memory.extend(value)
        print(len(replay_memory))
        print("Finish " + str((i + 1) * EPISODE_PER_ITERATION) + " episode")
        states, actions, pis, rewards = get_batch_from_memory()
        net.train(states, actions, pis, rewards)
        if i % SAVE_MODEL_PERIOD and i != 0:
            save_net(net, i)
            save_train_data(replay_memory, i)


def thread_thunk(useless):
    train_data = []
    for i in range(EPISODE_PER_ITERATION // NUM_OF_PROCESSES):
        train_data.extend(run_episode())
    return train_data

def run_episode():
    train_data = []
    game = Game(show = False)

    current_node = init_first_node(game.gameboard)

    while not game.termination():

        current_node, pi = monte_carlo_tree_search(current_node)

        choice = current_node.state.get_choice()
        flat_choice = flatten_action(choice)
        net_index =  action_index2net_index(flat_choice)
        one_data = [game.gameboard.board, net_index, pi, None]

        state, reward = game.input_pos(choice[0], choice[1])
        one_data[3] = reward
        train_data.append(one_data)

    ## correct the reward
    for i in reversed(range(len(train_data) - 1)):
        train_data[i][3] = GAMMA_RATE * train_data[i + 1][3]

    return train_data


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("enter your mode:")
        print("new or continue(number)")
        exit(0)
    mode = sys.argv[1]
    if mode != "new" and not mode.isdigit():
        print("Undefined mode!!")
        exit(0)
    if mode == "new":
        policy_iteration()
    else:
        policy_iteration(int(mode))
