from utility import *
from newMCTS import MCTS
from net import ResidualNet
from collections import deque
import argparse
from tqdm import tqdm, trange
from agent import Agent
import copy
import sys
import os
import pickle
sys.path.insert(1, "../game_simulation")
from GameCLI import Game
from torch.multiprocessing import Pool, Process, set_start_method


## step per iteration = Episode * max round number

def gen_args(agent, num_episode_per_thread, num_processes):
    a = []
    for i in range(num_processes):
        a.append((agent, num_episode_per_thread, i))
    return a

def policy_iteration(args):
    set_start_method('spawn')
    agent = Agent(args)
    replay_memory = deque(maxlen = args.MAX_MEMORY_SIZE)
    num_episode_per_thread = args.step_per_iteration // args.NUM_OF_PROCESSES // GAMEOVER_ROUND
    outer = tqdm(range(args.total_iterations), desc='Iteration', position=0)
    for i in outer:
        thread_args = gen_args(agent, num_episode_per_thread, args.NUM_OF_PROCESSES)
        with Pool(processes=args.NUM_OF_PROCESSES) as pool:
            for result in pool.imap(thread_thunk, thread_args):
                replay_memory.extend(result)
        message = "num of training data: {}".format(len(replay_memory))
        outer.write(message)
        
        
    with open(args.output_data, "wb") as fd:
        pickle.dump(replay_memory, fd)
        


def thread_thunk(args):
    agent = args[0]
    num_episode_per_thread = args[1]
    process_id = args[2]
    train_data = []
    if process_id == 0:
        inner = tqdm(range(num_episode_per_thread), desc='Episode', position=1)
    else:
        inner = range(num_episode_per_thread)
    for i in inner:
        train_data.append(run_episode(agent))
    return train_data


def run_episode(agent):
    train_data = []
    game = Game(show = False)

    mcts = MCTS(game.gameboard, agent)
    
    while not game.termination():
        board = copy.deepcopy(game.gameboard.board)
        current_node = mcts.get_next_node()
        if current_node == None:
            break
        choice = current_node.state.get_choice()
        _, reward = game.input_pos(choice[0], choice[1])
        train_data.append([board, reward, choice])

    return train_data


def main():
    parser = argparse.ArgumentParser(description='MCTS value PG.')
    parser.add_argument('--total_iterations', type=int, required=True,
                        help='total_iterations of playing + learning')
    
    parser.add_argument('--step_per_iteration', type=int, required=True,
                        help='step_per_iteration step is one move in game')
    
#     parser.add_argument('--max_round_per_episode', type=int, required=True,
#                         help='max_round_per_episode')

    
    parser.add_argument('--input_network', type=str,
                        help='path of the input network')
    
    parser.add_argument('--output_network', type=str, required=True,
                        help='path of the output network')
    
    parser.add_argument('--output_data', type=str, required=True,
                        help='path of the output data')
    
    parser.add_argument('--learning_rate', type=float, required=True,
                        help='learning_rate')
    
    parser.add_argument('--total_epoch', type=int, required=True,
                        help='total epoch of trainig network')
    
    parser.add_argument('--batch_size', type=int, required=True,
                        help='batch size when training network')
    
    parser.add_argument('--gamma', type=float, required=True,
                        help='how important for future reward')
    
    parser.add_argument('--MAX_MEMORY_SIZE', type=int, required=True,
                        help='how large is your replay memory')

    parser.add_argument('--NUM_OF_PROCESSES', type=int, required=True,
                            help='number of threads to play the games')
    
    args = parser.parse_args()
    policy_iteration(args)
    
    
if __name__ == "__main__":
    main()
#     agent = Agent(None, debug="./network/try5.pth")
#     tmp = run_episode(agent)
#     print([reward for _, reward, _ in tmp])
    