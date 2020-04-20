from utility import *
from net import ResidualNet
from collections import deque
import argparse
from tqdm import tqdm, trange
from agent import Agent
import copy
import sys
import os
sys.path.insert(1, "../game_simulation")
from GameCLI import Game
from torch.multiprocessing import Pool, Process, set_start_method
from eval_policy import eval_policy
GAMMA_RATE = 0.6
CUT = 3
## step per iteration = Episode * max round number

def gen_args(agent, num_episode_per_thread, num_processes):
    a = []
    for i in range(num_processes):
        a.append((agent, num_episode_per_thread, i))
    return a

def policy_iteration(args):
    set_start_method('spawn')
    agent = Agent(args)
    eval_policy(agent, 500)
    replay_memory = deque(maxlen = args.MAX_MEMORY_SIZE)
    num_episode_per_thread = args.step_per_iteration // args.NUM_OF_PROCESSES // (GAMEOVER_ROUND - CUT)
    outer = tqdm(range(args.total_iterations), desc='Iteration', position=0)
    for i in outer:
        if i == 0:
            num_episode = args.observation_data // args.NUM_OF_PROCESSES // (GAMEOVER_ROUND - CUT)
            thread_args = gen_args(agent, num_episode, args.NUM_OF_PROCESSES)
        else:
            thread_args = gen_args(agent, num_episode_per_thread, args.NUM_OF_PROCESSES)
        with Pool(processes=args.NUM_OF_PROCESSES) as pool:
            for result in pool.imap(thread_thunk, thread_args):
                replay_memory.extend(result)
        train_data = list(replay_memory)
        if i == 0:
            message = "num of training data: {}".format(len(train_data))
            outer.write(message)
            agent.train(train_data, 2)
        else:
            message = "num of training data: {}".format(args.step_per_iteration)
            outer.write(message)
            agent.train(train_data[-args.step_per_iteration:])
        eval_policy(agent, 500)
        
        agent.update_epsilon()
        


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
        train_data.extend(run_episode(agent))
    return train_data


def run_episode(agent):
    train_data = []
    game = Game(show = False)
    while not game.termination():
        board = copy.deepcopy(game.gameboard.board)
        choice = agent.greedy_policy(board, game.gameboard.get_available_choices())
        _, reward = game.input_pos(choice[0], choice[1])
        train_data.append([board, reward, choice])
        
    ## correct the reward
    for i in reversed(range(len(train_data) - 1)):
        train_data[i][1] += GAMMA_RATE * train_data[i + 1][1]

    return train_data[0:-CUT]


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
    
    parser.add_argument('--learning_rate', type=float, required=True,
                        help='learning_rate')
    
    parser.add_argument('--init_epsilon', type=float, required=True,
                        help='init_epsilon')
    
    parser.add_argument('--final_epsilon', type=float, required=True,
                        help='final_epsilon')
    
    parser.add_argument('--total_epoch', type=int, required=True,
                        help='total epoch of trainig network')
    
    parser.add_argument('--batch_size', type=int, required=True,
                        help='batch size when training network')
    
    parser.add_argument('--observation_data', type=int, required=True,
                        help='num of observation data')
    
    parser.add_argument('--MAX_MEMORY_SIZE', type=int, required=True,
                        help='how large is your replay memory')

    parser.add_argument('--NUM_OF_PROCESSES', type=int, required=True,
                            help='number of threads to play the games')
    
    args = parser.parse_args()
    policy_iteration(args)
    
    
if __name__ == "__main__":
    main()
#     agent = Agent(None, debug="./network/pg_try1.pth")
#     tmp = run_episode(agent)
#     print(tmp)
#     print([reward for _, reward, _ in tmp])
    