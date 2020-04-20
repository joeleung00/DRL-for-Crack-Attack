import sys
sys.path.insert(1, '../game_simulation')
import torch
import argparse
from tqdm import tqdm, trange
from GameCLI import Game
from agent import DQNAgent
from collections import deque
from eval_game import eval_game
import copy
import random
from utility import *
from torch.multiprocessing import set_start_method, Pool, Process


ACTION_SIZE = ROW_DIM * COLUMN_DIM
MAX_REPLAY_MEMORY_SIZE = 1000000
TRAINING_DATA_SIZE = 50000
observation_data = 0

def init_game():
    game = Game(show=False)
    return game

def td_learning(args):
    set_start_method('spawn')
    agent = DQNAgent(args)
    pre_agent = DQNAgent(args)
    teacher_agent = DQNAgent(args)
    replay_memory = deque(maxlen = MAX_REPLAY_MEMORY_SIZE)
    epsilon = args.initial_epsilon
    epsilon_decay = (args.initial_epsilon - args.final_epsilon) / args.total_iterations
    eval_game(agent, 300)
    outer = tqdm(range(args.total_iterations), desc='Iteration', position=0) 
    for epoch in outer:
        game = init_game()
        inner = tqdm(range(args.step_per_iteration), desc='Sampling', position=1)
        for step in inner:
            ##play and store batch
            board = copy.deepcopy(game.gameboard.board)
            choice = agent.greedy_policy(board, game.gameboard.get_available_choices(), epsilon)
            next_board, reward = game.input_pos(choice[0], choice[1])
            next_board = copy.deepcopy(next_board)
            replay_memory.append((board, choice, reward, next_board))
            if game.termination():
                game = init_game()
        
        if len(replay_memory) >= observation_data:
            pre_agent =  clone_agent(agent, pre_agent)
            data_size = TRAINING_DATA_SIZE if len(replay_memory) >= TRAINING_DATA_SIZE else len(replay_memory)
            agent.train(random.sample(replay_memory, data_size), teacher_agent, agent)
            if epoch > 0:
                teacher_agent = clone_agent(pre_agent, teacher_agent)
    
            eval_game(agent, 300)

            if epsilon > args.final_epsilon:
                epsilon -= epsilon_decay
        
        
def clone_agent(src_agent, target_agent):
    target_agent.net.load_state_dict(src_agent.net.state_dict())
    return target_agent



def gen_args(agent, num_episode_per_thread, num_processes, epsilon):
    a = []
    for i in range(num_processes):
        a.append((agent, num_episode_per_thread, i, epsilon))
    return a

def new_td_learning(args):
    set_start_method('spawn')
    agent = DQNAgent(args)
    pre_agent = DQNAgent(args)
    teacher_agent = DQNAgent(args)
    replay_memory = deque(maxlen = MAX_REPLAY_MEMORY_SIZE)
    epsilon = args.initial_epsilon
    epsilon_decay = (args.initial_epsilon - args.final_epsilon) / args.total_iterations
    eval_game(agent, 300)
    
    num_episode_per_thread = args.step_per_iteration // args.NUM_OF_PROCESSES // (GAMEOVER_ROUND)
    outer = tqdm(range(args.total_iterations), desc='Iteration', position=0)
    for epoch in outer:
        thread_args = gen_args(agent, num_episode_per_thread, args.NUM_OF_PROCESSES, epsilon)
        with Pool(processes=args.NUM_OF_PROCESSES) as pool:
            for result in pool.imap(thread_thunk, thread_args):
                replay_memory.extend(result)
        
        if len(replay_memory) >= observation_data:
            pre_agent =  clone_agent(agent, pre_agent)
            data_size = TRAINING_DATA_SIZE if len(replay_memory) >= TRAINING_DATA_SIZE else len(replay_memory)
            agent.train(random.sample(replay_memory, data_size), teacher_agent, agent)
            if epoch > 0:
                teacher_agent = clone_agent(pre_agent, teacher_agent)
    
            eval_game(agent, 500)

            if epsilon > args.final_epsilon:
                epsilon -= epsilon_decay
        

def thread_thunk(args):
    agent = args[0]
    num_episode_per_thread = args[1]
    process_id = args[2]
    epsilon = args[2]
    train_data = []
    if process_id == 0:
        inner = tqdm(range(num_episode_per_thread), desc='Episode', position=1)
    else:
        inner = range(num_episode_per_thread)
    for i in inner:
        train_data.extend(run_episode(agent, epsilon))
    return train_data


def run_episode(agent, epsilon):
    train_data = []
    game = Game(show = False)
    while not game.termination():
        board = copy.deepcopy(game.gameboard.board)
        choice = agent.greedy_policy(board, game.gameboard.get_available_choices(), epsilon)
        _, reward = game.input_pos(choice[0], choice[1])
        next_board = copy.deepcopy(game.gameboard.board) 
        train_data.append((board, choice, reward, next_board))

    return train_data


def main():
    parser = argparse.ArgumentParser(description='td learning.')
    parser.add_argument('--total_iterations', type=int, required=True,
                        help='total_iterations of playing + learning')
    
    parser.add_argument('--step_per_iteration', type=int, required=True,
                        help='step_per_iteration step is one move in game')
    
    parser.add_argument('--initial_epsilon', type=float, required=True,
                        help='minimum epsilon')
    
    parser.add_argument('--final_epsilon', type=float, required=True,
                        help='minimum epsilon')
    
    parser.add_argument('--input_network', type=str,
                        help='path of the input network')
    
    parser.add_argument('--output_network', type=str, required=True,
                        help='path of the output network')
    
    parser.add_argument('--learning_rate', type=float, required=True,
                        help='learning_rate')
    
    parser.add_argument('--total_epoch', type=int, required=True,
                        help='total epoch of trainig network')
    
    parser.add_argument('--batch_size', type=int, required=True,
                        help='batch size when training network')
    
    parser.add_argument('--gamma', type=float, required=True,
                        help='how important for future reward')
    
    parser.add_argument('--NUM_OF_PROCESSES', type=int, required=True,
                        help='num of process')
    
    

    
    args = parser.parse_args()
    td_learning(args)


if __name__ == "__main__":
    main()