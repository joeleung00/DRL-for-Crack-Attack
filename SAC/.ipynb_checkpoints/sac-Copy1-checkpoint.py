import sys
sys.path.insert(1, '../game_simulation')
import torch
import argparse
from tqdm import tqdm, trange
from GameCLI import Game
from policy_agent import PolicyAgent
from collections import deque
from eval_game import eval_game
import copy
import random
from utility import *
from torch.multiprocessing import set_start_method, Pool, Process

ACTION_SIZE = ROW_DIM * COLUMN_DIM
MAX_REPLAY_MEMORY_SIZE = 2000000
TRAINING_DATA_SIZE = 500000

def init_game():
    game = Game(show=False)
    return game

def sac(args):
    set_start_method('spawn')
    agent = PolicyAgent(args)
    replay_memory = deque(maxlen = MAX_REPLAY_MEMORY_SIZE)
    eval_game(agent, 500)
    
    if args.load_observe_data:
        with open(args.load_observe_data, "rb") as fd:
            replay_memory = pickle.load(fd)
    elif args.observation_data != None:
        observe = tqdm(range(args.observation_data), desc='Observations', position=0)
        game = init_game()
        for i in observe:
            board = copy.deepcopy(game.gameboard.board)
            avail_choices = game.gameboard.get_available_choices()
            index = np.random.randint(len(avail_choices))
            choice = avail_choices[index]
            next_board, reward = game.input_pos(choice[0], choice[1])
            next_board = copy.deepcopy(next_board)
            replay_memory.append((board, choice, reward, next_board))
            if game.termination():
                game = init_game()
                
        if args.save_observation:
            with open(args.save_observation, "wb") as fd:
                pickle.dump(replay_memory, fd)
    if len(replay_memory) > 0:
        agent.train(replay_memory)
        eval_game(agent, 500)
    
    outer = tqdm(range(args.total_iterations), desc='Iteration', position=0)
    for epoch in outer:
        game = init_game()
        inner = tqdm(range(args.step_per_iteration), desc='Sampling', position=1)
        for step in inner:
            ##play and store batch
            board = copy.deepcopy(game.gameboard.board)
            choice = agent.get_action(board)
            next_board, reward = game.input_pos(choice[0], choice[1])
            next_board = copy.deepcopy(next_board)
            replay_memory.append((board, choice, reward, next_board))
            if game.termination():
                game = init_game()
        
       
        data_size = TRAINING_DATA_SIZE if len(replay_memory) >= TRAINING_DATA_SIZE else len(replay_memory)
        agent.train(random.sample(replay_memory, data_size))

        eval_game(agent, 500)




def main():
    parser = argparse.ArgumentParser(description='td learning.')
    parser.add_argument('--total_iterations', type=int, required=True,
                        help='total_iterations of playing + learning')
    
    parser.add_argument('--step_per_iteration', type=int, required=True,
                        help='step_per_iteration step is one move in game')
    
    
    parser.add_argument('--input_network', type=str,
                        help='path of the input network')
    
    parser.add_argument('--output_network', type=str, required=True,
                        help='path of the output network')
    
    parser.add_argument('--q_lr', type=float, required=True,
                        help='learning_rate')

    
    parser.add_argument('--policy_lr', type=float, required=True,
                        help='learning_rate')
    
    parser.add_argument('--total_epoch', type=int, required=True,
                        help='total epoch of trainig network')
    
    parser.add_argument('--batch_size', type=int, required=True,
                        help='batch size when training network')
    
    parser.add_argument('--gamma', type=float, required=True,
                        help='how important for future reward')
    
    parser.add_argument('--alpha', type=float, required=True,
                        help='how important of entropy')
    
    parser.add_argument('--soft_tau', type=float, required=True,
                        help='soft tau update network')
    
#     parser.add_argument('--NUM_OF_PROCESSES', type=int, required=True,
#                         help='num of process')
    
    parser.add_argument('--observation_data', type=int,
                        help='num of observation_data')
    
    parser.add_argument('--load_observe_data', type=str,
                        help='path of the input data')
    
    parser.add_argument('--save_observation', type=str,
                        help='path of the input data')

    
    args = parser.parse_args()
    sac(args)


if __name__ == "__main__":
    main()