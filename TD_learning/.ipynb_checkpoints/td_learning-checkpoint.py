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
from torch.multiprocessing import set_start_method


ACTION_SIZE = ROW_DIM * COLUMN_DIM
MAX_REPLAY_MEMORY_SIZE = 200000
TRAINING_DATA_SIZE = 100000
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
    epsilon_decay = (1.0 - args.final_epsilon) / args.total_iterations
    
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

    
    args = parser.parse_args()
    td_learning(args)


if __name__ == "__main__":
    main()