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

# output_path = "./plot/td_small_board.txt"
# out_fd = open(output_path, "w")

def init_game():
    game = Game(show=False)
    return game


def td_learning(args):
    agent = DQNAgent(args)
    replay_memory = deque(maxlen = args.MAX_REPLAY_MEMORY_SIZE)
    #eval_game(agent, 500)
    outer = tqdm(range(args.total_steps), desc='Total steps', position=0)
    game = init_game()
    ave_score = 0
    count = 0
    for step in outer:
        board = copy.deepcopy(game.gameboard.board)
        if step < args.start_learn:
            avail_choices = game.gameboard.get_available_choices()
            index = np.random.randint(len(avail_choices))
            choice = avail_choices[index]
        else:
            choice = agent.greedy_policy(board, game.gameboard.get_available_choices())
            
        next_board, reward = game.input_pos(choice[0], choice[1])
        next_board = copy.deepcopy(next_board)
        replay_memory.append((board, choice, reward, next_board))
            
        if game.termination():
            ave_score += game.gameboard.score
            count += 1
            game = init_game()
            
        if step >= args.start_learn and step % args.train_freq == 0:
            if count > 0:
                message = "ave score of " + str(count) + " game: " + str(ave_score/count)
                #out_fd.write("{} {}\n".format(step, ave_score/count))
                outer.write(message)
                ave_score = 0
                count = 0
            if step == args.start_learn:
                if len(replay_memory) > 0:
                    agent.train(replay_memory)
            else:
                agent.train(random.sample(replay_memory, args.train_data_size))
            agent.update_target(args.soft_tau)
            agent.update_epsilon()
            
      
    eval_game(agent, 500)


def main():
    parser = argparse.ArgumentParser(description='td learning.')
    parser.add_argument('--total_steps', type=int, required=True,
                        help='total steps of playing')
    
    parser.add_argument('--train_data_size', type=int, required=True,
                        help='total steps of playing')
    
    parser.add_argument('--input_network', type=str,
                        help='path of the input network')
    
    parser.add_argument('--output_network', type=str, required=True,
                        help='path of the output network')
    
    parser.add_argument('--q_lr', type=float, required=True,
                        help='learning_rate')

    
    parser.add_argument('--initial_epsilon', type=float, required=True,
                        help='minimum epsilon')
    
    parser.add_argument('--final_epsilon', type=float, required=True,
                        help='minimum epsilon')
    
    parser.add_argument('--total_epoch', type=int, required=True,
                        help='total epoch of trainig network')
    
    parser.add_argument('--batch_size', type=int, required=True,
                        help='batch size when training network')
    
    parser.add_argument('--gamma', type=float, required=True,
                        help='how important for future reward')

    
    parser.add_argument('--soft_tau', type=float, required=True,
                        help='soft tau update network')
    
#     parser.add_argument('--NUM_OF_PROCESSES', type=int, required=True,
#                         help='num of process')
    
    parser.add_argument('--start_learn', type=int, required=True,
                        help='num of observation_data')
    
    parser.add_argument('--train_freq', type=int, required=True,
                        help='train_freq')
    
#     parser.add_argument('--update_target_freq', type=int, required=True,
#                         help='update_target_freq')
    
    parser.add_argument('--MAX_REPLAY_MEMORY_SIZE', type=int, required=True,
                        help='MAX_REPLAY_MEMORY_SIZE')
    


    
    args = parser.parse_args()
    td_learning(args)


if __name__ == "__main__":
    main()