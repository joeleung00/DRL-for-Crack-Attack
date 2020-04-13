from agent import Agent
import sys
sys.path.insert(1, "../game_simulation")
from GameCLI import Game
from newMCTS import MCTS
from utility import *
from torch.multiprocessing import Pool, Process, set_start_method

def eval_policy(agent, num):
    total_score = 0
    for i in range(num):
        score = policy_play(agent)
        total_score += score
    print("eval policy score:", total_score / num)
    print("score per swap:", total_score / num / GAMEOVER_ROUND)
    
def policy_play(agent):
    game = Game(show=False)
    while not game.termination():
        board = game.gameboard.board
        choice = agent.best_move(board)
        game.input_pos(choice[0], choice[1])
    return game.gameboard.score

def mcts_play(agent):
    game = Game(show=False)
    mcts = MCTS(game.gameboard, agent)
    while not game.termination():
        current_node = mcts.get_next_node()
        choice = current_node.state.choice
        game.input_pos(choice[0], choice[1])
    return game.gameboard.score

def thunk(args):
    agent = args[0]
    num_episode_per_process = args[1]
    total_score = 0
    for i in range(num_episode_per_process):
        score = mcts_play(agent)
        total_score +=  score
    return total_score


def eval_mc_value_pg(agent, num, num_processes):
    thread_args = [(agent, num // num_processes)] * num_processes
    total_score = 0
    with Pool(processes=num_processes) as pool:
        for result in pool.imap(thunk, thread_args):
            total_score += result
            
    print("eval mcts score:", total_score / num)
    print("score per swap:", total_score / num / GAMEOVER_ROUND)
    
if __name__ == "__main__":
    set_start_method('spawn')
    agent = Agent(None, debug="./network/train3.pth")
    eval_policy(agent, 300)
    eval_mc_value_pg(agent, 200, 5)