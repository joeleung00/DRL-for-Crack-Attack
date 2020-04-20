from agent import Agent
import sys
sys.path.insert(1, "../game_simulation")
from GameCLI import Game
from utility import *

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


    
if __name__ == "__main__":
    agent = Agent(None, debug="./network/test.pth")
    eval_policy(agent, 100)
    #policy_play(agent)