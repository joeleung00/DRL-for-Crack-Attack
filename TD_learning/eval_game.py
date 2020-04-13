import sys
sys.path.insert(1, '../game_simulation')
from GameCLI import Game
from agent import DQNAgent
import copy
                
def play(agent, show=False):
    game = Game(show=show)
    state = game.gameboard.board
    #game.gameboard.print_board()
    while not game.termination():
        choice = agent.best_move(game.gameboard.board, game.gameboard.get_available_choices())
        game.input_pos(choice[0], choice[1])
    return game.gameboard.score

def get_memory(replay_memory, agent, n = 10):
    for _ in range(n):
        game = Game(show=False)
        state = game.gameboard.board
        #game.gameboard.print_board()
        while not game.termination():
            board = copy.deepcopy(game.gameboard.board)
            choice = agent.best_move(game.gameboard.board, game.gameboard.get_available_choices())
            next_board, reward = game.input_pos(choice[0], choice[1])
            next_board = copy.deepcopy(next_board)
            replay_memory.append((board, choice, reward, next_board))
    
def eval_game(agent, total_num_games):
    total_score = 0
    for i in range(total_num_games):
        score = play(agent)
        total_score += score
        
    print("The average score after {} game: {}".format(total_num_games, total_score/total_num_games))
    
    
if __name__ == "__main__":
    agent = DQNAgent(None, debug="./network/test.pth")
    play(agent, show=True)
    #eval_game(agent, 500)