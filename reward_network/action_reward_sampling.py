
import random
import sys
sys.path.insert(1, '../game_simulation')
from GameBoard import GameBoard
sys.path.insert(1, '../MCTS')
from MCTS import *
import argparse

ROUND_PER_ROW = 2

class Game:

    def __init__(self):
        self.gameboard = GameBoard()
        #self.gameboard.init_board()

    def wait_input(self):
        self.gameboard.print_board()
        while True:
            pos = input()
            x, y = pos.split(' ')
            self.gameboard.proceed_next_state(int(x), int(y))
            self.gameboard.print_board()

    def input_pos(self, x, y):
        self.gameboard.proceed_next_state(int(x), int(y))
        self.gameboard.get_new_row()
        #self.gameboard.print_board()
        


    def insert_row_to_board(self):
        self.gameboard.board.remove(self.gameboard.board[0])
        self.gameboard.board.append(self.gameboard.new_row)
        self.gameboard.cursor_pos[0] -= 1
        self.gameboard.height += 1


def write_to_file(f, board, reward_board):

    output_string = ""
    m = len(board)
    n = len(board[0])
    for i in range(m):
        for j in range(n):
            output_string += str(board[i][j]) + ","
    output_string += "\n"
    f.write(output_string)
    
    output_string = ""
    m = len(reward_board)
    n = len(reward_board[0])
    for i in range(m):
        for j in range(n):
            output_string += str(reward_board[i][j]) + ","
    output_string += "\n"
    f.write(output_string)

def get_reward_board(board):
    m = len(board)
    n = len(board[0])
    original_gameboard = GameBoard(board=board)
    reward_board = [[-1 for j in range(n)] for i in range(m)]
    avail_choices = original_gameboard.get_available_choices()
    for (x, y) in avail_choices:
        new_gameboard = GameBoard(board=board)
        reward = new_gameboard.proceed_next_state(x, y)
        reward_board[x][y] = reward
    return reward_board
    
def sampling(filepath, TARGET_COUNT = 10):
    f = open(filepath, "a")
    data_count = 0
    while data_count < TARGET_COUNT:
        game = Game()
        gameover = False
        while not gameover:
            # mcts_read_board(game.gameboard.board):
            num_available_choices = len(game.gameboard.get_available_choices())
            init_state = State(game.gameboard.board, 0, [], num_available_choices)
            root_node = Node(state=init_state)
            current_node = root_node
            
            for i in range(ROUND_PER_ROW):
                sampling_board = current_node.state.current_board
                reward_board = get_reward_board(sampling_board)
                write_to_file(f, sampling_board, reward_board)
                data_count += 1
                
                current_node = monte_carlo_tree_search(current_node)
                choice = current_node.state.get_choice()
            
                
                game.input_pos(choice[0], choice[1])

 
            game.insert_row_to_board()
            if(game.gameboard.board[0]!=game.gameboard.empty_row):
                gameover = True
    f.close()
    
    
def main():
    parser = argparse.ArgumentParser(description='Reward sampling.')
    parser.add_argument('--output_path', type=str,
                        help='output file of the sampling data')
    parser.add_argument('--num_sample', type=int,
                        help='total number of sample you want to collect')

    args = parser.parse_args()
    
    sampling(args.output_path, args.num_sample)
    


if __name__ == "__main__":
    main()
#     gb = GameBoard(filename="./init_board")
#     print(get_reward_board(gb.board))