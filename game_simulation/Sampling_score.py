from GameBoard import GameBoard
import random
import sys
sys.path.insert(1, '../MCTS')
from MCTS import *

ROUND_PER_ROW = 2
TARGET_COUNT = int(sys.argv[2])
filepath = "./output/data" + sys.argv[1]
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


def write_to_file(f, value):

    output_string = ""
    output_string += str(value) + "\n"
    f.write(output_string)



if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("enter name num_of_test_case")
        exit(0)
    if not sys.argv[2].isdigit():
        print("wrong number")
        exit(0)
    f = open(filepath, "w")
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
                current_node = monte_carlo_tree_search(current_node)
                choice = current_node.state.get_choice()
                game.input_pos(choice[0], choice[1])

            game.insert_row_to_board()
            if(game.gameboard.board[0]!=game.gameboard.empty_row):
                gameover = True
                sampling_value = game.gameboard.score
                write_to_file(f, sampling_value)
                data_count += 1
    f.close()
