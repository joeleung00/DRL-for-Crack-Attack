import subprocess
import time
import sys
from pynput.keyboard import Key, Controller
from MCTS import *
sys.path.insert(1, '../game_simulation')
from GameBoard import GameBoard
# from Game import Game
WAITING_TIME = 0.2

class AI:
    def __init__(self):
        self.game = subprocess.Popen(["python3", "-u", "../game_simulation/Game.py", "ai"], stdout=subprocess.PIPE, bufsize=1)
        self.round = 0
        self.board = None
        self.keyboard = Controller()
        self.previous_cur = [8, 2]

    def process_raw_board(self, line):
        board = []
        rows = line.split(".")
        del rows[-1]
        for row in rows:
            board_row = row.split(",")

            del board_row[-1]
            int_board_row = map(lambda x: int(x), board_row)
            board.append(list(int_board_row))
        return board

    def check_end_game(self):
        poll = self.game.poll()
        if poll != None:
            print("End Game")
            exit(0)

    def wait_console(self, resend):
        while True:
            print ("reading board")
            line = self.game.stdout.readline()
            if line == b"":
                time.sleep(WAITING_TIME)
                print("wait for output")
                self.string2keyboard(resend)
                self.check_end_game()
            else:
                break

        return line.decode("utf-8")

    def string2keyboard(self, string):
        for ch in string:
            self.keyboard.press(ch)
            self.keyboard.release(ch)
            time.sleep(0.1)



    def ask_board(self):
        self.string2keyboard("b")
        line = self.wait_console("b")
        self.board = self.process_raw_board(line)

    def ask_round(self):
        self.string2keyboard("r")
        line = self.wait_console("r")
        return int(line)

    def offset_choice(self, row, col):
        new_round = self.ask_round()
        diff = new_round - self.round
        return (row - diff, col)

    def ask_cursor(self):
        self.string2keyboard("c")
        line = self.wait_console("c")
        pos = line.split(" ")
        print(pos)
        self.previous_cur[0] = int(pos[0])
        self.previous_cur[1] = int(pos[1])

    def send_position(self, row, col):
        #string = str(row) + "," + str(col) + "."
        #self.string2keyboard(string)
        self.coord2keyboard(row, col)

    def coord2keyboard(self, row, col):
        row_dist = row - self.previous_cur[0]
        col_dist = col - self.previous_cur[1]
        ## if row_dist is negative: move up, otherwise down or no move
        ## if col_dist is negative: move left, otherwise right or no move

        if row_dist < 0:
            str = "w" * abs(row_dist)
            self.string2keyboard(str)
        elif row_dist > 0:
            str = "s" * row_dist
            self.string2keyboard(str)
        if col_dist < 0:
            str = "a" * abs(col_dist)
            self.string2keyboard(str)
        elif col_dist > 0:
            str = "d" * col_dist
            self.string2keyboard(str)


        self.string2keyboard(" ")
        self.previous_cur[0] = row
        self.previous_cur[1] = col


if __name__ == "__main__":
    ai = AI()
    time.sleep(5)

    count = 0

    while True:
        if count == 0:
            ai.ask_board()
            ai.round = ai.ask_round()
            gameboard = GameBoard(ai.board)
            num_available_choices = len(gameboard.get_available_choices())
            init_state = State(gameboard.board, 0, [], num_available_choices)
            root_node = Node(state=init_state)
            current_node = root_node

        current_node = monte_carlo_tree_search(current_node)
        choice = current_node.state.get_choice()
        print("You have choosen : " + str(choice[0]) + " " + str(choice[1]))
        choice = ai.offset_choice(choice[0], choice[1])
        ai.ask_cursor()

        ai.send_position(choice[0], choice[1])
        time.sleep(0.1)
        count = (count + 1) % 3