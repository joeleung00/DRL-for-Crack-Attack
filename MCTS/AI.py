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


    def ask_board(self):
        self.string2keyboard("b")
        line = self.wait_console("b")
        self.board = self.process_raw_board(line)

    def ask_round(self):
        self.string2keyboard("r")
        line = self.wait_console("b")
        self.round = int(line)

    def send_position(self, row, col):
        string = str(row) + "," + str(col) + "."
        self.string2keyboard(string)

if __name__ == "__main__":
    ai = AI()
    time.sleep(3)



    while True:
        ai.ask_board()
        gameboard = GameBoard(ai.board)
        num_available_choices = len(gameboard.get_available_choices())
        init_state = State(gameboard.board, 0, [], num_available_choices)
        root_node = Node(state=init_state)

        current_node = monte_carlo_tree_search(root_node)
        choice = current_node.state.get_choice()
        print("You have choosen : " + str(choice[0]) + " " + str(choice[1]))
        ai.send_position(choice[0], choice[1])
        time.sleep(0.1)
