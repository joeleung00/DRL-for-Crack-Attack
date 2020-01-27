import sys
sys.path.insert(1, '../game_simulation')
sys.path.insert(1, "../cnn")
from GameBoard import GameBoard
from GameCLI import Game

from parameters import Parameter


NUM_OF_COLOR = Parameter.NUM_OF_COLOR
ROW_DIM = Parameter.ROW_DIM
COLUMN_DIM = Parameter.COLUMN_DIM
ACTION_SIZE = ROW_DIM * COLUMN_DIM



def play(filename, number):
    fd = open("./output/" + filename, 'w')
    for i in range(number):
        game = Game(show=False)
        state = game.gameboard.board
        game.gameboard.print_board()
        while not game.termination():
            choice = game.random_player()
            game.input_pos(choice[0], choice[1])
        fd.write(str(game.gameboard.score) + "\n")
    fd.close()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("enter your filename and number:")
        exit(0)


    filename = sys.argv[1]
    n = sys.argv[2]
    play(filename, int(n))
