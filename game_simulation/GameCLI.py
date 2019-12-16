from GameBoard import GameBoard
import random
class Game:

    def __init__(self, filename = None, show=True):
        self.gameboard = GameBoard(filename = filename)
        self.gameover_round = 50
        self.show = show
        #self.gameboard.init_board()

    def wait_input(self):
        self.gameboard.print_board()
        while True:
            pos = input()
            x, y = pos.split(' ')
            self.gameboard.proceed_next_state(int(x), int(y))
            self.gameboard.print_board()

    def input_pos(self, x, y):
        marginal_score = self.gameboard.proceed_next_state(int(x), int(y))
        if self.show:
            print("You have chosen: ", x, y)
            self.gameboard.print_board()
            print("score is %d" % (self.gameboard.score))
        return self.gameboard.board, marginal_score

    def test_game(self):
        self.gameboard.test_board()
        self.gameboard.print_board()
        print()
        pos = input()
        x, y = pos.split(' ')
        self.gameboard.proceed_next_state(int(x), int(y))
        self.gameboard.print_board()

    def test_elimination(self):
        self.gameboard.test_board()
        self.gameboard.print_board()
        tmp = self.gameboard.elimination()
        print("Number of elimination is " + str(tmp))
        self.gameboard.print_board()

    def test_marksub(self):
        self.gameboard.test_board()
        marked_board = [[0 for j in range(6)] for i in range(12)]
        #print_board(marked_board)
        self.gameboard.mark_sub(0, 1, 3, marked_board, True)
        self.gameboard.mark_sub(2, 1, 4, marked_board, True)
        self.gameboard.mark_sub(0, 4, 6, marked_board, False)
        self.gameboard.mark_sub(2, 4, 7, marked_board, False)
        print_board(marked_board)

    def random_player(self):
        available_choices = self.gameboard.get_available_choices()
        return random.choice(available_choices)

    def termination(self):
        if not self.gameboard.get_available_choices() or self.gameboard.round_index >= self.gameover_round:
            return True

        return False

if __name__ == "__main__":
    game = Game()
    game.wait_input()
    # for _ in range(20):
    #     choice = game.random_player()
    #     game.input_pos(choice[0], choice[1])
