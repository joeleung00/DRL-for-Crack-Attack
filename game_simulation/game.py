import random
from copy import copy, deepcopy
import math
EMPTY = -1
BLUE = 0
YELLOW = 1
GREEN = 2
ORANGE = 3
PURPLE = 4
MARKED = 12

def print_board(board):
    for i in range(4):
        for j in range(3):
            print(board[i][j], end = " ")
        print()


class Game:

    def __init__(self):
        self.gameboard = GameBoard()
        self.gameboard.init_board()

    def wait_input(self):
        self.gameboard.print_board()
        while True:
            pos = input()
            x, y = pos.split(' ')
            self.gameboard.proceed_next_state(int(x), int(y))
            self.gameboard.print_board()

    def input_pos(self, x, y):
        self.gameboard.proceed_next_state(int(x), int(y))
        self.gameboard.print_board()

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



class GameBoard:
    def __init__(self, board = None, simulation = False):
        self.score = 0.0
        # self.row_dim = 12
        # self.column_dim = 6
        # self.blocks_init_height = 8
        self.row_dim = 12
        self.column_dim = 6
        self.blocks_init_height = 8
        # self.num_of_color = 5
        self.num_of_color = 5
        if board == None:
            self.board = [[-1 for i in range(self.column_dim)] for j in range(self.row_dim)]
        else:
            self.board = deepcopy(board)
        self.round_index = 0
        self.simulation = simulation

    def print_board(self):
        for i in range(self.row_dim):
            for j in range(self.column_dim):
                print(self.board[i][j], end=" ")
            print()
        print("Your score is :" + str(self.score))

    def get_random_block(self, row, col):
        result = None
        if (row == self.row_dim - 1 and col == 0):
            result = random.randint(0,self.num_of_color - 1)
        elif (row == self.row_dim - 1):
            tmp = random.randint(0, self.num_of_color - 1)
            if (tmp == self.board[row][col - 1]):
                result = (tmp + 1) % self.num_of_color
            else:
                result = tmp
        elif (col == 0):
            tmp = random.randint(0, self.num_of_color - 1)
            if (tmp == self.board[row + 1][col]):
                result = (tmp + 1) % self.num_of_color
            else:
                result = tmp
        else:
            while True:
                tmp = random.randint(0, self.num_of_color - 1)
                if ((tmp != self.board[row + 1][col]) and (tmp != self.board[row][col - 1])):
                    result = tmp
                    break
        return result

    def init_board(self):
        for row in range(self.row_dim - 1, self.row_dim - self.blocks_init_height - 1, -1):
            for col in range(self.column_dim):
                self.board[row][col] = self.get_random_block(row, col)

    def swap(self, row, col):
        # Out of bound:
        if row < 0 or row >= self.row_dim or col < 0 or col >= (self.column_dim - 1):
            return False
        # Empty position:
        if self.board[row][col] == EMPTY and self.board[row][col + 1] == EMPTY:
            return False

        self.board[row][col + 1],  self.board[row][col] = self.board[row][col], self.board[row][col + 1]

        return True

    def gravity(self):
        for row in range(self.row_dim - 2, -1, -1):
            for col in range(self.column_dim):
                block = self.board[row][col]
                cur_row = row
                cur_col = col
                if block != EMPTY:
                    while cur_row < self.row_dim - 1 and self.board[cur_row + 1][cur_col] == EMPTY:
                         self.board[cur_row][cur_col] = EMPTY
                         self.board[cur_row + 1][cur_col] = block
                         cur_row += 1

    def test_board(self):
        #case 1
        # for i in range(1,4):
        #     self.board[0][i] = BLUE
        # # case 2
        # for i in range(1, 5):
        #     self.board[2][i] = BLUE
        # #case 3
        # for i in range(4, 7):
        #     self.board[i][0] = BLUE
        # #case 4
        # for i in range(4, 8):
        #     self.board[i][2] = BLUE
        # #case 5
        # for i in range(3, 6):
        #     self.board[9][i] = BLUE
        # for j in range(9, 12):
        #     self.board[j][4] = BLUE
        # self.board[8][4] = BLUE

        self.board[1][0] = 0
        self.board[2][0] = 0
        self.board[3][0] = 2
        self.board[1][2] = 2
        self.board[2][2] = 2
        self.board[3][2] = 0

    def mark_sub(self, i, start, end, marked_board, row_major):

        if row_major:
            if self.board[i][start] == EMPTY:
                return
            for j in range(start, end +  1):
                marked_board[i][j] = MARKED
        else:
            if self.board[start][i] == EMPTY:
                return
            for j in range(start, end + 1):
                marked_board[j][i] = MARKED

    def mark(self, marked_board):
        # row major scan:
        for row in range(self.row_dim):
            count = 1
            cur_color = self.board[row][0]
            for col in range(self.column_dim):
                if col == self.column_dim - 1:
                    if count >= 3:
                        self.mark_sub(row, col - count + 1, col, marked_board, True)
                else:
                    if self.board[row][col + 1] == cur_color:
                        count += 1
                    else:
                        if count >= 3:
                            self.mark_sub(row, col - count + 1, col, marked_board, True)
                        count = 1
                        cur_color = self.board[row][col + 1]

        # column major scan:
        for col in range(self.column_dim):
            count = 1
            cur_color = self.board[0][col]
            for row in range(self.row_dim):
                if row == self.row_dim - 1:
                    if count >= 3:
                        self.mark_sub(col, row - count + 1, row, marked_board, False)
                else:
                    if self.board[row + 1][col] == cur_color:
                        count += 1
                    else:
                        if count >= 3:
                            self.mark_sub(col, row - count + 1, row, marked_board, False)
                        count = 1
                        cur_color = self.board[row + 1][col]



    def elimination(self):
        count = 0
        marked_board = [[0 for i in range(self.column_dim)] for j in range(self.row_dim)]
        self.mark(marked_board)



        for row in range(self.row_dim):
            for col in range(self.column_dim):
                if marked_board[row][col] == MARKED:
                    self.board[row][col] = EMPTY
                    count += 1
        return count

    def proceed_next_state(self, row, col):
        tmp = self.swap(row, col)
        self.gravity()
        total_score_gain = 0
        while (True):
            score_gain = self.elimination()
            total_score_gain += score_gain
            self.gravity()
            if (score_gain == 0):
                break

        if self.simulation == True:
            self.score += total_score_gain * math.exp(-0.1 * self.round_index)
        else:
            self.score += total_score_gain
        self.round_index += 1
        return total_score_gain

    def get_available_choices(self):
        choices = []
        for row in range(self.row_dim):
            for col in range(self.column_dim - 1):
                if self.board[row][col] != EMPTY or self.board[row][col + 1] != EMPTY:
                    choices.append((row, col))
        return choices

    def reset_score(self):
        self.score = 0

if __name__ == "__main__":
    game = Game()
    game.wait_input()
    # for _ in range(5):
    #     choice = game.random_player()
    #     game.input_pos(choice[0], choice[1])
