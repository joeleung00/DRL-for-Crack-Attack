from parameters import Parameter
import numpy as np
NUM_OF_COLOR = Parameter.NUM_OF_COLOR
ROW_DIM = Parameter.ROW_DIM
COLUMN_DIM = Parameter.COLUMN_DIM


##############
## This program take a game board as a input,
## Then output a features vector
## input type: gameboard
## output type: a score for each color (n / sum of edge length)
##############

def to_one_hot(board):
    onehot = np.zeros((NUM_OF_COLOR, ROW_DIM, COLUMN_DIM))
    for row in range(ROW_DIM):
        for col in range(COLUMN_DIM):
            color = board[row, col]
            if color == -1:
                continue
            onehot[color, row, col] = 1
    return onehot

def find_verticle_ones(row, one_color_board):
    column_pos = []
    while True:
        flag = False
        for col in range(COLUMN_DIM):
            if one_color_board[row][col] == 1:
                flag = True
                column_pos.append(col)
                one_color_board[row][col] = 0
                row += 1
                break

        if not flag:
            break
        if row >= ROW_DIM:
            break

    return column_pos

def verticle_ones_score(column_pos):
    length = len(column_pos)
    if length < 3:
        return 0

    score = length
    distance = 0
    for i in range(length - 1):
        distance += abs(column_pos[i + 1]  - column_pos[i])

    return score / distance

def get_one_color_score(one_color_board):
    total_score = 0
    while True:
        row = 0
        marginal_score = 0
        while row < ROW_DIM:
            ## do something
            column_pos = find_verticle_ones(row, one_color_board)
            marginal_score += verticle_ones_score(column_pos)
            row += len(column_pos) + 1
        total_score += marginal_score
        if marginal_score == 0:
            break

    return total_score

def get_score_vector(board):
    ## cast the board to numpy type if it is a list
    if type(board) == list:
        board = np.array(board, dtype=np.int)
    onehot_board = to_one_hot(board)
    vector = np.zeros(NUM_OF_COLOR)
    for i in range(NUM_OF_COLOR):
        vector[i] = get_one_color_score(onehot_board[i])

    return vector


def get_one_score(board):
    return get_score_vector(board).sum()

if __name__ == "__main__":
    fd = open("./input_board", "r")
    board = []
    for line in fd:
        print(line, end="")
        tmp = line[0:-1].split()
        int_list = list(map(int, tmp))
        board.append(int_list)
    score = get_score_vector(board)
    print(score)
