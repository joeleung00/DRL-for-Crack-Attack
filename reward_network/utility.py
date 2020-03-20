import numpy as np
import sys
sys.path.insert(1, '../game_simulation')
from parameters import Parameter

NUM_OF_COLOR = Parameter.NUM_OF_COLOR
ROW_DIM = Parameter.ROW_DIM
COLUMN_DIM = Parameter.COLUMN_DIM


def extract_board_and_reward_board(path):
    fd = open(path, "r")
    all_boards = []
    all_reward_boards = []
    while True:
        line1 = fd.readline()
        line2 = fd.readline()
        if line1 == "":
            break
        
        data_list = line1.split(",")
        del data_list[-1]
        board = data_list_to_board(data_list)
        all_boards.append(board)
        
        data_list = line2.split(",")
        del data_list[-1]
        reward_board = data_list_to_board(data_list)
        all_reward_boards.append(reward_board)
        
    return all_boards, all_reward_boards
        
        
def data_list_to_board(data_list):
    board = np.array(data_list, dtype=np.int).reshape(ROW_DIM, COLUMN_DIM)
    return board
    
if __name__ == "__main__":
    all_boards, all_reward_boards = extract_board_and_reward_board("./output/data1")
    print(all_boards[0:2])
    print(all_reward_boards[0:2])
    