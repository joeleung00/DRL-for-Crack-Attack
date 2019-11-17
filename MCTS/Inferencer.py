## get the network
## change image to tensor
## feed into network
## transform result as a score
import sys
sys.path.insert(1, '../cnn')
sys.path.insert(1, '../game_simulation')
PATH = "/Users/joeleung/Documents/CUHK/yr4_term1/csfyp/csfyp/cnn/network/network4.pth"
from cnn import Net
from GameBoard import GameBoard
import torch
import numpy as np
NUM_OF_COLOR = 6
ROW_DIM = 12
COLUMN_DIM = 6

class Inferencer:
    def __init__(self, path, small):
        self.net = Net()
        self.path = path
        self.small = small
        self.net.load_state_dict(torch.load(path))

    def change_format(self, board):
        np_onehot = self.to_one_hot(board)
        tensor_onthot = torch.from_numpy(np_onehot).type(torch.float32)
        return tensor_onthot


    def inference(self, board):
        with torch.no_grad():
            tensor = self.change_format(board)
            output = self.net(tensor)
            _, predicted = torch.max(output.data, 1)
        return self.transform(predicted.item())


    def transform(self, value):
        if (self.small):
            return value * 3
        else:
            return value * 30

    def to_one_hot(self, board):
        onehot = np.zeros((1, NUM_OF_COLOR + 1, ROW_DIM, COLUMN_DIM))
        for row in range(ROW_DIM):
            for col in range(COLUMN_DIM):
                color = board[row][col]
                onehot[0, color, row, col] = 1
        return onehot

if __name__ == "__main__":
    inferencer = Inferencer()
    gameboard = GameBoard()
    score = inferencer.inference(gameboard.board)
    print(score)
