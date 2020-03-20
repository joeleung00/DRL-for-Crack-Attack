import torch
import torch.utils.data as Data
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import sys
sys.path.insert(1, '../game_simulation')
from parameters import Parameter
import features_extraction as fe

NUM_OF_COLOR = Parameter.NUM_OF_COLOR
ROW_DIM = Parameter.ROW_DIM
COLUMN_DIM = Parameter.COLUMN_DIM

def raw_to_numpy(filepath):
    raw_input = np.loadtxt(filepath, delimiter=',')
    features = np.delete(raw_input, -1, axis=1)
    features = features.astype("int16")
    features = features.reshape(-1, ROW_DIM, COLUMN_DIM)
    labels = raw_input[:, -1]
    return features, labels



class DataLoader:
    def __init__(self, input_path, labels_type = "int64"):
        npfeatures, nplabels = raw_to_numpy(input_path)
        self.data_size = len(nplabels)
        self.batch_size = 32
        self.features = torch.from_numpy(self.to_one_hot(npfeatures)).type(torch.float32)
        new_features = np.array(list(map(self.preprocess, npfeatures)))
        new_features = torch.from_numpy(new_features).type(torch.float32).view(self.data_size, 3, ROW_DIM, COLUMN_DIM)

        self.features = torch.cat((self.features, new_features), axis=1)

        if labels_type == "int64":
            self.labels = torch.from_numpy(nplabels).type(torch.int64)
        elif labels_type == "float":
            self.labels = torch.from_numpy(nplabels).type(torch.float)
            

        self.torch_dataset = Data.TensorDataset(self.features, self.labels)
        self.num_workers = 2
        self.shuffle_dataset = True
        self.validation_split = .2
        self.train_sampler = None
        self.test_sampler = None
        self.split_data()


    def get_trainloader(self):

        train_loader = Data.DataLoader(
            dataset=self.torch_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.train_sampler
        )

        return train_loader


    def get_testloader(self):

        test_loader = Data.DataLoader(
            dataset=self.torch_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.test_sampler
        )

        return test_loader

    def to_one_hot(self, npfeatures):
        onehot = np.zeros((self.data_size, NUM_OF_COLOR + 1, ROW_DIM, COLUMN_DIM))
        for i in range(self.data_size):
            for row in range(ROW_DIM):
                for col in range(COLUMN_DIM):
                    color = npfeatures[i, row, col]
                    onehot[i, color, row, col] = 1
        return onehot

    def check_boundary(self, row, col):
        return row == 0 or row == (ROW_DIM - 1) or col == 0 or col == (COLUMN_DIM - 1)

    def count_boundary(self, new_feature, board, row, col):
        start_row = row - 1
        start_col = col - 1

        if start_row < 0:
            start_row = 0
        if start_col < 0:
            start_col = 0

        end_row = row + 1
        end_column = col + 1

        if end_row >= ROW_DIM:
            end_row = ROW_DIM - 1
        if end_column >= COLUMN_DIM:
            end_column = COLUMN_DIM - 1

        color = board[row][col]
        if color == -1:
            return 0
        count = 0
        for i in range(start_row, end_row + 1):
            for j in range(start_col, end_column + 1):
                if board[i][j] == color:
                    count += 1
        new_feature[row][col] = count

    def count_middle(self, new_feature, board, row, col):
        color = board[row][col]

        if color == -1:
            return 0

        start_row = row - 1
        start_col = col - 1
        end_row = row + 1
        end_column = col + 1
        count = 0
        for i in range(start_row, end_row + 1):
            for j in range(start_col, end_column + 1):
                if board[i][j] == color:
                    count += 1
        if board[row - 1][col] == -1:
            for i in range(0, row - 1):
                if board[i][col - 1] == color:
                    count += 1
                if board[i][col + 1] == color:
                    count += 1

        new_feature[row][col] = count

    def count_row_color(self, new_feature, board, row, col):
        color = board[row][col]
        if color == -1:
            return 0
        count = 0
        for i in range(COLUMN_DIM):
            if board[row][i] == color:
                count += 1

        new_feature[row][col] = count

    def count_col_color(self, new_feature, board, row, col):
        color = board[row][col]
        if color == -1:
            return 0
        count = 0
        for i in range(ROW_DIM):
            if board[i][col] == color:
                count += 1

        new_feature[row][col] = count

    def preprocess(self, board):
        new_features = np.zeros((3, ROW_DIM, COLUMN_DIM))
        for row in range(ROW_DIM):
            for col in range(COLUMN_DIM):

                if self.check_boundary(row, col):
                    self.count_boundary(new_features[0], board, row, col)
                else:
                    self.count_middle(new_features[0], board, row, col)

                self.count_row_color(new_features[1], board, row, col)
                self.count_col_color(new_features[2], board, row, col)

        return new_features


    def split_data(self):
        # Creating data indices for training and validation splits:
        indices = list(range(self.data_size))
        split = int(np.floor(self.validation_split * self.data_size))
        if self.shuffle_dataset :
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        # Creating PT data samplers and loaders:
        self.train_sampler = SubsetRandomSampler(train_indices)
        self.test_sampler = SubsetRandomSampler(val_indices)


if __name__ == "__main__":
    data_loader = DataLoader("./input/full_data_6color", labels_type="float")

    test_loader = data_loader.get_testloader()
    train_loader = data_loader.get_trainloader()
