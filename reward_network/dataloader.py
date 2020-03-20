import torch
import sys
import os
sys.path.insert(1, '../game_simulation')
from utility import extract_board_and_reward_board
from parameters import Parameter
import torch.utils.data as Data
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import pickle

NUM_OF_COLOR = Parameter.NUM_OF_COLOR
ROW_DIM = Parameter.ROW_DIM
COLUMN_DIM = Parameter.COLUMN_DIM

class dataloader:
    def __init__(self, path, batch_size, shuffle=True, split_ratio=0.15, seed=None):
        self.filename = path.split("/")[-1]
        self.has_cache = False
        cache_path = "./input/" + self.filename
        if os.path.exists(cache_path):
            self.has_cache = True
            
        if self.has_cache:
            fd = open(cache_path, "rb")
            tensor_data = pickle.load(fd)
        else:
            tensor_data = self.create_tensor_data(path)
            fd = open(cache_path, "wb")
            pickle.dump(tensor_data, fd)


#         ## debug:
#         tensor_data = self.create_tensor_data(path)
#         fd = open(cache_path, "wb")
#         pickle.dump(tensor_data, fd)

        features, labels = tensor_data
        
        self.tensor_data = Data.TensorDataset(features, labels)
        
        self.data_size = len(self.tensor_data)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.split_ratio = split_ratio
        self.num_workers = 4
        self.seed = seed
        
        
    def get_loader(self):
        train_sampler, test_sampler = self.split_data()
        
        
        train_loader = Data.DataLoader(
            dataset=self.tensor_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=train_sampler
            )
        
        test_loader = Data.DataLoader(
            dataset=self.tensor_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=test_sampler
            )
        
        return train_loader, test_loader

            

    def split_data(self):
        # Creating data indices for training and validation splits:
        indices = list(range(self.data_size))
        split = int(np.floor(self.split_ratio * self.data_size))
        if self.shuffle :
            if self.seed != None:
                np.random.seed(self.seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(val_indices)
        return train_sampler, test_sampler
    
    
    def create_tensor_data(self, path):
        ## dataset is (cur_board, reward_board)
        all_boards, all_reward_boards = np.array(extract_board_and_reward_board(path))

        
        onehot_features = list(map(self.to_one_hot, all_boards))
        flatten_labels = list(map(self.flatten_label, all_reward_boards))
        tensor_features = torch.tensor(onehot_features).type(torch.float)
        tensor_labels = torch.tensor(flatten_labels).type(torch.float)
        print(tensor_features.size())
        print(tensor_labels.size())
        return tensor_features, tensor_labels
        
    def flatten_label(self, onehot_board):
        return onehot_board.flatten()
        
    
    def to_one_hot(self, board):
 
        
        onehot = np.zeros((NUM_OF_COLOR, ROW_DIM, COLUMN_DIM))

        for row in range(ROW_DIM):
            for col in range(COLUMN_DIM):
                color = board[row, col]
                if color == -1:
                    continue
                onehot[color, row, col] = 1
                
        return onehot
    
    
if __name__ == "__main__":
    dl = dataloader("./output/data1", 32)
    
    train_loader, _ = dl.get_loader()
    for i, data in enumerate(train_loader):
        if i > 0:
            break
        feature, label = data
        print(feature)
        print(label)