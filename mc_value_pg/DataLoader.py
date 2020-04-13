import torch
import torch.utils.data as Data
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import pickle
from utility import *

class DataLoader:
    def __init__(self, batch_size, replay_memory, shuffle=True, split_ratio=0.05, seed=None, path=None):
        
        if path != None:
            with open(path, "rb") as fd:
                self.tensor_data = pickle.load(fd)
        else:
            self.raw_data = replay_memory
            tensor_data = self.create_tensor_data(replay_memory)
            features, rewards, actions = tensor_data
            self.tensor_data = Data.TensorDataset(features, rewards, actions)
            with open("tensor_50data_pg", "wb") as fd:
                pickle.dump(self.tensor_data, fd)
                
         
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
    
    
    def create_tensor_data(self, replay_memory):
        ## dataset is (state, action, reward, next_state)
        all_states, all_rewards, all_actions = [value for value in zip(*replay_memory)]
    
        onehot_states = list(map(to_one_hot, all_states))
        all_actions_indexes = list(map(flatten_action, all_actions))
        
        tensor_features = torch.tensor(onehot_states).type(torch.float)
        tensor_rewards = torch.tensor(all_rewards).type(torch.float)
        tensor_actions = torch.tensor(all_actions_indexes).type(torch.long)

        return tensor_features, tensor_rewards, tensor_actions
    
        
    
    
if __name__ == "__main__":
    pass
#     from eval_game import get_memory
#     from collections import deque
#     from agent import DQNAgent
#     replay_memory = deque(maxlen = 500)
#     teacher_agent = DQNAgent(None, debug="../reward_network/network/n4.pth")
#     get_memory(replay_memory, teacher_agent)
#     dl = dataloader(2,  replay_memory, teacher_agent, 1.0)
    
#     train_loader, _ = dl.get_loader()
#     for i, data in enumerate(train_loader):
#         if i > 0:
#             break
#         feature, label, action = data
#         print(feature)
#         print(label)
#         print(action)