import torch
import torch.utils.data as Data
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import pickle
from utility import *
from torch.multiprocessing import Pool, Process
import os.path
import ray
ray.init()
class DataLoader:
    def __init__(self, batch_size, replay_memory, shuffle=False, split_ratio=0.05, seed=None):
        self.raw_data = replay_memory
    
#         if os.path.isfile("./1Mdata"):
#             with open("./1Mdata", "rb") as fd:
#                 tensor_data = pickle.load(fd)
#         else:
#             tensor_data = self.create_tensor_data((replay_memory, teacher_agent, student_agent))
#             with open("./1Mdata", "wb") as fd:
#                 pickle.dump(tensor_data, fd)
#         for item in replay_memory:
#             print(item)
    
        tensor_data = self.create_tensor_data(replay_memory)
      
        states, actions, rewards, next_states, weights = tensor_data
        
        self.tensor_data = Data.TensorDataset(states, actions, rewards, next_states, weights)
        self.data_size = len(self.tensor_data)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.split_ratio = split_ratio
        self.num_workers = 4
        self.seed = seed
        
        
    def get_loader(self):
        
        train_loader = Data.DataLoader(
            dataset=self.tensor_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            )
        return train_loader

            
  
    
    def create_tensor_data(self, replay_memory):
        ## dataset is (state, action, reward, next_state)
        all_states, all_actions, all_rewards, all_next_states, all_weights = replay_memory
        #onehot_states = list(map(to_one_hot, all_states))
        all_actions_indexes = list(map(flatten_action, all_actions))
        #onehot_next_states = list(map(to_one_hot, all_next_states))
        
        features = [ray_to_one_hot.remote(state) for state in all_states]
        onehot_states = ray.get(features)
        features = [ray_to_one_hot.remote(state) for state in all_next_states]
        onehot_next_states = ray.get(features)
   
        
#         onehot_states = self.thread_fn(all_states, to_one_hot_batch, num_thread)
#         all_actions_indexes = self.thread_fn(all_actions, flatten_action_batch, num_thread)
        
        tensor_weights = torch.tensor(all_weights).type(torch.float)
        tensor_rewards = torch.tensor(all_rewards).type(torch.float)
        tensor_states = torch.tensor(onehot_states).type(torch.float)
        tensor_actions = torch.tensor(all_actions_indexes).type(torch.long)
        tensor_next_states = torch.tensor(onehot_next_states).type(torch.float)
       
        return tensor_states, tensor_actions, tensor_rewards, tensor_next_states, tensor_weights
@ray.remote
def ray_to_one_hot(board):
    if type(board) == list:
        board = np.array(board, dtype=np.int)
    onehot = np.zeros((NUM_OF_COLOR, ROW_DIM, COLUMN_DIM))
    for row in range(ROW_DIM):
        for col in range(COLUMN_DIM):
            color = board[row, col]
            if color == -1:
                continue
            onehot[color, row, col] = 1

    return onehot
    
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