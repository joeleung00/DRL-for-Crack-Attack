import torch
import torch.utils.data as Data
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import pickle
from utility import *
from torch.multiprocessing import Pool, Process
import os.path
class DataLoader:
    def __init__(self, batch_size, replay_memory, teacher_agent, student_agent, gamma, shuffle=True, split_ratio=0.05, seed=None):
        self.raw_data = replay_memory
        self.gamma = gamma
#         if os.path.isfile("./100Mdata"):
#             with open("./100Mdata", "rb") as fd:
#                 tensor_data = pickle.load(fd)
#         else:
#             tensor_data = self.thread_create_tensor_data(replay_memory, teacher_agent, student_agent, 10)
#             with open("./100Mdata", "wb") as fd:
#                 pickle.dump(tensor_data, fd)
                
        tensor_data = self.thread_create_tensor_data(replay_memory, teacher_agent, student_agent, 10)
        features, labels, actions = tensor_data    
        self.tensor_data = Data.TensorDataset(features, labels, actions)
    
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
    
    def gen_args(self, replay_memory, teacher_agent, student_agent, num_threads):
        a = []
        ave_workload = len(replay_memory) // num_threads + 1
        for i in range(num_threads):
            start = i * ave_workload
            end = min((i + 1) * ave_workload, len(replay_memory))
            a.append((replay_memory[start:end], teacher_agent, student_agent))
        return a
    
    def combine_result(self, tmp_result):
        tensor_features = tmp_result[0][0]
        tensor_labels = tmp_result[0][1]
        tensor_actions = tmp_result[0][2]
        for i in range(1, len(tmp_result)):
            tensor_features = torch.cat((tensor_features, tmp_result[i][0]), 0)
            tensor_labels = torch.cat((tensor_labels, tmp_result[i][1]), 0)
            tensor_actions = torch.cat((tensor_actions, tmp_result[i][2]), 0)
            
        return tensor_features, tensor_labels, tensor_actions
    
    def thread_create_tensor_data(self, replay_memory, teacher_agent, student_agent, num_threads):
        tmp_result = []
        thread_args = self.gen_args(replay_memory, teacher_agent, student_agent, num_threads)
        with Pool(processes=num_threads) as pool:
            for result in pool.imap(self.create_tensor_data, thread_args):
                tmp_result.append(result)
        return self.combine_result(tmp_result)
        
    
    def create_tensor_data(self, arg):
        replay_memory, teacher_agent, student_agent = arg
        ## dataset is (state, action, reward, next_state)
        all_states, all_actions, all_rewards, all_next_states = [value for value in zip(*replay_memory)]
    
        onehot_states = list(map(to_one_hot, all_states))
        all_actions_indexes = list(map(flatten_action, all_actions))
        
        tensor_labels = self.create_labels(all_rewards, all_next_states, teacher_agent, student_agent)
        tensor_features = torch.tensor(onehot_states).type(torch.float)
        tensor_actions = torch.tensor(all_actions_indexes).type(torch.int)

        return tensor_features, tensor_labels, tensor_actions
    
    def create_labels(self, all_rewards, all_next_states, teacher_agent, student_agent):
        all_actions = list(map(student_agent.best_move, all_next_states))
        all_max_qvalues = list(map(teacher_agent.get_qvalue_by_action, all_next_states, all_actions))
        all_labels = list(map(lambda a, b: self.gamma * a + b, all_max_qvalues, all_rewards))
        return torch.tensor(all_labels).type(torch.float)
        
    def flatten_label(self, onehot_board):
        return onehot_board.flatten()
        
    
    
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