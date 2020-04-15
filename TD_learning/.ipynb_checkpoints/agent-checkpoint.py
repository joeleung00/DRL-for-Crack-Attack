import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import argparse
import numpy as np
from tqdm import tqdm, trange
sys.path.insert(1, "../Resnet/")
from resnet import ResidualNet
from dataloader import DataLoader
from utility import *
import os.path

class DQNAgent:
    def __init__(self, args, debug=False):
        self.net = ResidualNet(NUM_OF_COLOR, ROW_DIM * COLUMN_DIM, ROW_DIM * COLUMN_DIM, 128)
        if debug:
            self.net.load_state_dict(torch.load(debug))
        else:
            if args.input_network:
                self.net.load_state_dict(torch.load(args.input_network))
        self.net.cuda()
        self.args = args
        
        
    def greedy_policy(self, board, possible_actions, epsilon):
        if np.random.rand() <= epsilon:
            ## pick random action
            choice = np.random.randint(len(possible_actions))
            return possible_actions[choice]
        else:
            return self.best_move(board, possible_actions)
        
    def inference(self, board):
        onehot_board = to_one_hot(board)
        onehot_board = torch.from_numpy(onehot_board).type(torch.float32)
        onehot_board = onehot_board.cuda()
        with torch.no_grad():
            Q_values = self.net(onehot_board.unsqueeze(0)) ## output is a qvalue tensor for all actionss(size of  72)
        return Q_values    
        
    def best_move(self, board, possible_actions=None):
        ## pick best action
        ## current state is not tensor
        Q_values = self.inference(board)
        if possible_actions != None:
            Q_values = self.mask_actions(Q_values, possible_actions)
        _, index = torch.max(Q_values[0], 0)
        return deflatten_action(index.item())
    
    def get_qvalue_by_action(self, board, action):
        Q_values = self.inference(board)
        action_index = flatten_action(action)
        return Q_values[0][action_index].item()
            
    def get_max_qvalue(self, board):
        Q_values = self.inference(board)
        max_q, _ = torch.max(Q_values[0], 0)
        return max_q.item()
    
    def mask_actions(self, Q_values, possible_actions, mask_value=-100):
        ### for all not possible actions , sub the q values as -100, so i will not choose these actions
        possible_actions_index = list(map(flatten_action, possible_actions))
        not_possible_actions_mask = [True] * len(Q_values[0]) ## init with all actions are not possible
        ## specify some actions are possible so set it as False
        for index in possible_actions_index:
            not_possible_actions_mask[index] = False
            
        Q_values[0][not_possible_actions_mask] = mask_value
        return Q_values
    
    def extract_q_values(self, outputs, actions_indexes):
        new_outputs = torch.zeros(len(actions_indexes)).cuda()
        for i in range(len(actions_indexes)):
            new_outputs[i] = outputs[i][actions_indexes[i]]
        return new_outputs
    
    def MSELoss(self, outputs, labels):
        return torch.pow((outputs - labels), 2).mean()
    
    def train(self, replay_memory, teacher_agent, student_agent):
        dataloader = DataLoader(self.args.batch_size, replay_memory, teacher_agent, student_agent, self.args.gamma)
        train_loader, test_loader = dataloader.get_loader()
            
        total_loss = 0
        optimizer = optim.SGD(self.net.parameters(), lr=self.args.learning_rate, momentum=0.7)
        count_per_look = len(train_loader) // 10
        
        pbar = tqdm(range(self.args.total_epoch), desc='Epoch', position=1)
        for epoch in pbar:
            train_iter = tqdm(train_loader, desc='Step', position=2)
            for step, data in enumerate(train_iter):
                ## features is a onehot tensor board
                ## labels is a int tensor
                ## actions is a int tensor for action index
                features, labels, actions = data
                features = features.cuda()
                labels = labels.cuda()
                actions = actions.cuda()
                # zero the parameter gradients
                optimizer.zero_grad()

                outputs = self.net(features)
                outputs = self.extract_q_values(outputs, actions)

                loss = self.MSELoss(outputs, labels)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()

#                 if step % count_per_look == (count_per_look - 1):
#                     print('[%d, %5d] loss: %.3f' % (epoch + 1, step + 1, total_loss / count_per_look))
#                     total_loss = 0.0
            message = '[epoch %d] loss: %.3f' % (epoch + 1, total_loss / len(train_loader))
            pbar.write(message)                                     
            total_loss = 0.0
                    
        torch.save(self.net.state_dict(), self.args.output_network)
        self.test(test_loader)

            
    def test(self, test_loader):
        print("start testing network")
        test_loss = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                features, labels, actions = data
                features = features.cuda()
                labels = labels.cuda()
                actions = actions.cuda()
                outputs = self.net(features)
                outputs = self.extract_q_values(outputs, actions)
                loss = self.MSELoss(outputs, labels)
                test_loss += loss.item()
                total += 1

        print("test loss: " + str(test_loss / total))