import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import numpy as np
from tqdm import tqdm, trange
####
import sys
sys.path.insert(1, "../Resnet")
from resnet import ResidualNet
#####
#from dueling_net import ResidualNet
from dataloader import DataLoader
from utility import *
#from eval_game import eval_game
import os.path

class DQNAgent:
    def __init__(self, args, debug=False):
        self.net = ResidualNet(NUM_OF_COLOR, ROW_DIM * COLUMN_DIM, ROW_DIM * COLUMN_DIM, 128)
        self.target_net = ResidualNet(NUM_OF_COLOR, ROW_DIM * COLUMN_DIM, ROW_DIM * COLUMN_DIM, 128)
        if debug:
            self.net.load_state_dict(torch.load(debug))
        else:
            if args != None:
                self.args = args
                if args.input_network:
                    self.net.load_state_dict(torch.load(args.input_network))
                    self.target_net.load_state_dict(torch.load(args.input_network))
                    
                self.epsilon = args.initial_epsilon
                training_times = args.total_steps // args.train_freq
                self.epsilon_decay = (args.initial_epsilon - args.final_epsilon) / training_times
                self.optimizer = optim.Adam(self.net.parameters(), lr=self.args.q_lr)
        self.net.cuda()
        self.target_net.cuda()
        
        self.criterion = nn.MSELoss()
    
    def update_epsilon(self):
        self.epsilon -= self.epsilon_decay
    
    def update_target(self, soft_tau):
        for target_param, param in zip(self.target_net.parameters(), self.net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        
        
        
    def greedy_policy(self, board, possible_actions):
        if np.random.rand() <= self.epsilon:
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
    
    def calculate_loss(self, states, rewards, actions, next_states):
        with torch.no_grad():
            _, next_states_best_moves = torch.max(self.net(next_states), 1)
            next_states_q_valules = self.target_net(next_states)
            next_states_q_valules = next_states_q_valules.gather(1, next_states_best_moves.unsqueeze(-1))
            target_q_values = rewards.unsqueeze(-1) + self.args.gamma * next_states_q_valules
        
        predicted_q_values = self.net(states).gather(1, actions.unsqueeze(-1))
        loss = self.criterion(predicted_q_values, target_q_values)
        return loss
        
        
    def train(self, replay_memory):
        dataloader = DataLoader(self.args.batch_size, replay_memory)
        train_loader, test_loader = dataloader.get_loader()
        
        total_loss = 0
        pbar = tqdm(range(self.args.total_epoch), desc='Epoch', position=1)
        for epoch in pbar:
            train_iter = tqdm(train_loader, desc='Step', position=2)
            for step, data in enumerate(train_iter):
                states, rewards, actions, next_states = data
                states = states.cuda()
                rewards = rewards.cuda()
                actions = actions.cuda()
                next_states = next_states.cuda()
                
                loss = self.calculate_loss(states, rewards, actions, next_states)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
                
                total_loss += loss.item()
                
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
                states, rewards, actions, next_states = data
                states = states.cuda()
                rewards = rewards.cuda()
                actions = actions.cuda()
                next_states = next_states.cuda()
                
                loss = self.calculate_loss(states, rewards, actions, next_states)
                test_loss += loss.item()
                total += 1

        print("test loss: " + str(test_loss / total))
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='td learning.')
   
    
    parser.add_argument('--input_network', type=str,
                        help='path of the input network')
    
    parser.add_argument('--output_network', type=str, required=True,
                        help='path of the output network')
    
    parser.add_argument('--learning_rate', type=float, required=True,
                        help='learning_rate')
    
    parser.add_argument('--total_epoch', type=int, required=True,
                        help='total epoch of trainig network')
    
    parser.add_argument('--batch_size', type=int, required=True,
                        help='batch size when training network')
    
    parser.add_argument('--gamma', type=float, required=True,
                        help='how important for future reward')

    
    args = parser.parse_args()
    
    
    
    teacher_agent = DQNAgent(args)
    student_agent = DQNAgent(args)

    student_agent.train(None, teacher_agent, student_agent)
    eval_game(agent, 300)