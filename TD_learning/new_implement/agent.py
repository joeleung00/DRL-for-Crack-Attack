import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import numpy as np
from tqdm import tqdm, trange
from dueling_net import ResidualNet
from dataloader import DataLoader
import torch.utils.data as data
from utility import *
#from eval_game import eval_game
import os.path

class DQNAgent:
    def __init__(self, args, debug=False):
        self.net = ResidualNet(NUM_OF_COLOR, ROW_DIM * COLUMN_DIM, ROW_DIM * COLUMN_DIM, 128)
        if debug:
            if type(debug) == str:
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
            return self.best_move([board], possible_actions)[0]
        
    def inference(self, board_list):
        onehot_boards = np.array(list(map(to_one_hot, board_list)))
        onehot_boards = torch.from_numpy(onehot_boards).type(torch.float32)
        onehot_boards = onehot_boards.cuda()
        with torch.no_grad():
            if len(onehot_boards) > 50000:
                all_Q_values = torch.zeros((0, ROW_DIM * COLUMN_DIM)).cuda()
                batches = data.DataLoader(onehot_boards, batch_size=50000)
                print(onehot_boards.shape)
                for batch in batches:
                    Q_values = self.net(batch)
                    all_Q_values = torch.cat((all_Q_values, Q_values), 0)
                print(all_Q_values.size())
                return all_Q_values
            else:
                Q_values = self.net(onehot_boards) ## output is a qvalue tensor for all actionss(size of  72)
                return Q_values    
        
    def best_move(self, board_list, possible_actions=None):
        ## pick best action
        ## current state is not tensor
        Q_values = self.inference(board_list)
        if possible_actions != None:
            ### should be one board only
            Q_values = self.mask_actions(Q_values, possible_actions)
        _, index = torch.max(Q_values, 1)
        return list(map(deflatten_action, index.view(-1).tolist()))
    
    def get_qvalue_by_action(self, board_list, action_list):
        Q_values = self.inference(board_list)
        action_index_list = list(map(flatten_action, action_list))
        action_index_list = torch.tensor(action_index_list).view(len(action_index_list), -1)
        return Q_values.gather(1, action_index_list.cuda())
    
    def get_max_qvalue(self, board_list):
        Q_values = self.inference(board_list)
        max_q, _ = torch.max(Q_values, 1)
        return max_q
    
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
        return outputs.gather(1, actions_indexes.view(len(actions_indexes), -1)).view(-1)
    
        
    
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