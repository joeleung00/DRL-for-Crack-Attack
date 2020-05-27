import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import numpy as np
from tqdm import tqdm, trange
from policy_net import ResidualNet as PolicyNet
from value_net import ResidualNet as ValueNet
from dueling_net import ResidualNet as QValueNet
from dataloader import DataLoader
import torch.utils.data as Data
from utility import *
#from eval_game import eval_game
import os.path
from torch.distributions.categorical import Categorical
class PolicyAgent:
    def __init__(self, args, debug=False):
        self.policy_net = PolicyNet(NUM_OF_COLOR, ROW_DIM * COLUMN_DIM, ROW_DIM * COLUMN_DIM, 128)
        self.value_net = ValueNet(NUM_OF_COLOR, ROW_DIM * COLUMN_DIM, ROW_DIM * COLUMN_DIM, 128)
        self.q_value_net1 = QValueNet(NUM_OF_COLOR, ROW_DIM * COLUMN_DIM, ROW_DIM * COLUMN_DIM, 128)
        self.q_value_net2 = QValueNet(NUM_OF_COLOR, ROW_DIM * COLUMN_DIM, ROW_DIM * COLUMN_DIM, 128)
        self.target_value_net = ValueNet(NUM_OF_COLOR, ROW_DIM * COLUMN_DIM, ROW_DIM * COLUMN_DIM, 128)
        
        if debug:
            if type(debug) == str:
                self.load_net(debug)
        else:
            if args.input_network:
                self.load_net(args.input_network)
                
                
            self.soft_q_optimizer1 = optim.Adam(self.q_value_net1.parameters(), lr=args.q_lr)
            self.soft_q_optimizer2 = optim.Adam(self.q_value_net2.parameters(), lr=args.q_lr)
            self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=args.value_lr)
            self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=args.policy_lr)
            self.q_criterion1 = nn.MSELoss()
            self.q_criterion2 = nn.MSELoss()
            self.value_criterion = nn.MSELoss()
                
        self.to_cuda()
        self.args = args
    
    def to_cuda(self):
        self.policy_net.cuda()
        self.value_net.cuda()
        self.q_value_net1.cuda()
        self.q_value_net2.cuda()
        self.target_value_net.cuda()
        
        
    def load_net(self, prefix):
        policy_net_path = prefix + "/policy_net.pth"
        value_net_path = prefix + "/value_net.pth"
        q_value_net_path1 = prefix + "/q_value_net1.path"
        q_value_net_path2 = prefix + "/q_value_net2.path"
        
        self.policy_net.load_state_dict(torch.load(policy_net_path))
        self.value_net.load_state_dict(torch.load(value_net_path))
        self.q_value_net1.load_state_dict(torch.load(q_value_net_path1))
        self.q_value_net2.load_state_dict(torch.load(q_value_net_path2))
        self.target_value_net.load_state_dict(torch.load(value_net_path))
        
        
    def get_action(self, board):
        action = self.get_actions([board])[0]
        return deflatten_action(action.item())
    
    
    def get_actions(self, board_list):
        prob = self.policy_inference(board_list)
        print(prob)
        actions = torch.zeros(len(prob))
        for i in range(len(actions)):
            m = Categorical(prob[i])
            actions[i] = m.sample()
        return actions
    
    def evaluate(self, states):
        prob = self.policy_net(states)
        actions = torch.zeros(len(prob), dtype=torch.int).cuda()
        log_probs = torch.zeros(len(prob)).cuda()
        for i in range(len(actions)):
            m = Categorical(prob[i])
            actions[i] = m.sample()
            log_probs[i] = m.log_prob(actions[i])
        return actions, log_probs
 
    def update_target(self, soft_tau):
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

    def policy_inference(self, board_list):
        onehot_boards = np.array(list(map(to_one_hot, board_list)))
        onehot_boards = torch.from_numpy(onehot_boards).type(torch.float32)
        onehot_boards = onehot_boards.cuda()
        with torch.no_grad():
            prob = self.policy_net(onehot_boards) ## output is a qvalue tensor for all actionss(size of  72)
        return prob
    
        
    def best_move(self, board_list):
        prob = self.policy_inference(board_list)
        _, index = torch.max(prob, 1)
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

    
#     def extract_q_values(self, outputs, actions_indexes):
#         return outputs.gather(1, actions_indexes.view(len(actions_indexes), -1)).view(-1)

    def extract_q_values(self, outputs, actions_indexes):
        new_outputs = torch.zeros(len(actions_indexes)).cuda()
        for i in range(len(actions_indexes)):
            new_outputs[i] = outputs[i][actions_indexes[i]]
        return new_outputs
    
    def update_q_net(self, rewards, next_states, predicted_q_value1, predicted_q_value2, train=True):
        target_value = self.target_value_net(next_states)
        target_q_value = rewards + self.args.gamma * target_value.view(-1)
        loss1 = self.q_criterion1(predicted_q_value1, target_q_value.detach())
        loss2 = self.q_criterion2(predicted_q_value2, target_q_value.detach())

        
        if train:
            self.soft_q_optimizer1.zero_grad()
            loss1.backward()
            self.soft_q_optimizer1.step()

            self.soft_q_optimizer2.zero_grad()
            loss2.backward()
            self.soft_q_optimizer2.step()
            
        return loss1, loss2
    
    def update_value_net(self, new_actions, states, log_probs, predicted_value, train=True):
        predicted_new_q_value1 = self.q_value_net1(states)
        predicted_new_q_value1 = self.extract_q_values(predicted_new_q_value1, new_actions)
        
        predicted_new_q_value2 = self.q_value_net2(states)
        predicted_new_q_value2 = self.extract_q_values(predicted_new_q_value2, new_actions)
        
        predicted_new_q_value = torch.min(predicted_new_q_value1, predicted_new_q_value2)
        target_value_func = predicted_new_q_value - log_probs
        value_loss = self.value_criterion(predicted_value.view(-1), target_value_func.detach())


        if train:
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
        
        return predicted_new_q_value, value_loss
    
    def update_policy_net(self, log_probs, predicted_new_q_value, train=True):
        policy_loss = (log_probs - predicted_new_q_value).mean()

        if train:
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
        
        return policy_loss
    
    def predict(self, states, rewards, actions, next_states):
        predicted_q_value1 = self.q_value_net1(states)
        predicted_q_value2 = self.q_value_net2(states)
        predicted_q_value1 = self.extract_q_values(predicted_q_value1, actions)
        predicted_q_value2 = self.extract_q_values(predicted_q_value2, actions)
        predicted_value    = self.value_net(states)
        new_actions, log_probs = self.evaluate(states)
        log_probs = self.args.alpha * log_probs
        
        return predicted_q_value1, predicted_q_value2, predicted_value, new_actions, log_probs
    
    def train(self, replay_memory):
        
        dataloader = DataLoader(self.args.batch_size, replay_memory)
        train_loader, test_loader = dataloader.get_loader()
    
        
        pbar = tqdm(range(self.args.total_epoch), desc='Epoch', position=1)
        for epoch in pbar:
            total_policy_loss = 0
            total_q_loss1 = 0
            total_q_loss2 = 0
            total_value_loss = 0
            train_iter = tqdm(train_loader, desc='Step', position=2)
            for step, data in enumerate(train_iter):
                ## features is a onehot tensor board
                ## labels is a int tensor
                ## actions is a int tensor for action index
                states, rewards, actions, next_states = data
                states = states.cuda()
                rewards = rewards.cuda()
                actions = actions.cuda()
                next_states = next_states.cuda()
                
                predicted_q_value1, predicted_q_value2, predicted_value, new_actions, log_probs =\
                self.predict(states, rewards, actions, next_states)
                q_loss1, q_loss2 = self.update_q_net(rewards, next_states, predicted_q_value1, predicted_q_value2)
                predicted_new_q_value, value_loss = self.update_value_net(new_actions, states, log_probs, predicted_value)
                policy_loss = self.update_policy_net(log_probs, predicted_new_q_value)
                total_policy_loss += policy_loss.item()
                total_q_loss1 += q_loss1.item()
                total_q_loss2 += q_loss2.item()
                total_value_loss += value_loss.item()
                
                

            message1 = '[epoch %d] policy loss: %.3f' % (epoch + 1, total_policy_loss / len(train_loader))
            message2 = '[epoch %d] q_loss1: %.3f' % (epoch + 1, total_q_loss1 / len(train_loader))
            message3 = '[epoch %d] q_loss2: %.3f' % (epoch + 1, total_q_loss2 / len(train_loader))
            message4 = '[epoch %d] value loss: %.3f' % (epoch + 1, total_value_loss / len(train_loader))
            pbar.write(message1)
            pbar.write(message2)
            pbar.write(message3)
            pbar.write(message4)

    
        self.save_net()
        self.test(test_loader)
        self.update_target(self.args.soft_tau)
        

    def save_net(self):
        prefix = self.args.output_network
        policy_net_path = prefix + "/policy_net.pth"
        value_net_path = prefix + "/value_net.pth"
        q_value_net_path1 = prefix + "/q_value_net1.path"
        q_value_net_path2 = prefix + "/q_value_net2.path"
        
        
        torch.save(self.value_net.state_dict(), value_net_path)
        torch.save(self.policy_net.state_dict(), policy_net_path)
        torch.save(self.q_value_net1.state_dict(), q_value_net_path1)
        torch.save(self.q_value_net2.state_dict(), q_value_net_path2)
            
    def test(self, test_loader):
        print("start testing network")
        total_policy_loss = 0
        total_q_loss1 = 0
        total_q_loss2 = 0
        total_value_loss = 0
        a = 0
        b = 0
        
        with torch.no_grad():
            for data in test_loader:
                states, rewards, actions, next_states = data
                states = states.cuda()
                rewards = rewards.cuda()
                actions = actions.cuda()
                next_states = next_states.cuda()
                
                predicted_q_value1, predicted_q_value2, predicted_value, new_actions, log_probs =\
                self.predict(states, rewards, actions, next_states)
                q_loss1, q_loss2 = self.update_q_net(rewards, next_states, predicted_q_value1, predicted_q_value2, train=False)
                predicted_new_q_value, value_loss = self.update_value_net(new_actions, states, log_probs, predicted_value, train=False)
                policy_loss = self.update_policy_net(log_probs, predicted_new_q_value, train=False)
                
                total_policy_loss += policy_loss.item()
                total_q_loss1 += q_loss1.item()
                total_q_loss2 += q_loss2.item()
                total_value_loss += value_loss.item()
                
                a += predicted_new_q_value.mean().item()
                b += log_probs.mean().item()
        print("score")
        print(a/len(test_loader))
        print("entropy")
        print(b/len(test_loader))

        message1 = 'policy loss: %.3f' % (total_policy_loss / len(test_loader))
        message2 = 'q_loss1: %.3f' % (total_q_loss1 / len(test_loader))
        message3 = 'q_loss2: %.3f' % (total_q_loss2 / len(test_loader))
        message4 = 'value loss: %.3f' % (total_value_loss / len(test_loader))
        print(message1)
        print(message2)
        print(message3)
        print(message4)
        
        
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