import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import numpy as np
from tqdm import tqdm, trange
from policy_net import ResidualNet as PolicyNet
from dueling_net import ResidualNet as QValueNet
from dataloader import DataLoader
import torch.utils.data as Data
from utility import *
#from eval_game import eval_game
from torch.distributions.categorical import Categorical
import os.path
class PolicyAgent:
    def __init__(self, args, debug=False):
        self.policy_net = PolicyNet(NUM_OF_COLOR, ROW_DIM * COLUMN_DIM, ROW_DIM * COLUMN_DIM, 128)
        self.q_value_net1 = QValueNet(NUM_OF_COLOR, ROW_DIM * COLUMN_DIM, ROW_DIM * COLUMN_DIM, 128)
        self.q_value_net2 = QValueNet(NUM_OF_COLOR, ROW_DIM * COLUMN_DIM, ROW_DIM * COLUMN_DIM, 128)
        self.q_value_target1 = QValueNet(NUM_OF_COLOR, ROW_DIM * COLUMN_DIM, ROW_DIM * COLUMN_DIM, 128)
        self.q_value_target2 = QValueNet(NUM_OF_COLOR, ROW_DIM * COLUMN_DIM, ROW_DIM * COLUMN_DIM, 128)
        
        if debug:
            if type(debug) == str:
                self.load_net(debug)
        else:
            self.args = args
            if args.input_network:
                self.load_net(args.input_network)
    
            self.soft_q_optimizer1 = optim.Adam(self.q_value_net1.parameters(), lr=args.q_lr)
            self.soft_q_optimizer2 = optim.Adam(self.q_value_net2.parameters(), lr=args.q_lr)
            self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=args.policy_lr)
            self.q_criterion1 = nn.MSELoss()
            self.q_criterion2 = nn.MSELoss()
            self.init_alpha()
              
        self.to_cuda()
        
    def init_alpha(self):
        self.target_entropy = -np.log((1.0 / ROW_DIM * COLUMN_DIM)) * 0.98
        self.log_alpha = torch.tensor(self.args.init_log_alpha, requires_grad=True)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=self.args.alpha_lr)
        self.alpha = self.log_alpha.exp()
        
        
    def update_alpha(self):
        self.alpha = self.log_alpha.exp()
        print("alpha:", str(self.alpha))
    
    def to_cuda(self):
        self.policy_net.cuda()
        self.q_value_net1.cuda()
        self.q_value_net2.cuda()
        self.q_value_target1.cuda()
        self.q_value_target2.cuda()
        
        
    def load_net(self, prefix):
        policy_net_path = prefix + "/policy_net.pth"
        q_value_net_path1 = prefix + "/q_value_net1.path"
        q_value_net_path2 = prefix + "/q_value_net2.path"
        print("net loaded")
        self.policy_net.load_state_dict(torch.load(policy_net_path))
        self.q_value_net1.load_state_dict(torch.load(q_value_net_path1))
        self.q_value_net2.load_state_dict(torch.load(q_value_net_path2))
        
        self.q_value_target1.load_state_dict(torch.load(q_value_net_path1))
        self.q_value_target2.load_state_dict(torch.load(q_value_net_path2))
        
        
    def get_action(self, board):
        prob = self.policy_inference([board])[0].cpu().numpy()
        prob = prob / prob.sum()
        action = np.random.choice(range(len(prob)), p=prob)
        return deflatten_action(action)
    

    def evaluate(self, states):
        action_probabilities = self.policy_net(states)
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probabilities + z)
        return action_probabilities, log_action_probabilities
       
 
    def update_target(self, soft_tau):
        for target_param, param in zip(self.q_value_target1.parameters(), self.q_value_net1.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
            
        for target_param, param in zip(self.q_value_target2.parameters(), self.q_value_net2.parameters()):
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

    
    def extract_q_values(self, outputs, actions_indexes):
        return outputs.gather(1, actions_indexes.view(len(actions_indexes), -1)).view(-1)

#     def extract_q_values(self, outputs, actions_indexes):
#         new_outputs = torch.zeros(len(actions_indexes)).cuda()
#         for i in range(len(actions_indexes)):
#             new_outputs[i] = outputs[i][actions_indexes[i]]
#         return new_outputs
    
    def q_net_loss(self, states, rewards, actions, next_states):
        with torch.no_grad():
            actions_probs, log_actions_probs = self.evaluate(next_states)
            
            target_next_q_values1 = self.q_value_target1(next_states)
            target_next_q_values2 = self.q_value_target2(next_states)
            
            min_target_next_q_value = actions_probs * (torch.min(target_next_q_values1, target_next_q_values2) - self.alpha.cuda() * log_actions_probs)
            min_target_next_q_value = min_target_next_q_value.mean(dim=1).unsqueeze(-1)
            target_q_value = rewards.unsqueeze(-1) + min_target_next_q_value 
        actions = actions.unsqueeze(-1)
        q_value1 = self.q_value_net1(states).gather(1, actions)
        q_value2 = self.q_value_net2(states).gather(1, actions)
        loss1 = self.q_criterion1(q_value1, target_q_value)
        loss2 = self.q_criterion2(q_value2, target_q_value)

        return loss1, loss2
    
    
    def policy_net_loss(self, states):
        
        actions_probs, log_actions_probs = self.evaluate(states)
        q_value1 = self.q_value_net1(states)
        q_value2 = self.q_value_net2(states)
        min_q_value = torch.min(q_value1, q_value2)
        
        policy_loss = actions_probs * (self.alpha.cuda().detach() * log_actions_probs - min_q_value)
        policy_loss = policy_loss.mean()
        return policy_loss, actions_probs, log_actions_probs
    
    def alpha_loss_fn(self, states):
        with torch.no_grad():
            actions_probs, log_actions_probs = self.evaluate(states)
        alpha_loss = -(self.log_alpha.cuda() * (log_actions_probs + self.target_entropy)) * actions_probs
        return alpha_loss.mean()

    def update(self, q_loss1, q_loss2, policy_loss, alpha_loss):
        ### for q_net1
        self.soft_q_optimizer1.zero_grad()
        q_loss1.backward()
        self.soft_q_optimizer1.step()
        ############
        
        ### for q_net2
        self.soft_q_optimizer2.zero_grad()
        q_loss2.backward()
        self.soft_q_optimizer2.step()
        #############
        
        ### for policy net
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        #########
        
        ### for alpha
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        ########
        
    
    def train(self, replay_memory, total_epoch):
        
        dataloader = DataLoader(self.args.batch_size, replay_memory)
        train_loader, test_loader = dataloader.get_loader()
    
        
        pbar = tqdm(range(total_epoch), desc='Epoch', position=1)
        for epoch in pbar:
            total_policy_loss = 0
            total_q_loss1 = 0
            total_q_loss2 = 0
            total_alpha_loss = 0
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
            
                q_loss1, q_loss2 = self.q_net_loss(states, rewards, actions, next_states)
                policy_loss, actions_probs, log_actions_probs = self.policy_net_loss(states)
                alpha_loss = self.alpha_loss_fn(states)
                self.update(q_loss1, q_loss2, policy_loss, alpha_loss)
                
                total_policy_loss += policy_loss.item()
                total_q_loss1 += q_loss1.item()
                total_q_loss2 += q_loss2.item()
                total_alpha_loss += alpha_loss.item()
                

            message1 = '[epoch %d] policy loss: %.3f' % (epoch + 1, total_policy_loss / len(train_loader))
            message2 = '[epoch %d] q_loss1: %.3f' % (epoch + 1, total_q_loss1 / len(train_loader))
            message3 = '[epoch %d] q_loss2: %.3f' % (epoch + 1, total_q_loss2 / len(train_loader))
            message4 = '[epoch %d] alpha loss: %.3f' % (epoch + 1, total_alpha_loss / len(train_loader))
            pbar.write(message1)
            pbar.write(message2)
            pbar.write(message3)
            pbar.write(message4)
        self.save_net()
        self.update_alpha()
        self.test(test_loader)
        self.update_target(self.args.soft_tau)
        

    def save_net(self):
        prefix = self.args.output_network
        policy_net_path = prefix + "/policy_net.pth"
        q_value_net_path1 = prefix + "/q_value_net1.path"
        q_value_net_path2 = prefix + "/q_value_net2.path"
        
    
        torch.save(self.policy_net.state_dict(), policy_net_path)
        torch.save(self.q_value_net1.state_dict(), q_value_net_path1)
        torch.save(self.q_value_net2.state_dict(), q_value_net_path2)
            
    def test(self, test_loader):
        print("start testing network")
        total_policy_loss = 0
        total_q_loss1 = 0
        total_q_loss2 = 0
        total_alpha_loss = 0

        a = 0
        b = 0
        
        with torch.no_grad():
            for data in test_loader:
                states, rewards, actions, next_states = data
                states = states.cuda()
                rewards = rewards.cuda()
                actions = actions.cuda()
                next_states = next_states.cuda()
                
                q_loss1, q_loss2 = self.q_net_loss(states, rewards, actions, next_states)
                policy_loss, actions_probs, log_actions_probs = self.policy_net_loss(states)
                alpha_loss = self.alpha_loss_fn(states)
                total_policy_loss += policy_loss.item()
                total_q_loss1 += q_loss1.item()
                total_q_loss2 += q_loss2.item()
                total_alpha_loss += alpha_loss.item()
                
#                 a += predicted_new_q_value.mean().item()
#                 b += log_probs.mean().item()
                
#         print("score")
#         print(a/len(test_loader))
#         print("entropy")
#         print(b/len(test_loader))

        message1 = 'policy loss: %.3f' % (total_policy_loss / len(test_loader))
        message2 = 'q_loss1: %.3f' % (total_q_loss1 / len(test_loader))
        message3 = 'q_loss2: %.3f' % (total_q_loss2 / len(test_loader))
        message4 = 'alpha loss: %.3f' % (total_alpha_loss / len(test_loader))
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