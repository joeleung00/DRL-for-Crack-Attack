import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import argparse
import numpy as np
from tqdm import tqdm, trange
from DataLoader import DataLoader
from net import ResidualNet
from utility import *

class Agent:
    def __init__(self, args, debug=False):
        self.net = ResidualNet(NUM_OF_COLOR, ROW_DIM * COLUMN_DIM, ROW_DIM * COLUMN_DIM, 128)
        if type(debug) == str:
            self.net.load_state_dict(torch.load(debug))
        elif args != None:
            if args.input_network:
                self.net.load_state_dict(torch.load(args.input_network))
                print("network loaded")
        self.net.cuda()
        self.args = args
        if args != None:
            self.init_epsilon = args.init_epsilon
            self.final_epsilon = args.final_epsilon
            self.epsilon_decay = (self.init_epsilon - self.final_epsilon) / args.total_iterations
            self.epsilon = self.init_epsilon
        else: self.epsilon = 0.9

    def update_epsilon(self):
        if self.epsilon > self.final_epsilon:
            self.epsilon -= self.epsilon_decay
        
    def inference(self, board):
        onehot_board = to_one_hot(board)
        onehot_board = torch.from_numpy(onehot_board).type(torch.float32)
        onehot_board = onehot_board.cuda()
        with torch.no_grad():
            Probs, values = self.net(onehot_board.unsqueeze(0)) ## output is prob, value tensor for all actionss(size of  72)
        #max_value = torch.max(values[0])
        #mean = torch.mean(values[0])
        #print(values[0])
        return Probs[0].cpu().numpy().reshape(ROW_DIM, COLUMN_DIM), values[0].cpu().numpy()
        
    def best_move(self, board, possible_actions=None):
        ## pick best action
        ## current state is not tensor
        prob, v = self.inference(board)
        if possible_actions != None:
            prob = self.mask_actions(prob, possible_actions)
        index = np.argmax(prob.flatten())
        return deflatten_action(index)
    
    
    def mask_actions(self, prob, possible_actions, mask_value=-100):
        masks = np.zeros(prob.shape)
        for action in possible_actions:
            r, c = action[0], action[1]
            masks[r][c] = 1.0
        return prob * masks
    
    def greedy_policy(self, board, possible_actions):
        if np.random.rand() <= self.epsilon:
            ## pick random action
            choice = np.random.randint(len(possible_actions))
            return possible_actions[choice]
        else:
            return self.best_move(board, possible_actions)
    
    def MSELoss(self, vs, rewards):
        return (vs - rewards).pow(2).mean()
    
    def ACLoss(self, probs, vs, rewards, actions):
        action_log_probs = probs.log().gather(1, actions.view(len(actions), -1))
        #critic = 1/2 * (vs - rewards).pow(2).mean()
        #advantage = rewards - vs
        #actor = -(advantage.detach() * action_log_probs).mean()
        actor = -(rewards * action_log_probs).mean()
        loss = actor# + critic
        return loss
    
    def extract_q_values(self, outputs, actions_indexes):
        new_outputs = torch.zeros(len(actions_indexes)).cuda()
        for i in range(len(actions_indexes)):
            new_outputs[i] = outputs[i][actions_indexes[i]]
        return new_outputs
    
    def train(self, replay_memory, total_epoch_arg=None):
        dataloader = DataLoader(self.args.batch_size, replay_memory, path="./input/tensor_100Mdata")
        train_loader, test_loader = dataloader.get_loader()
        clip_grad = 3
        total_loss = 0
        optimizer = optim.SGD(self.net.parameters(), lr=self.args.learning_rate, momentum=0.7)
        total_epoch = self.args.total_epoch if total_epoch_arg == None else total_epoch_arg
        pbar = tqdm(range(total_epoch), desc='Epoch', position=1)
        for epoch in pbar:
            train_iter = tqdm(train_loader, desc='Step', position=2)
            for step, data in enumerate(train_iter):
                ## features is a onehot tensor board
                ## rewards is a float tensor
                ## actions is a int tensor for action index
                features, rewards, actions = data
                features = features.cuda()
                rewards = rewards.cuda()
                actions = actions.cuda()
#                 print(features)
#                 print(rewards)
#                 print(actions)
#                 exit(0)
                # zero the parameter gradients
                optimizer.zero_grad()

                probs, vs = self.net(features)
                #vs = self.extract_q_values(vs, actions)
                
                loss = self.ACLoss(probs, vs, rewards, actions)
                total_loss += loss.item()
                loss.backward()
                #clip_gradient(optimizer, clip_grad)
                optimizer.step()
                

            message = '[epoch %d] loss: %.3f' % (epoch + 1, total_loss / len(train_loader))
            pbar.write(message)                                     
            total_loss = 0.0
                    
        torch.save(self.net.state_dict(), self.args.output_network)
        self.test(test_loader)

            
    def test(self, test_loader):
        print("start testing network")
        test_loss = 0
        total = 0
        mse = 0
        with torch.no_grad():
            for data in test_loader:
                features, rewards, actions = data
                features = features.cuda()
                rewards = rewards.cuda()
                actions = actions.cuda()
                
                probs, vs = self.net(features)
                #vs = self.extract_q_values(vs, actions)
                
                loss = self.ACLoss(probs, vs, rewards, actions)
                mse += self.MSELoss(vs, rewards).item()
                test_loss += loss.item()
                total += 1

        print("test loss: " + str(test_loss / total))
        print("MSE (value loss): ", str(mse / total))
    

def preprocess_episode(episode_data):
    gamma = 0.6
    for i in reversed(range(len(episode_data) - 1)):
        episode_data[i][1] += gamma * episode_data[i+1][1]
    return episode_data
    
def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)
    
def main():
    
    parser = argparse.ArgumentParser(description='MCTS value PG.')
    
    
    parser.add_argument('--batch_size', type=int, required=True,
                            help=' ')
    
    parser.add_argument('--learning_rate', type=float, required=True,
                            help=' ')
    
    parser.add_argument('--output_network', type=str, required=True,
                            help=' ')
    parser.add_argument('--input_network', type=str,
                            help=' ')
    parser.add_argument('--init_epsilon', type=float, default=0.1,
                        help='init_epsilon')
    
    parser.add_argument('--final_epsilon', type=float, default=0.1,
                        help='final_epsilon')
    
    parser.add_argument('--total_epoch', type=int, required=True,
                        help='total epoch of trainig network')
    
    parser.add_argument('--total_iterations', type=int, default=1,
                        help='total_iterations of playing + learning')
    
    
    args = parser.parse_args()
    
    
    agent = Agent(args)
    
    
#     import pickle
#     with open("./input/100Mdata", "rb") as fd:
#         train_data = pickle.load(fd)
    
    
#     train_data = list(map(preprocess_episode, train_data))
#     train_data = [item for episode in train_data for item in episode]
    #train_data = train_data[:100000]
    #print(train_data[0])
    
    agent.train(None)
    
## train1.pth has no preprocess
    
        
if __name__ == "__main__":
    main()
    
    