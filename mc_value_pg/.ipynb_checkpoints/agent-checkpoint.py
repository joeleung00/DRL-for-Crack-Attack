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
        self.net.cuda()
        self.args = args
        
        
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
        
    def best_move(self, board):
        ## pick best action
        ## current state is not tensor
        prob, v = self.inference(board)
        #print(prob)
        index = np.argmax(prob.flatten())
        return deflatten_action(index)
    
    def MSELoss(self, vs, rewards):
        return (vs - rewards).pow(2).mean()
    
    def ACLoss(self, probs, vs, rewards, actions):
        action_log_probs = probs.log().gather(1, actions.view(len(actions), -1))
        critic = (vs - rewards).pow(2).mean()
        actor = -((rewards) * action_log_probs).mean()
        loss = actor + critic * 0.1
        return loss
    
    def extract_q_values(self, outputs, actions_indexes):
        new_outputs = torch.zeros(len(actions_indexes)).cuda()
        for i in range(len(actions_indexes)):
            new_outputs[i] = outputs[i][actions_indexes[i]]
        return new_outputs
    
    def train(self, replay_memory):
        dataloader = DataLoader(self.args.batch_size, replay_memory)
        train_loader, test_loader = dataloader.get_loader()

        total_loss = 0
        optimizer = optim.SGD(self.net.parameters(), lr=self.args.learning_rate, momentum=0.7)
        
        pbar = tqdm(range(self.args.total_epoch), desc='Epoch', position=1)
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
                # zero the parameter gradients
                optimizer.zero_grad()

                probs, vs = self.net(features)
                #vs = self.extract_q_values(vs, actions)
                
                loss = self.ACLoss(probs, vs, rewards, actions)
                total_loss += loss.item()
                loss.backward()
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
    
    
def main():
    import pickle
    with open("./50Mdata_pg", "rb") as fd:
        train_data = pickle.load(fd)
    
    
    train_data = list(map(preprocess_episode, train_data))
    train_data = [item for episode in train_data for item in episode]
    #train_data = train_data[:100000]
    #print(train_data[0])
    
    parser = argparse.ArgumentParser(description='MCTS value PG.')
    
    
    parser.add_argument('--batch_size', type=int, required=True,
                            help=' ')
    
    parser.add_argument('--learning_rate', type=float, required=True,
                            help=' ')
    
    parser.add_argument('--output_network', type=str, required=True,
                            help=' ')
    parser.add_argument('--input_network', type=str,
                            help=' ')
    parser.add_argument('--total_epoch', type=int, required=True,
                            help=' ')
    
    args = parser.parse_args()
    
    
    agent = Agent(args)
    
    agent.train(train_data)
    
## train1.pth has no preprocess
    
        
if __name__ == "__main__":
    main()
    
    