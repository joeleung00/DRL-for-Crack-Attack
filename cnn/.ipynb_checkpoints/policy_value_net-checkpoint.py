import torch
import torch.nn as nn
import torch.nn.functional as F
from NDataLoader import DataLoader
import torch.optim as optim
import pickle
from collections import deque
import numpy as np
import sys
from copy import copy, deepcopy
sys.path.insert(1, '../game_simulation')
from parameters import Parameter
## image size is 12 * 6
train_data_path = "/Users/joeleung/Documents/CUHK/yr4_term1/csfyp/csfyp/PG/train_data/"
PATH = '../PG/network/network2.pth'
TOTAL_EPOCH = 10
NUM_OF_COLOR = Parameter.NUM_OF_COLOR
ROW_DIM = Parameter.ROW_DIM
COLUMN_DIM = Parameter.COLUMN_DIM
MAX_POSSIBLE_MOVE = ROW_DIM * (COLUMN_DIM - 1)

replay_memory = deque(maxlen = 50000)
torch.set_num_threads(1)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(NUM_OF_COLOR, 14, 3, padding=1)
        torch.nn.init.xavier_uniform(self.conv1.weight)
        #self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(14, 28, 3, padding=1)
        torch.nn.init.xavier_uniform(self.conv2.weight)

        #self.conv3 = nn.Conv2d(28, 56, 3, padding=1)

        #self.conv3 = nn.Conv2d(56, 128, 3, padding=1)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(NUM_OF_COLOR * ROW_DIM * COLUMN_DIM, 256)
        self.fc_bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_bn2 = nn.BatchNorm1d(128)

        self.fc3 = nn.Linear(128, MAX_POSSIBLE_MOVE)

        self.fc4 = nn.Linear(128, 1)


        #self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.005, momentum=0.5)
        #self.optimizer = optim.Adam(self.parameters(), lr=0.0005)
    ## action is a onehot tensor
    def forward(self, x):

        # Max pooling over a (2, 2) window
        #x = F.relu(self.conv1(x))
        #x = self.pool(x)
        # If the size is a square you can only specify a single number
        #x = F.relu(self.conv2(x))
        #x = F.relu(self.conv3(x))
        #print(x.shape)
        x = x.view(-1, self.num_flat_features(x))

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        pis = F.relu(self.fc3(x))
        vs = F.relu(self.fc4(x))
        return F.softmax(pis, dim=1), vs

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


    ## action is in respective of the network
    def train(self, all_states, all_actions, all_pis, all_rewards):
        data_loader = DataLoader(all_states, all_actions, all_pis, all_rewards)
        trainloader = data_loader.get_trainloader()
        testloader = data_loader.get_testloader()
        batch_size = data_loader.batch_size
        data_size = len(all_actions)
        min_batch_per_look = data_size // batch_size // 4
        running_loss = 0.0
        for epoch in range(TOTAL_EPOCH):
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels, actions, pis]
                states, rewards, actions, pis = data
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                probi, v = self.forward(states)
                action_log_probs = probi.log().gather(1, actions.view(len(actions), -1))

                critic = (v - rewards).pow(2).mean()
                actor = -(rewards * action_log_probs).mean()
                loss = actor
                #loss = critic + 10 * actor
                loss.backward(retain_graph=True)
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % min_batch_per_look == min_batch_per_look - 1:    # print every 500 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / min_batch_per_look))
                    running_loss = 0.0

        torch.save(net.state_dict(), PATH)

        test_loss = 0
        count = 0
        show_count = 0
        show_target = 20
        match = 0
        total = 0
        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                states, rewards, actions, pis = data

                probi, v = self.forward(states)
                _, indexs = torch.max(probi, 1)


                action_log_probs = probi.log().gather(1, actions.view(len(actions), -1))

                critic = (v - rewards).pow(2).mean()
                actor = -(rewards * action_log_probs).mean()
                loss = actor
                #loss = critic + 10 * actor

                for i in range(len(probi)):
                    if indexs[i].item() == actions[i].item():
                        match += 1
                    total += 1

                test_loss += loss.item()
                count += 1
                if show_count < show_target:
                    if show_count == show_target - 1:

                        print("action prob")
                        print(probi)

                    print("actor: ")
                    print(actor)
                    show_count += 1

            print("total loss: " + str(test_loss / count))
            print("total match: " + str(match / total))



def get_batch_from_memory():
    ## min_batch are all python data type (state, pi, reward)
    train_data = replay_memory

    ## they are batch states
    states = np.zeros((len(train_data), ROW_DIM, COLUMN_DIM)).astype(int)
    actions = []
    pis = []
    rewards = []
    for i, value in enumerate(train_data):
        states[i] = value[0]
        actions.append(value[1])
        pis.append(value[2])
        rewards.append(value[3])

    return (states, actions, pis, rewards)

def load_train_data():
    global replay_memory
    fullpathname = train_data_path + "full_data"
    fd = open(fullpathname, 'rb')
    replay_memory =  pickle.load(fd)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("enter new or continue or testonly")
        exit(0)

    MODE = sys.argv[1]
    if MODE != "new" and MODE != "continue" and MODE != "testonly":
        print("enter new or continue or testonly")
        exit(0)

    net = Net()
    load_train_data()
    #print(len(replay_memory))
    #print(replay_memory[10])
    states, actions, pis, rewards = get_batch_from_memory()
    net.train(states, actions, pis, rewards)
