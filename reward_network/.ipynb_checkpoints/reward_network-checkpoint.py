import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader import dataloader
import torch.optim as optim
import argparse
from resnet import ResidualNet

import sys
sys.path.insert(1, '../game_simulation')
from parameters import Parameter

NUM_OF_COLOR = Parameter.NUM_OF_COLOR
ROW_DIM = Parameter.ROW_DIM
COLUMN_DIM = Parameter.COLUMN_DIM


torch.set_num_threads(15)

TOTAL_EPOCH = 15
PATH = "./network/n4.pth"


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
         
        self.conv1 = nn.Conv2d(NUM_OF_COLOR, 32, 3, padding=1)
    
        self.conv2 = nn.Conv2d(32, 128, 3, padding=1)

        #self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        
        #self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        
        #self.conv4 = nn.Conv2d(56, 56, 3, padding=1)


        self.fc1 = nn.Linear(128 * ROW_DIM * COLUMN_DIM, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 36)

        
    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        #x = F.relu(self.conv3(x))
        #print(x.shape)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



    
if __name__ == "__main__":
    
    dl = dataloader("./output/0.9M_data", 64, seed=12)
    trainloader, testloader = dl.get_loader()
    
    net = ResidualNet(NUM_OF_COLOR, ROW_DIM * COLUMN_DIM, 128, ROW_DIM * COLUMN_DIM)
    #net.load_state_dict(torch.load(PATH))
    
    net.cuda()
    
    optimizer = optim.SGD(net.parameters(), lr=0.0005, momentum=0.6)
    
    for epoch in range(TOTAL_EPOCH):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            features, labels = data
            features = features.cuda()
            labels = labels.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(features)
    
            loss = torch.pow((outputs - labels), 2).sum(axis=1).mean()

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 500 == 499:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 500))
                running_loss = 0.0


    torch.save(net.state_dict(), PATH)
    
    
    test_loss = 0
    total = 0
    with torch.no_grad():
        print("Start testing")
        for data in testloader:
            features, labels = data
            features = features.cuda()
            labels = labels.cuda()
            
            outputs = net(features)
            loss = torch.pow((outputs - labels), 2).sum(axis=1).mean()
            test_loss += loss.item()
            total += 1

    print("test loss: " + str(test_loss / total))
    print(outputs[0].reshape(ROW_DIM, COLUMN_DIM))
    print(labels[0].reshape(ROW_DIM, COLUMN_DIM))

        
        
        
        