import torch
import torch.nn as nn
import torch.nn.functional as F
from DataLoader import DataLoader
import torch.optim as optim


import sys
sys.path.insert(1, '../game_simulation')
from parameters import Parameter
## image size is 12 * 6

PATH = './network/network6.pth'
TOTAL_EPOCH = 2
NUM_OF_COLOR = Parameter.NUM_OF_COLOR
ROW_DIM = Parameter.ROW_DIM
COLUMN_DIM = Parameter.COLUMN_DIM
MAX_POSSIBLE_MOVE = ROW_DIM * COLUMN_DIM
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(NUM_OF_COLOR + 1, 56, 3, padding=1)
        #self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(56, 112, 3, padding=1)

        #self.conv3 = nn.Conv2d(56, 128, 3, padding=1)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(112 * ROW_DIM * COLUMN_DIM, 144)
        self.fc2 = nn.Linear(144, MAX_POSSIBLE_MOVE)

    ## action is a onehot tensor
    def forward(self, x, action):
        # Max pooling over a (2, 2) window
        x = F.relu(self.conv1(x))
        #x = self.pool(x)
        # If the size is a square you can only specify a single number
        x = F.relu(self.conv2(x))
        #print(x.shape)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x * action

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def train(total_epoch, current_states, actions, targets, net, criterion, optimizer):
    running_loss = 0.0
    for epoch in range(total_epoch):

        # get the inputs; data is a list of [inputs, labels]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(current_states, actions)
        outputs = outputs * actions
        for i, value in enumerate(outputs[0]):
            if value > 0:
                print(i, value)
        loss = criterion(outputs, targets)
        loss.backward(retain_graph=True)
        optimizer.step()

        # print statistics
        running_loss += loss.item()

    #print('loss: %.3f' % (running_loss / len(current_states)))



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("enter new or continue or testonly")
        exit(0)

    MODE = sys.argv[1]
    if MODE != "new" and MODE != "continue" and MODE != "testonly":
        print("enter new or continue or testonly")
        exit(0)


    ## input format + loss function

    data_loader = DataLoader(labels_type="float")
    trainloader = data_loader.get_trainloader()
    testloader = data_loader.get_testloader()

    net = Net()
    if MODE == "continue" or MODE == "testonly":
        net.load_state_dict(torch.load(PATH))

    criterion = nn.SmoothL1Loss()
    #criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0005, momentum=0.7)


    if MODE != "testonly":
        for epoch in range(TOTAL_EPOCH):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                features, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(features)
                print("output")
                print(outputs)
                print("labels")
                print(labels)

                loss = criterion(outputs, labels.view(len(labels), -1))
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 500 == 499:    # print every 500 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 500))
                    running_loss = 0.0


        torch.save(net.state_dict(), PATH)



    show_target = 20
    show_count = 0
    with torch.no_grad():
        test_loss = 0.0
        count = 0
        for data in testloader:
            images, labels = data
            outputs = net(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            count += 1
            if show_count < show_target:
                print(outputs)
                print(labels)
                show_count += 1

        print("total loss: " + str(test_loss / count))
