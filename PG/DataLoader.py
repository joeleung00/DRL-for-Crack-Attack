import torch
import torch.utils.data as Data
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from parameters import Parameter

NUM_OF_COLOR = Parameter.NUM_OF_COLOR
ROW_DIM = Parameter.ROW_DIM
COLUMN_DIM = Parameter.COLUMN_DIM
BATCH_SIZE = Parameter.BATCH_SIZE

class DataLoader:
    def __init__(self, states, actions, pis, rewards, labels_type = "int64"):
        self.data_size = len(states)
        self.batch_size = BATCH_SIZE
        self.features = torch.from_numpy(self.to_one_hot(np.array(states))).type(torch.float32)
        if labels_type == "int64":
            self.labels = torch.tensor(rewards).type(torch.int64)
        elif labels_type == "float":
            self.labels = torch.tensor(rewards).type(torch.float)

        self.actions = torch.tensor(actions).type(torch.int64)
        self.pis = torch.tensor(pis).type(torch.float)
        self.torch_dataset = Data.TensorDataset(self.features, self.labels, self.actions, self.pis)
        self.num_workers = 2
        self.shuffle_dataset = True
        self.validation_split = .2
        self.train_sampler = None
        self.test_sampler = None
        self.split_data()

    def get_trainloader(self):
        train_loader = Data.DataLoader(
            dataset=self.torch_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.train_sampler
            )
        return train_loader


    def get_testloader(self):
        test_loader = Data.DataLoader(
            dataset=self.torch_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.test_sampler
            )
        return test_loader

    def to_one_hot(self, states):
        onehot = np.zeros((self.data_size, NUM_OF_COLOR, ROW_DIM, COLUMN_DIM))
        for i in range(self.data_size):
            for row in range(ROW_DIM):
                for col in range(COLUMN_DIM):

                    color = states[i, row, col]
                    onehot[i, color, row, col] = 1
        return onehot

    def split_data(self):
        # Creating data indices for training and validation splits:
        indices = list(range(self.data_size))
        split = int(np.floor(self.validation_split * self.data_size))
        if self.shuffle_dataset :
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        # Creating PT data samplers and loaders:
        self.train_sampler = SubsetRandomSampler(train_indices)
        self.test_sampler = SubsetRandomSampler(val_indices)


if __name__ == "__main__":
    data_loader = DataLoader()
    #loader = data_loader.get_loader()
    # print(npfeatures[0])
    # print(data_loader.features[0])
    test_loader = data_loader.get_testloader()
    train_loader = data_loader.get_trainloader()
