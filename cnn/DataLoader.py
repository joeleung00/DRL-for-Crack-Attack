import torch
import torch.utils.data as Data
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
data = np.load("/Users/joeleung/Documents/CUHK/yr4_term1/csfyp/csfyp/cnn/input/full_data_6color_score.npz")
npfeatures = data["features"]
nplabels = data["labels"]
NUM_OF_COLOR = 6
ROW_DIM = 12
COLUMN_DIM = 6
class DataLoader:
	def __init__(self):
		self.data_size = len(nplabels)
		self.batch_size = 5
		self.features = torch.from_numpy(self.to_one_hot()).type(torch.float32)
		self.labels = torch.from_numpy(nplabels).type(torch.int64)
		self.torch_dataset = Data.TensorDataset(self.features, self.labels)
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

	def to_one_hot(self):
		onehot = np.zeros((self.data_size, NUM_OF_COLOR + 1, ROW_DIM, COLUMN_DIM))
		for i in range(self.data_size):
			for row in range(ROW_DIM):
				for col in range(COLUMN_DIM):
					color = npfeatures[i, row, col]
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
