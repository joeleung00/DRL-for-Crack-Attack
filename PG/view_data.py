import pickle

from collections import deque
replay_memory = deque(maxlen = 50000)
train_data_path = "./train_data/" 
def load_train_data(number, name):
	if name == None:
		fullpathname = train_data_path + "data" + str(number)
	else:
		fullpathname = train_data_path + "data_" + name + "_" + str(number)
	fd = open(fullpathname, 'rb')
	return pickle.load(fd)

replay_memory = load_train_data(1, "12")
print(replay_memory)
