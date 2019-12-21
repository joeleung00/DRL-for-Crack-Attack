from collections import deque
import pickle

train_data_path = "./train_data/"

full_data = deque(maxlen=500000)
NAME_NUMBER = 12
MAX_NUMBER = 10
STEP = 10
def load_train_data(number, name):
	if name == None:
		fullpathname = train_data_path + "data" + str(number)
	else:
		fullpathname = train_data_path + "data_" + name + "_" + str(number)
	fd = open(fullpathname, 'rb')
	return pickle.load(fd)


for i in range(1, NAME_NUMBER):
    for j in range(STEP, MAX_NUMBER, STEP):
        full_data.extend(load_train_data(j, str(name)))

print(len(full_data))

print(full_data[:10])
