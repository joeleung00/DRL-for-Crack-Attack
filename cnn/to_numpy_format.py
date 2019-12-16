import numpy as np
# import matplotlib.pyplot as plt

NUM_OF_COLOR = Parameter.NUM_OF_COLOR
ROW_DIM = Parameter.ROW_DIM
COLUMN_DIM = Parameter.COLUMN_DIM

raw_input = np.loadtxt("./input/full_data_6color_small", delimiter=',')
features = np.delete(raw_input, -1, axis=1)
labels = raw_input[:, -1]
bins = [0, 3, 6, 9, 12]
labels = np.digitize(labels, bins,right=False) - 1
# onehot_labels = np.zeros((len(labels), len(bins)))
# onehot_labels[np.arange(len(labels)), labels] = 1
# onehot_labels = onehot_labels.astype("int16")
features = features.astype("int16")
features = features.reshape(-1, ROW_DIM, COLUMN_DIM)
np.savez("./input/full_data_6color_small.npz", features=features, labels=labels)

#
# _ = plt.hist(labels, bins='auto')
# plt.title("Histogram with 'auto' bins")
# plt.show()
