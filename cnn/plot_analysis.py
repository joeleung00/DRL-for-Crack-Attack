import matplotlib.pyplot as plt
import sys
import numpy as np
sys.path.insert(1, '../game_simulation')
import features_extraction as fe
data = np.load("./input/full_data_6color_small.npz")
npfeatures = data["features"]
nplabels = data["labels"]
n = len(nplabels)
length = n // 10
new_features = np.array(list(map(fe.get_one_score, npfeatures)))[:length]
new_labels = np.array(list(map(lambda x: x * 3, nplabels)))[:length]

plt.scatter(new_features, new_labels)
plt.show()
