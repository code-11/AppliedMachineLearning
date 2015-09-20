import numpy as np
from scipy import misc
from matplotlib import pylab as plt
import matplotlib.cm as cm

def setup():
	train_labels, train_data = [], []
	for line in open('./faces/train.txt'):
		im = misc.imread(line.strip().split()[0])
		train_data.append(im.reshape(2500,))
		train_labels.append(line.strip().split()[1])
	train_data, train_labels = np.array(train_data, dtype=float), np.array(train_labels, dtype=int)
	return (train_labels,train_data)

def plot_picture(pic_data):
	plt.imshow(pic_data.reshape(50,50), cmap = cm.Greys_r)
	plt.show()

train_labels,train_data=setup()
plot_picture(train_data[10, :])
# print train_data.shape, train_labels.shape
print train_data[0]