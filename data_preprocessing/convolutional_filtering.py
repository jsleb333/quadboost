import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as skit
import sys, os
sys.path.append(os.getcwd())
from numpy.random import randint

from mnist_dataset import MNISTDataset
from utils import *
from haar_preprocessing import *

def plot_images(images, titles):

    fig, axes = make_fig_axes(len(images))

    for im, title, ax in zip(images, titles, axes):
        # ax.imshow(im, cmap='gray_r')
        vmax = np.max(np.abs(im))
        ax.imshow(im, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_title(title)

    plt.get_current_fig_manager().window.showMaximized()
    plt.show()

mnist = MNISTDataset.load('mnist.pkl')
(Xtr, Ytr), (Xts, Yts) = mnist.get_train_test(center=True, reduce=True)

# Xtr = Xtr[:20]
# Ytr = Ytr[:20]
# Xts = Xts[:20]
# Yts = Yts[:20]

### Creating filters

# Taking median of pixels
median = []
for i in range(10):
    X_i = np.array([x for x, y in zip(Xtr, Ytr) if y == i])
    median.append(np.median(X_i, axis=0))
# plot_images(median, range(10))

class Filter:
    def __init__(self, i, j, weights):
        self.i, self.j = i, j
        self.weights = weights


filters = []
filter_size = (5,5)
for X in median:
    for i in range(28-filter_size[0]+1):
        for j in range(28-filter_size[1]+1):
            filters.append(Filter(i, j, X[i:i+filter_size[0], j:j+filter_size[1]]))


def min_max_ij(filt):
    min_i = filt.i - filter_size[0]//2
    min_i = min_i if min_i >= 0 else 0
    max_i = filt.i + filter_size[0]//2
    max_i = max_i if max_i <=28-filter_size[0] else 28-filter_size[0]

    min_j = filt.i - filter_size[1]//2
    min_j = min_j if min_j >= 0 else 0
    max_j = filt.i + filter_size[1]//2
    max_j = max_j if max_j <=28-filter_size[1] else 28-filter_size[1]

    return (min_i, max_i), (min_j, max_j)

def apply_filter(filt, x):
    lim_i, lim_j = min_max_ij(filt)
    max_val = -np.inf
    for i in range(*lim_i):
        for j in range(*lim_j):
            val = np.sum(filt.weights * x[i:i+filter_size[0], j:j+filter_size[1]])
            if val > max_val:
                max_val = val

    return val


### Applying the filters
filtered_Xtr = np.zeros((Xtr.shape[0], len(filters)))
for m, x in enumerate(Xtr):
    for n, filt in enumerate(filters):
        filtered_Xtr[m, n] = apply_filter(filt, x)
    print(f'Train: example {m+1}/60000.', end='\r')
print('\n')

filtered_Xts = np.zeros((Xts.shape[0], len(filters)))
for m, x in enumerate(Xts):
    for n, filt in enumerate(filters):
        filtered_Xts[m, n] = apply_filter(filt, x)
    print(f'Test: example {m+1}/10000.', end='\r')
print('\n')

filtered_mnist = MNISTDataset(filtered_Xtr, Ytr, filtered_Xts, Yts, shape=(10,24,24))
filtered_mnist.save('filtered_mnist.pkl')
