import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as skit
import sys, os
sys.path.append(os.getcwd())
from numpy.random import randint
from itertools import product

from mnist_dataset import MNISTDataset
from utils import *
from haar_preprocessing import *


def plot_images(images, titles=None, block=True):

    fig, axes = make_fig_axes(len(images))

    vmax = min(np.max(np.abs(im)) for im in images)
    for im, title, ax in zip(images, titles, axes):
        # ax.imshow(im, cmap='gray_r')
        ax.imshow(im, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        if titles:
            ax.set_title(title)

    plt.get_current_fig_manager().window.showMaximized()
    plt.show(block=block)


class Filter:
    def __init__(self, i, j, weights, shape=None):
        self.i, self.j = i, j
        self.weights = weights
        self.shape = shape or weights.shape
        self.compute_limits()

    def apply(self, x):
        max_val = -np.inf
        for i, j in product(range(*self.lim_i), range(*self.lim_j)):
            val = np.sum(self.weights * x[i:i+self.shape[0], j:j+self.shape[1]])
            if val > max_val:
                max_val = val

        return val

    def compute_limits(self):
        min_i = self.i - self.shape[0]//2
        min_i = min_i if min_i >= 0 else 0
        max_i = self.i + self.shape[0]//2
        max_i = max_i if max_i <=28-self.shape[0] else 28-self.shape[0]
        self.lim_i = (min_i, max_i)

        min_j = self.j - self.shape[1]//2
        min_j = min_j if min_j >= 0 else 0
        max_j = self.j + self.shape[1]//2
        max_j = max_j if max_j <=28-self.shape[1] else 28-self.shape[1]
        self.lim_j = (min_j, max_j)


def filter_mnist(X, filters):
    filtered_X = np.zeros((X.shape[0], len(filters)))
    for m, x in enumerate(X):
        for n, filt in enumerate(filters):
            filtered_X[m, n] = filt.apply(x)
        print(f'Example {m+1}/{X.shape[0]}.', end='\r')
    print('\n')
    return filtered_X

if __name__ == '__main__':


    mnist = MNISTDataset.load('mnist.pkl')
    (Xtr, Ytr), (Xts, Yts) = mnist.get_train_test(center=True, reduce=True)

    # Xtr = Xtr[:20]
    # Ytr = Ytr[:20]
    # Xts = Xts[:18]
    # Yts = Yts[:18]

    ### Creating filters
    # Taking median of pixels
    median = []
    for i in range(10):
        X_i = np.array([x for x, y in zip(Xtr, Ytr) if y == i])
        median.append(np.median(X_i, axis=0))

    filters = []
    filter_shape = (5,5)
    for X in median:
        for i, j in product(range(28-filter_shape[0]+1), range(28-filter_shape[1]+1)):
            filters.append(Filter(i, j, X[i:i+filter_shape[0], j:j+filter_shape[1]]))


    dataset_shape = (10, 28-filter_shape[0]+1, 28-filter_shape[1]+1)

    ### Applying the filters
    print('Train dataset:')
    filtered_Xtr = filter_mnist(Xtr, filters)
    # for xs, y in zip(filtered_Xtr, Ytr):
    #     titles = (f'label={y} - filter={k}' for k in range(10))
    #     plot_images(xs.reshape(dataset_shape), titles, block=False)
    # plot_images(median, range(10))
    print('Test dataset:')
    filtered_Xts = filter_mnist(Xts, filters)


    filtered_mnist = MNISTDataset(filtered_Xtr, Ytr, filtered_Xts, Yts, shape=dataset_shape)
    filtered_mnist.save('median_filtered_centered_mnist.pkl')
