import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.getcwd())

from quadboost.datasets import MNISTDataset
from quadboost.utils import *
import json

mnist = MNISTDataset.load()
(Xtr, Ytr), (Xts, Yts) = mnist.get_train_test(center=False, reduce=False)

def find_images_per_label(n_images=12, label=0):
    im_iter = ((image.reshape((28,28)), i) for i, image in enumerate(Xtr) if Ytr[i] == label)
    while True:
        yield [im_num for _, im_num in zip(range(n_images), im_iter)]

def plot_images(images, titles):

    fig, axes = make_fig_axes(len(images))

    for im, title, ax in zip(images, titles, axes):
        ax.imshow(im, cmap='gray_r')
        ax.set_title(title)

    plt.get_current_fig_manager().window.showMaximized()
    plt.show()

if __name__ == '__main__':
    ### Per label visualization
    # for im_num in find_images_per_label(n_images=12, label=0):
    #     plot_images(*zip(*im_num))

    ### Ideal sample visualization
    # with open('mnist/ideal_mnist_sample.json') as file:
    #     samples = json.load(file)

    # for label, numbers in samples.items():
    #     images = [Xtr[i] for i in numbers]
    #     plot_images(images, numbers)

    ### Ideal MNIST
    with open('mnist/ideal_mnist.json') as file:
        samples = json.load(file)

    plot_images(*zip(*((Xtr[i], k) for k, i in samples.items())))
