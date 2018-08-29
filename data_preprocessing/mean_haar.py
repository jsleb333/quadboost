import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as skit
import sys, os
sys.path.append(os.getcwd())

from mnist_dataset import MNISTDataset
from utils import *
from haar_preprocessing import *

def plot_images(images, titles):

    fig, axes = make_fig_axes(len(images))

    for im, title, ax in zip(images, titles, axes):
        ax.imshow(im, cmap='gray_r')
        ax.set_title(title)

    plt.get_current_fig_manager().window.showMaximized()
    plt.show()

mnist = MNISTDataset.load('haar_mnist_pad.pkl')
(Xtr, Ytr), (Xts, Yts) = mnist.get_train_test(center=False, reduce=False)

mean_haar = []
for i in range(10):
    X_i = np.array([x for x, y in zip(Xtr, Ytr) if y == i])
    mean_haar.append(np.mean(X_i, axis=0))

# show_im(mean_haar, range(10))
mean_haar = [2*(mh-np.min(mh))/(np.max(mh)-np.min(mh))-1 for mh in mean_haar]
mean_haar = [skit.resize(mh, (8,8), order=1, mode='reflect', anti_aliasing=True) for mh in mean_haar]
mean_haar = [2*(mh-np.min(mh))/(np.max(mh)-np.min(mh))-1 for mh in mean_haar]

plot_images(mean_haar, range(10))
encodings = {str(i):mh.reshape(-1).tolist() for i, mh in enumerate(mean_haar)}
print(encodings)
