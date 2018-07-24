import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.getcwd())

from utils import haar_projection
from mnist_dataset import MNISTDataset

mnist = MNISTDataset.load()
(Xtr, Ytr), (Xts, Yts) = mnist.get_train_test(center=False, reduce=False)

haar_Xtr = haar_projection(Xtr)
haar_Xts = haar_projection(Xts)

def show_im(images):
    for im, y in zip(images, Ytr):
        vmax = np.max(np.abs(im))
        plt.imshow(im, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        plt.colorbar()
        plt.title(y)
        plt.show()

# show_im(haar_Xtr)

haar_mnist = MNISTDataset(haar_Xtr, Ytr, haar_Xts, Yts)
haar_mnist.save(filename='haar_mnist.pkl')

haar_mnist = MNISTDataset.load('haar_mnist.pkl')
(Xtr, Ytr), (Xts, Yts) = haar_mnist.get_train_test(center=False, reduce=False)
show_im(Xtr[:5])
