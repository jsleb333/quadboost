import numpy as np
import scipy.ndimage as spim
import scipy.misc as spm
import skimage.transform as skit
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.getcwd())

from mnist_dataset import MNISTDataset
from mnist.mnist_visualization import plot_images
from utils import *
import json

mnist = MNISTDataset.load()
(Xtr, Ytr), (Xts, Yts) = mnist.get_train_test(center=False, reduce=False)

with open('mnist/ideal_mnist.json') as file:
    samples = json.load(file)

images, numbers = zip(*((Xtr[i].reshape((28,28)), k) for k, i in samples.items()))
### Cropping of zeros
images = [im[np.any(im, axis=1)] for im in images]
images = [im.T[np.any(im.T, axis=1)].T for im in images]
### Final manual cropping
images[0] = images[0][:,:-1]
images[2] = images[2][:,:-1]
images[3] = images[3][:,:-1]
images[4] = images[4][:,:-1]
images[5] = images[5][:,1:-1]
images[7] = images[7][:,1:-1]
images[8] = images[8][:,:-1]

### Filling with extra zeros for the 1
images[1] = np.pad(images[1], [(0,0), (3,2)], 'constant')

### Rescaling to the interval [-1, 1]
images = [2*(im-np.min(im))/(np.max(im)-np.min(im))-1 for im in images]

### Resizing image to have a smaller encoding
# images = [spim.zoom(im, (8/im.shape[0], 6/im.shape[1]), order=4, mode='nearest') for im in images]
# images = [spm.imresize(im, (8, 6), interp='bicubic') for im in images]
images = [skit.resize(im, (8,6), order=3, mode='reflect', anti_aliasing=True) for im in images]
# for im in images: print(np.min(im))

# ### Rescaling to the interval [-1, 1]
images = [2*(im-np.min(im))/(np.max(im)-np.min(im))-1 for im in images]

# titles = [(im.shape, np.max(im), np.min(im)) for im in images]
# plot_images(images, titles)
encodings = {str(i):im.reshape(-1).tolist() for i, im in enumerate(images)}
print(encodings)
