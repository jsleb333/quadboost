import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.getcwd())

from utils import haar_projection
from mnist_dataset import MNISTDataset


def haar_projection(images):
    """
    Recursively computes the Haar projection of an array of 2D images.
    Uses a non-standard Haar projection for sizes that are not powers of 2.
    """
    projected_images = images.astype(dtype=float)
    m, N, _ = images.shape
    while N > 1:
        projector = haar_projector(N)
        np.matmul(np.matmul(projector, projected_images[:,:N,:N]), projector.T, out=projected_images[:,:N,:N])
        N = N//2 if N%2 == 0 else N//2+1
        # print(N)
    return projected_images


def haar_projector(N):
    """
    Generates the Haar projector of size N.
    """
    projection = np.zeros((N,N))
    for i in range(N):
        projection[i//2,i] = 1
        projection[(i+N)//2,i] = 1 if (i+N)%2 == 0 else -1
    projection /= 2

    return projection


def show_im(images):
    for im, y in zip(images, Ytr):
        vmax = np.max(np.abs(im))
        plt.imshow(im, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        plt.colorbar()
        plt.title(y)
        plt.show()

        
if __name__ == '__main__':
    mnist = MNISTDataset.load()
    (Xtr, Ytr), (Xts, Yts) = mnist.get_train_test(center=False, reduce=False)

    haar_Xtr = haar_projection(Xtr)
    haar_Xts = haar_projection(Xts)

    # show_im(haar_Xtr)

    haar_mnist = MNISTDataset(haar_Xtr, Ytr, haar_Xts, Yts)
    haar_mnist.save(filename='haar_mnist.pkl')

    haar_mnist = MNISTDataset.load('haar_mnist.pkl')
    (Xtr, Ytr), (Xts, Yts) = haar_mnist.get_train_test(center=False, reduce=False)
    show_im(Xtr[:5])
