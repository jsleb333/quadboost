import os, struct
import numpy as np
import matplotlib.pyplot as plt
import gzip, urllib, io # To download MNIST
try:
    from datasets import path_to
except:
    def path_to(dataset='mnist'):
        return "data/mnist"

import pickle as pkl
from sklearn.preprocessing import StandardScaler
import warnings
from time import time

# File paths, names and format
filename_images_train = 'train-images-idx3-ubyte'
filename_labels_train = 'train-labels-idx1-ubyte'
filename_images_test = 't10k-images-idx3-ubyte'
filename_labels_test = 't10k-labels-idx1-ubyte'
data_type_header, data_bytes_header = 'I', 4
data_type_labels, data_bytes_labels = 'B', 1
data_type_images, data_bytes_images = 'B', 1


def download_mnist(filepath='data/mnist/raw/'):
    os.makedirs(filepath, exist_ok=True)
    for filename in [filename_images_train,
                     filename_images_test,
                     filename_labels_train,
                     filename_labels_test]:
        url = f'http://yann.lecun.com/exdb/mnist/{filename}.gz'
        content = urllib.request.urlopen(url)
        with open(filepath + filename, 'wb') as file:
            file.write(gzip.decompress(content.read()))

def load_raw_data(filename, N):
    with open(filename, 'rb') as file:
        header_size = 4 * data_bytes_header
        header_format = '>' + data_bytes_header*data_type_header #'>IIII' is the format of the data. '>' means big-endian, and 'I' means unsigned integer
        header = struct.unpack(header_format, file.read(header_size))

        magic_number, nber_images, nber_rows, nber_columns = header
        nber_pixels = nber_columns * nber_rows
        images = struct.unpack(
                        '>' + nber_pixels*data_type_images*N,
                        file.read(data_bytes_images*nber_pixels*N))
        images = np.array(images).reshape((N, 784))
        return header, images


def load_raw_labels(filename, N):
	with open(filename, 'rb') as labels:
		header_size = 2 * data_bytes_header
		header_format = '>' + 2 * data_type_header
		header = struct.unpack(header_format, labels.read(header_size))
		labels = struct.unpack('>'+data_bytes_labels*data_type_labels*N,labels.read(data_bytes_labels*N))
		return header, np.array(labels)


def load_raw_mnist(Ntr=60000, Nts=10000, path=None):
    mnist_path = path or path_to('mnist') + '/raw/'
    h, Xtr = load_raw_data(mnist_path + filename_images_train, Ntr)
    h, Xts = load_raw_data(mnist_path + filename_images_test, Nts)
    h, Ytr = load_raw_labels(mnist_path + filename_labels_train, Ntr)
    h, Yts = load_raw_labels(mnist_path + filename_labels_test, Nts)

    return (Xtr, Ytr), (Xts, Yts)


def visualize_mnist(X, Y):
    for x, y in zip(X, Y):
        vmax = np.max(np.abs(x))
        plt.imshow(x.reshape((28,28)), cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        plt.colorbar()
        plt.title(y)
        plt.show()


class MNISTDataset:
    def __init__(self, Xtr, Ytr, Xts=None, Yts=None, side=28):
        self.Xtr = Xtr.reshape(Xtr.shape[0],-1)
        self.Ytr = Ytr
        self.Xts = Xts.reshape(Xts.shape[0],-1)
        self.Yts = Yts

        self.side_size = side

        self.scaler = StandardScaler()
        self.scaler.fit(self.Xtr)

    @property
    def mean(self):
        return self.scaler.mean_

    @property
    def std(self):
        return self.scaler.scale_


    def get_train(self, center=True, reduce=False):
        return self._get_data(self.Xtr, self.Ytr, center, reduce)

    def get_test(self, center=True, reduce=False):
        if self.Xts is None:
            return self.Xts, self.Yts
        else:
            return self._get_data(self.Xts, self.Yts, center, reduce)

    def _get_data(self, X, Y, center, reduce):
        with warnings.catch_warnings(): # Catch conversion type warning
            warnings.simplefilter("ignore")
            if center and reduce:
                X = self.scaler.transform(X)
            elif center and not reduce:
                X = StandardScaler(with_std=False).fit(self.Xtr).transform(X)
            elif not center and reduce:
                X = StandardScaler(with_mean=False).fit(self.Xtr).transform(X)
        return X.reshape((-1, self.side_size, self.side_size)), Y

    def get_train_test(self, center=True, reduce=False):
        return self.get_train(center, reduce), self.get_test(center, reduce)


    @staticmethod
    def load(filename='mnist.pkl', filepath='./data/preprocessed/'):
        with open(filepath + filename, 'rb') as file:
            return pkl.load(file)


    def save(self, filename='mnist.pkl', filepath='./data/preprocessed/'):
        os.makedirs(filepath, exist_ok=True)
        with open(filepath + filename, 'wb') as file:
            pkl.dump(self, file)
        print('saved')


    def test(self):
        print(self.mean.shape, self.std.shape)
        self.get_train_test(center=True, reduce=True)
        self.get_train_test(center=True, reduce=False)
        self.get_train_test(center=False, reduce=True)
        self.get_train_test(center=False, reduce=False)


if __name__ == '__main__':

    download_mnist()
    # path_to_mnist = '/home/jsleb333/OneDrive/Doctorat/Apprentissage par réseaux de neurones profonds/Datasets/mnist/raw/'
    (Xtr, Ytr), (Xts, Yts) = load_raw_mnist()
    dataset = MNISTDataset(Xtr, Ytr, Xts, Yts)
    dataset.save()
    # dataset = MNISTDataset.load()
    # dataset.test()

    # visualize_mnist(Xtr[:5], Ytr[:5])
