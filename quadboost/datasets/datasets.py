import os
import numpy as np
import matplotlib.pyplot as plt
try:
    from datasets import path_to
except:
    def path_to(dataset='mnist'):
        return "data/mnist"
import sys, os
sys.path.append(os.getcwd())

import pickle as pkl
from sklearn.preprocessing import StandardScaler
import warnings


def visualize_mnist(X, Y):
    for x, y in zip(X, Y):
        vmax = np.max(np.abs(x))
        plt.imshow(x.reshape((28,28)), cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        plt.colorbar()
        plt.title(y)
        plt.show()


class ImageDataset:
    def __init__(self, Xtr, Ytr, Xts=None, Yts=None, shape=None):
        self.shape = shape or Xtr.shape[1:] # Infers shape from data if not specified.

        self.n_examples_train = Xtr.shape[0]
        self.Xtr = Xtr.reshape(self.n_examples_train, -1)
        self.Ytr = Ytr

        self.n_examples_test = Xts.shape[0] if Xts is not None else 0
        self.Xts = Xts.reshape(self.n_examples_test, -1) if Xts is not None else Xts
        self.Yts = Yts

        self.scaler = StandardScaler()
        self.scaler.fit(self.Xtr)

    def get_train_valid(self, valid=0, shuffle=True, center=True, reduce=False):
        """
        Gets the training set and splits it in a train and valid datasets.

        Args:
            valid (float between 0 and 1 or int, optional): If a float, it is interpreted as the fraction of examples to get for the validation dataset. If an integer, it is interpreted as the number of examples to select. If 0, not validation set is returned.
            shuffle (bool, optional): Whether to shuffle the examples or not. If False, the validation examples as taken from the start of the training dataset.
            center (bool, optional): Whether or not to center (zero mean) the images pixels using the training set only.
            reduce (bool, optional): Whether or not to reduce (unit variance) the images pixels using the training set only.
        """
        idx = np.arange(self.n_examples_train)
        if shuffle: np.random.shuffle(idx)

        X, Y = self.Xtr[idx], self.Ytr[idx]

        if 0 < valid < 1:
            valid = int(valid*self.n_examples_train)

        Xtr, Ytr = X[valid:], Y[valid:]
        Xval, Yval = X[:valid], Y[:valid]

        # Recompute mean and std to account for validation only if needed.
        transform_needed = center or reduce
        if valid and transform_needed:
            self.scaler = StandardScaler(with_std=reduce, with_mean=center).fit(Xtr)

        return self._prepare_data(Xtr, Ytr, transform_needed), self._prepare_data(Xval, Yval, transform_needed)

    def _prepare_data(self, X, Y, transform_needed):
        """
        Centers, reduces and reshapes data if needed.
        """
        if transform_needed:
            with warnings.catch_warnings(): # Catch conversion type warning
                warnings.simplefilter("ignore")
                X = self.scaler.transform(X)
        return X.reshape((-1,) + self.shape), Y

    def get_train(self, shuffle=False, center=True, reduce=False):
        train, valid = self.get_train_valid(0, shuffle, center, reduce)
        return train


class MNISTDataset:
    def __init__(self, Xtr, Ytr, Xts=None, Yts=None, shape=(28,28)):
        self.Xtr = Xtr.reshape(Xtr.shape[0],-1)
        self.Ytr = Ytr
        self.Xts = Xts.reshape(Xts.shape[0],-1)
        self.Yts = Yts

        self.shape = shape

        self.scaler = StandardScaler()
        self.scaler.fit(self.Xtr)

    @property
    def shape(self):
        if (not hasattr(self, '_shape')) and hasattr(self, 'side_size'): # Retrocompatibility
            self._shape = (self.side_size, self.side_size)
        return self._shape
    @shape.setter
    def shape(self, shape):
        self._shape = shape

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
        return X.reshape((-1,) + self.shape), Y

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
        print(f'Saved to {filepath+filename}')

    def test(self):
        print(self.mean.shape, self.std.shape)
        self.get_train_test(center=True, reduce=True)
        self.get_train_test(center=True, reduce=False)
        self.get_train_test(center=False, reduce=True)
        self.get_train_test(center=False, reduce=False)


if __name__ == '__main__':

    # download_mnist()
    path_to_mnist = '/home/jsleb333/OneDrive/Doctorat/Apprentissage par réseaux de neurones profonds/Datasets/mnist/raw/'
    # (Xtr, Ytr), (Xts, Yts) = load_raw_mnist()
    # dataset = MNISTDataset(Xtr, Ytr, Xts, Yts)
    # dataset.save()
    dataset = MNISTDataset.load('mnist.pkl')
    dataset.test()

    # visualize_mnist(Xtr[:5], Ytr[:5])
