import os
import numpy as np
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.getcwd())

import pickle as pkl
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array
from sklearn.preprocessing import StandardScaler
def transform(self, X, copy=None): # Monkey patch the dtype of
    """Perform standardization by centering and scaling

    Args:
        X : array-like, shape [n_samples, n_features]
            The data used to scale along the features axis.
        copy : bool, optional (default: None)
            Copy the input X or not.
    """
    check_is_fitted(self, 'scale_')

    copy = copy or self.copy
    X = check_array(X, accept_sparse='csr', copy=copy, estimator=self, dtype=np.float32)

    if self.with_mean:
        X -= self.mean_
    if self.with_std:
        X /= self.scale_
    return X
StandardScaler.transform = transform

import warnings
try:
    from quadboost.utils import identity_func
except ModuleNotFoundError:
    from utils import identity_func


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
        self.fit_scaler(self.Xtr, center=True, reduce=True)

    @property
    def n_examples_train(self): # Retrocompatibility
        if not hasattr(self, '_n_examples_train'):
            self.n_examples_train = self.Xtr.shape[0]
        return self._n_examples_train
    @n_examples_train.setter
    def n_examples_train(self, value):
        self._n_examples_train = value

    @property
    def n_examples_test(self): # Retrocompatibility
        if not hasattr(self, '_n_examples_test'):
            self.n_examples_test = self.Xts.shape[0] if self.Xts is not None else 0
        return self._n_examples_test
    @n_examples_test.setter
    def n_examples_test(self, value):
        self._n_examples_test = value

    @property
    def shape(self): # Retrocompatibility
        if (not hasattr(self, '_shape')) and hasattr(self, 'side_size'):
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

    def fit_scaler(self, X, *, center, reduce):
        if center or reduce: # Only if needed
            with warnings.catch_warnings(): # Catch conversion type warning
                warnings.simplefilter("ignore")
                self.scaler = StandardScaler(with_std=reduce, with_mean=center).fit(X.reshape(X.shape[0],-1))
        else:
            self.scaler.transform = identity_func

    def get_train_valid(self, valid=0, center=False, reduce=False, shuffle=True):
        """
        Gets the training set and splits it in a train and valid datasets.

        Args:
            valid (float between 0 and 1 or int, optional): If a float, it is interpreted as the fraction of examples to get for the validation dataset. If an integer, it is interpreted as the number of examples to select. If 0, not validation set is returned.
            center (bool, optional): Whether or not to center (zero mean) the images pixels using the training set only.
            reduce (bool, optional): Whether or not to reduce (unit variance) the images pixels using the training set only.
            shuffle (bool, optional): Whether to shuffle the examples or not. If False, the validation examples as taken from the start of the training dataset.
        """
        idx = np.arange(self.n_examples_train)
        if shuffle: np.random.shuffle(idx)

        X, Y = self.Xtr[idx], self.Ytr[idx]

        if 0 < valid < 1:
            valid = int(valid*self.n_examples_train)

        Xtr, Ytr = X[valid:], Y[valid:]
        Xval, Yval = X[:valid], Y[:valid]

        # Recompute mean and std to account for validation.
        self.fit_scaler(Xtr, center=center, reduce=reduce)

        return self.transform_data(Xtr, Ytr), self.transform_data(Xval, Yval)

    def transform_data(self, X, Y):
        """
        Centers, reduces and reshapes data if needed.
        """
        if X.size > 0:
            with warnings.catch_warnings(): # Catch conversion type warning
                warnings.simplefilter("ignore")
                X = self.scaler.transform(X)
        return X.reshape((-1,) + self.shape), Y

    def get_train(self, center=False, reduce=False, shuffle=True):
        """
        Gets the training dataset with wanted transformations.

        Args:
            center (bool, optional): Whether or not to center (zero mean) the images pixels using the training set only.
            reduce (bool, optional): Whether or not to reduce (unit variance) the images pixels using the training set only.
            shuffle (bool, optional): Whether to shuffle the examples or not.
        """
        train, valid = self.get_train_valid(0, center=center, reduce=reduce, shuffle=shuffle)
        return train

    def get_test(self, center=False, reduce=False, scale_with=None):
        """
        Gets the testing dataset with wanted transformations.

        Args:
            center (bool, optional): Whether or not to center (zero mean) the images pixels using the 'scale_with' dataset.
            reduce (bool, optional): Whether or not to reduce (unit variance) the images pixels using the 'scale_with' dataset.
            scale_with (Array of shape (n_examples, ...) or None, optional): Dataset to use to scale the test set. If None, the complete training set is used.
        """
        X = scale_with if scale_with is not None else self.Xtr
        self.fit_scaler(X, center=center, reduce=reduce)
        return self._get_test()

    def _get_test(self):
        if self.Xts is None:
            return self.Xts, self.Yts
        else:
            return self.transform_data(self.Xts, self.Yts)

    def get_train_valid_test(self, valid=0, center=False, reduce=False, shuffle=True):
        train, valid = self.get_train_valid(valid, center=center, reduce=reduce, shuffle=shuffle)
        test = self._get_test()
        return train, valid, test

    def get_train_test(self, center=False, reduce=False):
        train, valid, test = self.get_train_valid_test(0, center=center, reduce=reduce)
        return train, test

    @staticmethod
    def load(filename, filepath='./data/'):
        with open(filepath + filename, 'rb') as file:
            return pkl.load(file)

    def save(self, filename, filepath='./data/'):
        os.makedirs(filepath, exist_ok=True)
        with open(filepath + filename, 'wb') as file:
            pkl.dump(self, file)
        print(f'Saved to {filepath+filename}')


class MNISTDataset(ImageDataset):
    def __init__(self, Xtr, Ytr, Xts=None, Yts=None, shape=(28,28)):
        super().__init__(Xtr, Ytr, Xts, Yts, shape)

    @staticmethod
    def load(filename='mnist.pkl', filepath='./quadboost/data/mnist/preprocessed/'):
        return ImageDataset.load(filename, filepath)

    def save(self, filename='mnist.pkl', filepath='./quadboost/data/mnist/preprocessed/'):
        super().save(filename, filepath)


class CIFAR10Dataset(ImageDataset):
    def __init__(self, Xtr, Ytr, Xts=None, Yts=None, shape=(3,32,32)):
        super().__init__(Xtr, Ytr, Xts, Yts, shape)

    @staticmethod
    def load(filename='cifar10.pkl', filepath='./quadboost/data/cifar10/preprocessed/'):
        return ImageDataset.load(filename, filepath)

    def save(self, filename='cifar10.pkl', filepath='./quadboost/data/cifar10/preprocessed/'):
        super().save(filename, filepath)


def _generate_mnist_dataset():
    (Xtr, Ytr), (Xts, Yts) = load_mnist()
    print(Xtr.shape, Xts.shape)
    dataset = MNISTDataset(Xtr, Ytr, Xts, Yts)
    dataset.save()

def _generate_cifar10_dataset():
    (Xtr, Ytr), (Xts, Yts) = load_cifar10()
    print(Xtr.shape, Xts.shape)
    dataset = CIFAR10Dataset(Xtr, Ytr, Xts, Yts)
    dataset.save()

if __name__ == '__main__':
    from mnist import load_mnist
    _generate_mnist_dataset()

    from cifar10 import load_cifar10
    _generate_cifar10_dataset()


    ### Update old dataset
    # dataset = MNISTDataset.load('mnist.pkl', 'quadboost/data/preprocessed/')
    # print(dataset.shape)
    # print(dataset._shape)
    # print(dataset.n_examples_train)
    # print(dataset._n_examples_train)
    # print(dataset.n_examples_test)
    # print(dataset._n_examples_test)
    # dataset.save()
