import numpy as np
from sklearn.linear_model import Ridge
import torch
from torch import nn
from torch.nn import functional as F
import warnings

import sys, os
sys.path.append(os.getcwd())

from weak_learner import _WeakLearnerBase
from utils import timed


class Filters(nn.Module):
    def __init__(self, n_filters, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(1, n_filters, kernel_size)
        self.maxpool = nn.MaxPool2d((3,3))

        for param in self.conv.parameters():
            nn.init.normal_(param)
            param.requires_grad = False

    def forward(self, x):
        return self.maxpool(self.conv(x))


class RandomFilters(_WeakLearnerBase):
    """
    This weak learner is takes random filters and convolutes the dataset with them. It then applies a non-linearity on the resulting random features and makes a Ridge regression as final classifier.
    """
    def __init__(self, encoder=None, n_filters=2, kernel_size=(5,5), init_filters='random'):
        """
        Args:
            encoder (LabelEncoder object, optional):
            n_filters (int, optional):
            kernel_size ((int, int), optional):
            init_filters (str, either 'random' or 'from_data'):
        """
        self.encoder = encoder
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        if init_filters == 'random':
            self.init_filters = None # Already random by default
        elif init_filters == 'from_data':
            self.init_filters = self.pick_from_dataset
        else:
            raise ValueError(f"'{init_filters} is an invalid init_filters option.")

    def fit(self, X, Y, W=None):
        """
        Args:
            X (Array of shape (n_examples, ...)): Examples to fit.
            Y (Array of shape (n_examples) or (n_examples, encoding_dim)): Labels of the examples. If an encoder is provided, Y should have shape (n_examples), otherwise it should have a shape (n_examples, encoding_dim).
            W (Array of shape (n_examples, encoding_dim), optional): Weights of the examples for each labels.

        Returns self.
        """
        if self.encoder is not None:
            Y, W = self.encoder.encode_labels(Y)
        X = self._format_X(X)

        self.filters = Filters(self.n_filters, self.kernel_size)
        if self.init_filters: self.init_filters(X=X)

        random_feat = self.filters(X).numpy().reshape((X.shape[0], -1))

        with warnings.catch_warnings():
            warnings.simplefilter('ignore') # Ignore ill-defined matrices
            self._classifier = Ridge().fit(random_feat, Y)

        return self

    def _format_X(self, X):
        X = torch.from_numpy(X).float()
        if len(X.shape) == 3:
            X = torch.unsqueeze(X, dim=1)
        return X

    def predict(self, X):
        """
        Predicts the label of the sample X.

        Args:
            X (Array of shape (n_examples, ...)): Examples to predict.
        """
        X = self._format_X(X)

        random_feat = self.filters(X).numpy().reshape((X.shape[0], -1))

        return self._classifier.predict(random_feat)

    def pick_from_dataset(self, X, **kwargs):
        """
        Assumes X is a torch tensor with shape (n_examples, n_channels, width, height).
        """
        weights = []
        n_examples = X.shape[0]
        i_max = X.shape[-2] - self.kernel_size[0] + 1
        j_max = X.shape[-1] - self.kernel_size[1] + 1

        for i in range(self.n_filters):
            x = X[np.random.randint(n_examples)]
            i = np.random.randint(i_max)
            j = np.random.randint(j_max)

            weights.append(x[:, i:i+self.kernel_size[0], j:j+self.kernel_size[1]])

        self.filters.conv.weight = torch.nn.Parameter(torch.unsqueeze(torch.cat(weights), dim=1))
        self.filters.conv.weight.requires_grad = False


@timed
def main():
    mnist = MNISTDataset.load()
    (Xtr, Ytr), (Xts, Yts) = mnist.get_train_test(center=True, reduce=True)

    encoder = OneHotEncoder(Ytr)

    m = 1_000
    Xtr, Ytr = Xtr[:m], Ytr[:m]

    # init_filters='from_data'
    # print(init_filters)
    # tr_acc = []
    # val_acc = []
    # for i in range(100):
    #     wl = RandomFilters(n_filters=3, encoder=encoder, init_filters=init_filters).fit(Xtr, Ytr)
    #     tr_acc.append(wl.evaluate(Xtr, Ytr))
    #     val_acc.append(wl.evaluate(Xts, Yts))

    #     print(f'mean train acc: {np.mean(tr_acc):4f}. mean valid acc: {np.mean(val_acc):4f} on {i} trials.', end='\r')
    # print('\n')

    init_filters='random'
    print(init_filters)
    tr_acc = []
    val_acc = []
    for i in range(100):
        wl = RandomFilters(n_filters=3, encoder=encoder, init_filters=init_filters).fit(Xtr, Ytr)
        tr_acc.append(wl.evaluate(Xtr, Ytr))
        val_acc.append(wl.evaluate(Xts, Yts))

        print(f'mean train acc: {np.mean(tr_acc):4f}. mean valid acc: {np.mean(val_acc):4f} on {i} trials.', end='\r')
    print('\n')


if __name__ == '__main__':
    from mnist_dataset import MNISTDataset
    from label_encoder import OneHotEncoder

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    main()
