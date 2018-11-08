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
    def __init__(self, encoder=None, n_filters=2, kernel_size=(5,5), init_filters='random', filter_normalization=None, filter_bank=None):
        """
        Args:
            encoder (LabelEncoder object, optional): Encoder to encode labels. If None, no encoding will be made before fitting.
            n_filters (int, optional): Number of filters.
            kernel_size ((int, int), optional): Size of the filters.
            init_filters (str, either 'random' or 'from_data'): Choice of initialization of the filters weights. If 'random', the weights are drawn from a normal distribution. If 'from_data', the weights are taken as patches from the data, uniformly drawn.
            filter_normalization (str or None, either 'c', 'r', 'n', 'cr' or 'cn'): If 'c', weights of the filters will be centered (i.e. of mean 0), if 'r', they will be reduced (i.e. of unit standard deviation), if 'n' they will be normalized (i.e. of unit euclidean norm) and 'cr' and 'cn' are combinations. If 'n' combined with 'r', the 'r' flag prevails.
            filter_bank (Array or None, optional): Bank of images for filters to be drawn. Only valid if 'init_filters' is set to 'from_bank'.
        """
        self.encoder = encoder
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.filter_normalization = filter_normalization or ''
        self.filter_bank = filter_bank

        if init_filters == 'random':
            self.init_filters = None # Already random by default
        elif init_filters == 'from_data' or init_filters == 'from_bank':
            self.init_filters = self.init_from_images
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
        with torch.no_grad():
            if self.encoder is not None:
                Y, W = self.encoder.encode_labels(Y)
            X = self._format_data(X)

            self.filters = Filters(self.n_filters, self.kernel_size)
            if self.init_filters:
                filter_bank = self.filter_bank or X
                self.init_filters(filter_bank, filter_normalization=self.filter_normalization)

            random_feat = self.filters(X).numpy().reshape((X.shape[0], -1))

        with warnings.catch_warnings():
            warnings.simplefilter('ignore') # Ignore ill-defined matrices
            self._classifier = Ridge().fit(random_feat, Y)

        return self

    def _format_data(self, data):
        data = torch.from_numpy(data).float()
        if len(data.shape) == 3:
            data = torch.unsqueeze(data, dim=1)
        return data

    def predict(self, X):
        """
        Predicts the label of the sample X.

        Args:
            X (Array of shape (n_examples, ...)): Examples to predict.
        """
        X = self._format_data(X)

        random_feat = self.filters(X).numpy().reshape((X.shape[0], -1))

        return self._classifier.predict(random_feat)

    def init_from_images(self, filter_bank, filter_normalization='', **kwargs):
        """
        Assumes X is a torch tensor with shape (n_examples, n_channels, width, height).
        """
        weights = []
        for _ in range(self.n_filters):
            weight = self._draw_from_images(self._format_data(filter_bank))
            if 'c' in filter_normalization:
                weight -= torch.mean(weight)
            if 'n' in filter_normalization:
                weight /= torch.norm(weight, p=2)
            if 'r' in filter_normalization:
                weight /= torch.std(weight)

            weights.append(weight)

        self.filters.conv.weight = torch.nn.Parameter(torch.unsqueeze(torch.cat(weights), dim=1))
        self.filters.conv.weight.requires_grad = False

    def _draw_from_images(self, X):
        n_examples = X.shape[0]
        i_max = X.shape[-2] - self.kernel_size[0]
        j_max = X.shape[-1] - self.kernel_size[1]

        x = X[np.random.randint(n_examples)]
        i = np.random.randint(i_max)
        j = np.random.randint(j_max)

        weights = torch.tensor(x[:, i:i+self.kernel_size[0], j:j+self.kernel_size[1]], requires_grad=False)

        return weights


@timed
def main():
    mnist = MNISTDataset.load()
    (Xtr, Ytr), (Xts, Yts) = mnist.get_train_test(center=True, reduce=True)

    encoder = OneHotEncoder(Ytr)

    m = 1_000

    # init_filters = 'random'
    init_filters='from_data'
    print('RandomFilters')

    wl = RandomFilters(n_filters=3, encoder=encoder, init_filters=init_filters, filter_normalization='c').fit(Xtr[:m], Ytr[:m])
    print('Train acc', wl.evaluate(Xtr[:m], Ytr[:m]))
    print('All train acc', wl.evaluate(Xtr, Ytr))
    print('Test acc', wl.evaluate(Xts, Yts))


if __name__ == '__main__':
    from mnist_dataset import MNISTDataset
    from label_encoder import OneHotEncoder

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    main()
