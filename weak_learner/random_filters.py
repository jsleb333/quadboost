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
    def __init__(self, encoder=None, n_filters=2, kernel_size=(5,5), init_filters='random', filter_normalization=None):
        """
        Args:
            encoder (LabelEncoder object, optional): Encoder to encode labels. If None, no encoding will be made before fitting.
            n_filters (int, optional): Number of filters.
            kernel_size ((int, int), optional): Size of the filters.
            init_filters (str, either 'random' or 'from_data'): Choice of initialization of the filters weights. If 'random', the weights are drawn from a normal distribution. If 'from_data', the weights are taken as patches from the data, uniformly drawn.
            filter_normalization (str or None, either 'c', 'r' or 'cr'): If 'c', weights of the filters will be centered (i.e. of mean 0), if 'r', they will be reduces (i.e. of euclidean norm 1) and if 'cr', they will be both.
        """
        self.encoder = encoder
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.filter_normalization = filter_normalization

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
        if self.init_filters: self.init_filters(X=X, filter_normalization=self.filter_normalization)

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

    def pick_from_dataset(self, X, filter_normalization=None, **kwargs):
        """
        Assumes X is a torch tensor with shape (n_examples, n_channels, width, height).
        """
        weights = []
        n_examples = X.shape[0]
        i_max = X.shape[-2] - self.kernel_size[0] + 1
        j_max = X.shape[-1] - self.kernel_size[1] + 1

        for _ in range(self.n_filters):
            x = X[np.random.randint(n_examples)]
            i = np.random.randint(i_max)
            j = np.random.randint(j_max)

            weight = torch.tensor(x[:, i:i+self.kernel_size[0], j:j+self.kernel_size[1]])
            if filter_normalization:
                if 'c' in filter_normalization:
                    weight -= torch.sum(weight)
                if 'r' in filter_normalization:
                    weight /= torch.norm(weight, p=2)
            print(weight, torch.sum(weight))

            weights.append(weight)

        self.filters.conv.weight = torch.nn.Parameter(torch.unsqueeze(torch.cat(weights), dim=1))
        self.filters.conv.weight.requires_grad = False


@timed
def main():
    mnist = MNISTDataset.load()
    (Xtr, Ytr), (Xts, Yts) = mnist.get_train_test(center=True, reduce=True)

    encoder = OneHotEncoder(Ytr)

    m = 1_000
    Xtr, Ytr = Xtr[:m], Ytr[:m]

    # init_filters = 'random'
    init_filters='from_data'
    print('RandomFilters')
    
    wl = RandomFilters(n_filters=3, encoder=encoder, init_filters=init_filters, filter_normalization=None).fit(Xtr, Ytr)
    print('Train acc', wl.evaluate(Xtr, Ytr))
    print('Test acc', wl.evaluate(Xts, Yts))


if __name__ == '__main__':
    from mnist_dataset import MNISTDataset
    from label_encoder import OneHotEncoder

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    main()
