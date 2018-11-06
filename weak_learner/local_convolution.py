import numpy as np
from sklearn.linear_model import Ridge
import inspect
import torch
from torch import nn
from torch.nn import functional as F
import warnings

import sys, os
sys.path.append(os.getcwd())

from weak_learner import _WeakLearnerBase
from utils import timed


class Filters(nn.Module):
    def __init__(self, kernel_size=(5,5), maxpool=3):
        super().__init__()
        self.kernel_size = kernel_size
        in_ch, out_ch = 1, 1

        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size)

        for param in conv_filter.parameters():
            nn.init.normal_(param)
            param.requires_grad = False

        self.maxpool = nn.MaxPool2d((maxpool, maxpool))

    def forward(self, X):
        return self.maxpool(self.conv(X)).reshape((X.shape[0], 1, -1))


        # output = []
        # for conv_filter, (i, j) in zip(self.conv_filters, self.positions):
        #     i_min = i
        #     j_min = j
        #     i_max = i + self.kernel_size[0] + 2*self.locality
        #     j_max = j + self.kernel_size[1] + 2*self.locality
        #     local_X = padded_X[:,:,i_min:i_max, j_min:j_max]
        #     output.append(self.maxpool(conv_filter(local_X)))

        # output = torch.cat(output, dim=1).numpy()

        # return output.reshape((X.shape[0], -1))


class LocalConvolution(_WeakLearnerBase):
    """
    This weak learner is takes random filters and convolutes locally the dataset with them. It then applies a non-linearity on the resulting random features and fits a weak_learner on these new random features as final classifier.
    """
    def __init__(self, encoder=None, weak_learner=Ridge, n_filters=2, kernel_size=(5,5), init_filters='from_data', locality=0):
        """
        Args:
            encoder (LabelEncoder object, optional): Encoder to encode labels. If None, no encoding will be made before fitting.
            weak_learner (Class that defines the 'fit' and 'predict' methods or object instance that inherits from _WeakLearnerBase, optional): Regressor that will fit the data. Default is a Ridge regressor from scikit-learn.
            n_filters (int, optional): Number of filters.
            kernel_size ((int, int), optional): Size of the filters.
            init_filters (str, only 'from_data' for now): Choice of initialization of the filters weights. If 'from_data', the weights are taken as patches from the data, uniformly drawn.
            locality (int, optional): Applies the filters locally around the place where the patch was taken from the picture. Only applies if 'init_filters' is 'from_data'. For example, locality=2 will convolute the filter only ±2 pixels translated to yield 9 values.
        """
        self.encoder = encoder
        self.weak_learner = weak_learner()
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.locality = locality

        if init_filters == 'random':
            self.init_filters = None # Already random by default
        elif init_filters == 'from_data':
            self.init_filters = self.pick_from_dataset
        else:
            raise ValueError(f"'{init_filters} is an invalid init_filters option.")

    def fit(self, X, Y, W=None, **weak_learner_kwargs):
        """
        Args:
            X (Array of shape (n_examples, ...)): Examples to fit.
            Y (Array of shape (n_examples) or (n_examples, encoding_dim)): Labels of the examples. If an encoder is provided, Y should have shape (n_examples), otherwise it should have a shape (n_examples, encoding_dim).
            W (Array of shape (n_examples, encoding_dim), optional): Weights of the examples for each labels.
            weak_learner_kwargs: Keyword arguments needed to fit the weak learner.

        Returns self.
        """
        with torch.no_grad():
            if self.encoder is not None:
                Y, W = self.encoder.encode_labels(Y)
            X = self._format_X(X)

            self.filters = [Filter(self.kernel_size) for _ in range(self.n_filters)]


            random_feat = self._apply_filter(X)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore') # Ignore ill-defined matrices
            fit_sig = inspect.signature(self.weak_learner.fit)
            if 'W' in fit_sig.parameters:
                self.weak_learner.fit(random_feat, Y=Y, W=W, **weak_learner_kwargs)
            else:
                self.weak_learner.fit(random_feat, Y, **weak_learner_kwargs)

        del random_feat

        return self

    def _format_X(self, X):
        X = torch.from_numpy(X).float()
        if len(X.shape) == 3:
            X = torch.unsqueeze(X, dim=1)
        return X

    def _apply_filter(self, X):
        random_feat = []
        for filt in self.filters:
            random_feat.append(filt(X))
            self.init_filter(X=X, conv_filter=filt)

        return torch.cat(random_feat, dim=1).numpy().reshape((X.shape[0], -1))

    def predict(self, X):
        """
        Predicts the label of the sample X.

        Args:
            X (Array of shape (n_examples, ...)): Examples to predict.
        """
        with torch.no_grad():
            X = self._format_X(X)
            random_feat = self.filters(X)

        return self.weak_learner.predict(random_feat)

    def pick_from_dataset(self, X, conv_filter, **kwargs):
        """
        Assumes X is a torch tensor with shape (n_examples, n_channels, width, height).
        """
        n_examples = X.shape[0]
        i_max = X.shape[-2] - self.kernel_size[0] + 1
        j_max = X.shape[-1] - self.kernel_size[1] + 1

        x = X[np.random.randint(n_examples)]
        i = np.random.randint(i_max)
        j = np.random.randint(j_max)

        weights = torch.tensor(x[:, i:i+self.kernel_size[0], j:j+self.kernel_size[1]], requires_grad=False)
        conv_filter.weight = torch.nn.Parameter(torch.unsqueeze(weights, dim=1))
        conv_filter.weight.requires_grad = False

        self.positions.append
        for conv_filter in self.filters.conv_filters:
            x = X[np.random.randint(n_examples)]
            i = np.random.randint(i_max)
            j = np.random.randint(j_max)

            weights = torch.tensor(x[:, i:i+self.kernel_size[0], j:j+self.kernel_size[1]], requires_grad=False)
            conv_filter.weight = torch.nn.Parameter(torch.unsqueeze(weights, dim=1))
            conv_filter.weight.requires_grad = False

            self.filters.positions.append((i,j))

            del weights


@timed
def main():
    mnist = MNISTDataset.load()
    (Xtr, Ytr), (Xts, Yts) = mnist.get_train_test(center=True, reduce=True)

    encoder = OneHotEncoder(Ytr)

    m = 1_000

    # init_filters = 'random'
    init_filters='from_data'
    print('LocalConvolution')

    # wl = LocalConvolution(weak_learner=Ridge, n_filters=10, encoder=encoder, init_filters=init_filters, locality=5).fit(Xtr[:m], Ytr[:m])
    wl = LocalConvolution(weak_learner=MulticlassDecisionStump(), n_filters=3, encoder=encoder, init_filters=init_filters, locality=3).fit(Xtr[:m], Ytr[:m])
    print('Train acc', wl.evaluate(Xtr[:m], Ytr[:m]))
    print('Train acc', wl.evaluate(Xtr, Ytr))
    print('Test acc', wl.evaluate(Xts, Yts))


if __name__ == '__main__':
    from mnist_dataset import MNISTDataset
    from label_encoder import OneHotEncoder
    from weak_learner import MulticlassDecisionStump


    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    main()
