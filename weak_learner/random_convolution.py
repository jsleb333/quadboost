import numpy as np
from sklearn.linear_model import Ridge
import inspect
import torch
from torch import nn
from torch.nn import functional as F
import warnings

import sys, os
sys.path.append(os.getcwd())

from weak_learner import _WeakLearnerBase, _Cloner
from utils import timed


class Filters(_Cloner):
    """
    """
    def __init__(self, n_filters, filters_generator, maxpool_shape=(3,3)):
        """
        Args:
            n_filters (int): Number of filters.
            filters_generator (iterable, optional): Iterable that yields filters weight.
            maxpool_shape ((int, int), optional): Shape of the maxpool kernel layer.
        """
        self.n_filters = n_filters
        self.filters_generator = filters_generator or torch.rand()
        self.maxpool_shape = maxpool_shape
        # self.filter_normalization = filter_normalization
        # self.filter_bank = filter_bank
        # self.filter_transform = filter_transform

        self.weights, self.positions = [], []
        for (weight, position), _ in zip(filters_generator, range(n_filters)):
            self.weights.append(torch.unsqueeze(weight, dim=0))
            self.positions.append(position)

        self.weights = torch.cat(self.weights, dim=0)#.to(device=device)

    def _send_weights_to_device(self, X):
        self.weights = self.weights.to(device=X.device)

    def apply(self, X):
        self._send_weights_to_device(X)

        output = F.conv2d(X, self.weights)
        output = F.max_pool2d(output, self.maxpool_shape, ceil_mode=True)
        return output.reshape((X.shape[0], -1))


class LocalFilters(Filters):
    """
    """
    def __init__(self, *args, locality=5, **kwargs):
        """
        Args:
            locality (int, optional): Applies the filters locally around the place where the filter was taken from the picture in the filter bank. For example, for a filter_shape=(N,N) and a locality=L, the convolution will be made on a square of side N+2L centered around the original position. It will yield an array of side 2L+1. No padding is made in case the square exceeds the size of the examples.
        """
        super().__init__(*args, **kwargs)
        self.locality = locality

    def apply(self, X):
        self._send_weights_to_device(X)

        n_examples, n_channels, height, width = X.shape
        random_feat = []
        for (i, j), weight in zip(self.positions, self.weights):
            i_min = max(i - self.locality, 0)
            j_min = max(j - self.locality, 0)
            i_max = min(i + self.weights.shape[2] + self.locality, height)
            j_max = min(j + self.weights.shape[3] + self.locality, width)

            output = F.conv2d(X[:,:,i_min:i_max, j_min:j_max], torch.unsqueeze(weight, dim=0))
            output = F.max_pool2d(output, self.maxpool_shape, ceil_mode=True)
            random_feat.append(output.reshape(n_examples, -1))

        random_feat = torch.cat(random_feat, dim=1)

        return random_feat


class WeightFromBankGenerator:
    """
    Infinite generator of weights.
    """
    def __init__(self, filter_bank, filter_shape=(5,5), filter_processing=None):
        """
        Args:
            filter_shape ((int, int), optional): Shape of the filters.
            filter_bank (tensor or array of shape (n_examples, n_channels, height, width)): Bank of images for filters to be drawn. Only valid if 'init_filters' is set to 'from_bank'.
            filter_processing (callable or iterable of callables or None, optional): Callable or iterable of callable that processes one weight and returns the result.
        """
        self.filter_bank = filter_bank
        self.filter_shape = filter_shape
        if callable(filter_processing): filter_processing = [filter_processing]
        self.filter_processing = filter_processing or []

        self.n_examples, n_channels, height, width = filter_bank.shape
        self.i_max = height - filter_shape[0]
        self.j_max = width - filter_shape[1]

    def __iter__(self):
        while True:
            yield self._draw_from_bank()

    def _draw_from_bank(self):
        # (i, j) is the top left corner where the filter position was taken
        i, j = np.random.randint(self.i_max), np.random.randint(self.j_max)

        x = self.filter_bank[np.random.randint(self.n_examples)]
        weight = torch.tensor(x[:, i:i+self.filter_shape[0], j:j+self.filter_shape[1]], requires_grad=False)
        for process in self.filter_processing:
            weight = process(weight)

        return weight, (i, j)


class RandomConvolution(_WeakLearnerBase):
    """
    This weak learner is takes random filters and convolutes the dataset with them. It then applies a non-linearity on the resulting random features and uses an other weak regressor as final classifier.
    """
    def __init__(self, filters, encoder=None, weak_learner=Ridge):
        """
        Args:
            filters (Uninstanciated class that defines __call__)): Callable that returns that receives the examples (torch array of shape (n_examples, n_channels, width, height)) and outputs extracted features (torch array of shape (n_examples, n_features)).
            encoder (LabelEncoder object, optional): Encoder to encode labels. If None, no encoding will be made before fitting.
            weak_learner (Callable that returns a new object that defines the 'fit' and 'predict' methods, such as object inheriting from _WeakLearnerBase, optional): Regressor that will fit the data. Default is a Ridge regressor from scikit-learn.
        """
        self.filters = filters()
        self.encoder = encoder
        self.weak_learner = weak_learner()

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
            X = self.format_data(X)

            random_feat = self.filters.apply(X)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore') # Ignore ill-defined matrices
            fit_sig = inspect.signature(self.weak_learner.fit)
            if 'W' in fit_sig.parameters:
                self.weak_learner.fit(random_feat, Y=Y, W=W, **weak_learner_kwargs)
            else:
                self.weak_learner.fit(random_feat, Y, **weak_learner_kwargs)

        return self

    @staticmethod
    def format_data(data):
        if type(data) is np.ndarray:
            data = torch.from_numpy(data).float()
        if len(data.shape) == 3:
            data = torch.unsqueeze(data, dim=1)
        return data

    def predict(self, X):
        """
        Predicts the label of the sample X.

        Args:
            X (Array of shape (n_examples, n_channels, height, width)): Examples to predict.
        """
        with torch.no_grad():
            X = self.format_data(X)
            random_feat = self.filters.apply(X)

        return self.weak_learner.predict(random_feat)


def transform_filter(degrees=0, scale=None, shear=None):
    affine = RandomAffine(degrees=degrees, scale=scale, shear=shear)
    def transform(weight):
        tmp_weight = to_pil_image(weight)
        tmp_weight = affine(tmp_weight)
        return to_tensor(tmp_weight)
    return transform


def center_weight(weight):
    weight -= torch.mean(weight)
    return weight

def normalize_weight(weight):
    weight /= torch.norm(weight, p=2)
    return weight

def reduce_weight(weight):
    weight /= torch.std(weight)
    return weight


@timed
def main():
    mnist = MNISTDataset.load()
    (Xtr, Ytr), (Xts, Yts) = mnist.get_train_test(center=True, reduce=True)
    Xtr = torch.unsqueeze(torch.from_numpy(Xtr), dim=1)
    Xts = torch.unsqueeze(torch.from_numpy(Xts), dim=1)

    encoder = OneHotEncoder(Ytr)

    m = 1_000
    bank = 1000

    # print('CPU')
    # print('CUDA')
    # Xtr = Xtr[:m+bank].to(device='cuda:0')
    # Xts = Xts.to(device='cuda:0')

    random_transform = transform_filter(15,(.9,1.1))
    filter_gen = WeightFromBankGenerator(filter_bank=Xtr[m:m+bank],
                                         filter_shape=(5,5),
                                         filter_processing=[center_weight])
    filters = Filters(n_filters=10,
                      maxpool_shape=(3,3),
                      filters_generator=filter_gen)
    weak_learner = Ridge
    # weak_learner = MulticlassDecisionStump

    wl = RandomConvolution(filters=filters,
                           weak_learner=weak_learner,
                           encoder=encoder,
                           )
    wl.fit(Xtr[:m], Ytr[:m])
    print('Train acc', wl.evaluate(Xtr[:m], Ytr[:m]))
    print('Test acc', wl.evaluate(Xts, Yts))


if __name__ == '__main__':
    from mnist_dataset import MNISTDataset
    from label_encoder import OneHotEncoder
    from weak_learner import MulticlassDecisionStump
    from torchvision.transforms.functional import to_pil_image, to_tensor
    from torchvision.transforms import RandomAffine

    # seed = 42
    # torch.manual_seed(seed)
    # np.random.seed(seed)

    main()
