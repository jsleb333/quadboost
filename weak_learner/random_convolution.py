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


class Filters(_WeakLearnerBase):
    def __init__(self, n_filters, maxpool=(3,3), device='cpu', filters_generator=None):
        # filter_normalization=None, filter_bank=None, filter_transform=None):
        """
        Args:
            n_filters (int): Number of filters.
            maxpool_shape ((int, int), optional): Shape of the maxpool kernel layer.
            device (str, either 'cpu' or 'cuda:X' where X is the number of the device, optional): Device on which the weights will be created.
            filters_generator (iterable, optional): Iterable that yields filters weight.

            filter_shape ((int, int)): Size of the filters.
            init_filters (str, either 'random', 'from_data' or 'from_bank'): Choice of initialization of the filters weights. If 'random', the weights are drawn from a normal distribution. If 'from_data', the weights are patches uniformly drawn from the data. If 'from_bank', the weights are patches uniformly drawn from the 'filter_bank'.
            filter_normalization (str or None, either 'c', 'r', 'n', 'cr' or 'cn'): If 'c', weights of the filters will be centered (i.e. of mean 0), if 'r', they will be reduced (i.e. of unit standard deviation), if 'n' they will be normalized (i.e. of unit euclidean norm) and 'cr' and 'cn' are combinations. If 'n' combined with 'r', the 'r' flag prevails.
            filter_bank (Array or None, optional): Bank of images for filters to be drawn. Only valid if 'init_filters' is set to 'from_bank'.
            filter_transform (callable or None, optional): Callable to apply on the filters. Should transform one filter (torch.Tensor) and return the new filter of the same size.
            If None, no transform is made.
        """
        self.maxpool_shape = maxpool
        self.n_filters = n_filters
        # self.filter_normalization = filter_normalization
        # self.filter_bank = filter_bank
        # self.filter_transform = filter_transform

        self.weights, self.positions = [], []
        for (weight, position), _ in zip(filters_generator, range(n_filters)):
            self.weights.append(torch.unsqueeze(weight, dim=0))
            self.positions.append(position)

        self.weights = torch.cat(self.weights, dim=0).to(device=device)

    def apply(self, X):
        output = F.conv2d(X, self.weights)
        output = F.max_pool2d(output, self.maxpool_shape, ceil_mode=True)
        return output.reshape((X.shape[0], -1))


class LocalFilters:
    """

    """
    def __init__(self, *args, locality=5, **kwargs):
        """
        Args:
            locality (int, optional):
        """
        super().__init__(self, *args, **kwargs)
        self.locality = locality

    def apply(self, X):
        output = F.conv2d(X, self.weights, groups=self.n_filters)


class WeightFromBankGenerator:
    """
    Infinite generator of weights.
    """
    def __init__(self, filter_bank, filter_shape=(5,5)):
        """
        Args:
            filter_shape ((int, int), optional): Shape of the filters.
            filter_bank (tensor or array of shape (n_examples, n_channels, height, width)):
        """
        self.filter_bank = filter_bank
        self.n_examples, n_channels, height, width = filter_bank.shape
        self.filter_shape = filter_shape
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

        return weight, (i, j)


class _RandomConvolution(_WeakLearnerBase):
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
        self.filters = filters
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
            X = self._format_data(X)

            random_feat = self.filters.apply(X)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore') # Ignore ill-defined matrices
            fit_sig = inspect.signature(self.weak_learner.fit)
            if 'W' in fit_sig.parameters:
                self.weak_learner.fit(random_feat, Y=Y, W=W, **weak_learner_kwargs)
            else:
                self.weak_learner.fit(random_feat, Y, **weak_learner_kwargs)

        return self

    def _format_data(self, data):
        if type(data) is np.ndarray:
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
        with torch.no_grad():
            X = self._format_data(X)
            random_feat = self.filters.apply(X)

        return self.weak_learner.predict(random_feat)

    def _draw_from_images(self, X):
        n_examples = X.shape[0]
        i_max = X.shape[-2] - self.filter_shape[0]
        j_max = X.shape[-1] - self.filter_shape[1]

        x = X[np.random.randint(n_examples)]
        i = np.random.randint(i_max)
        j = np.random.randint(j_max)
        position = (i, j)

        weight = torch.tensor(x[:, i:i+self.filter_shape[0], j:j+self.filter_shape[1]], requires_grad=False)
        if self.filter_transform:
            weight = self.filter_transform(weight)
        self._normalize_weight(weight)

        return weight, position

    def _normalize_weight(self, weight):
        if 'c' in self.filter_normalization:
            weight -= torch.mean(weight)
        if 'n' in self.filter_normalization:
            weight /= torch.norm(weight, p=2)
        if 'r' in self.filter_normalization:
            weight /= torch.std(weight)


class RandomCompleteConvolution(_RandomConvolution):
    pass

class RandomLocalConvolution(_RandomConvolution):
    def __init__(self, *args, locality=5, init_filters='from_data', **kwargs):
        """
        Args:
            encoder (LabelEncoder object, optional): Encoder to encode labels. If None, no encoding will be made before fitting.
            weak_learner (Class that defines the 'fit' and 'predict' methods or object instance that inherits from _WeakLearnerBase, optional): Regressor that will fit the data. Default is a Ridge regressor from scikit-learn.
            n_filters (int, optional): Number of filters.
            filter_shape ((int, int), optional): Size of the filters.
            init_filters (str, either 'from_data' or 'from_bank'): Choice of initialization of the filters weights. If 'from_data', the weights are patches uniformly drawn from the data. If 'from_bank', the weights are patches uniformly drawn from the 'filter_bank'.
            filter_normalization (str or None, either 'c', 'r', 'n', 'cr' or 'cn'): If 'c', weights of the filters will be centered (i.e. of mean 0), if 'r', they will be reduced (i.e. of unit standard deviation), if 'n' they will be normalized (i.e. of unit euclidean norm) and 'cr' and 'cn' are combinations. If 'n' combined with 'r', the 'r' flag prevails.
            filter_bank (Array or None, optional): Bank of images for filters to be drawn. Only valid if 'init_filters' is set to 'from_bank'.
            locality (int, optional): Applies the filters locally around the place where the patch was taken from the picture. Only applies if 'init_filters' is 'from_data'. For example, locality=2 will convolute the filter only ±2 pixels translated to yield 9 values.
        """
        super().__init__(*args, **kwargs)
        self.locality = locality

        if init_filters == 'random':
            raise ValueError(f'Invalid init_filters {init_filters} argument for RandomLocalConvolution.')

    def _generate_filters(self, X):
        return [Filters(1, self.filter_shape, self.maxpool_size) for _ in range(self.n_filters)]

    def _apply_filters(self, X):
        height, width = X.shape[-2:]
        random_feat = []
        for filt in self.filters:
            i, j = filt.position
            i_min = max(i - self.locality, 0)
            j_min = max(j - self.locality, 0)
            i_max = min(i + self.filter_shape[0] + self.locality, height)
            j_max = min(j + self.filter_shape[1] + self.locality, width)
            local_X = X[:,:,i_min:i_max, j_min:j_max]
            random_feat.append(filt(local_X))

        return torch.cat(random_feat, dim=1).numpy()

    def init_from_normal(self, **kwargs):
        raise RuntimeError('This should not have happened. Try another init_filters option.')

    def init_from_images(self, *, filter_bank, **kwargs):
        """
        Assumes X is a torch tensor with shape (n_examples, n_channels, width, height).
        """
        formatted_bank = self._format_data(filter_bank)
        for conv_filter in self.filters:
            weights, position = self._draw_from_images(formatted_bank)
            conv_filter.weight = torch.unsqueeze(weights, dim=1)
            conv_filter.position = position


def transform_filter(degrees=0, scale=None, shear=None):
    def transform(weight):
        tmp_weight = to_pil_image(weight)
        affine = RandomAffine(degrees=degrees, scale=scale, shear=shear)
        tmp_weight = affine(tmp_weight)
        return to_tensor(tmp_weight)
    return transform

@timed
def main():
    mnist = MNISTDataset.load()
    (Xtr, Ytr), (Xts, Yts) = mnist.get_train_test(center=True, reduce=True)
    Xtr, Xts = torch.from_numpy(Xtr), torch.from_numpy(Xts)
    Xtr = torch.unsqueeze(Xtr, dim=1)
    Xts = torch.unsqueeze(Xts, dim=1)
    print(Xtr.shape)

    encoder = OneHotEncoder(Ytr)

    m = 1_000

    # init_filters = 'random'
    # init_filters = 'from_data'
    filter_gen = WeightFromBankGenerator(filter_bank=Xtr[m:m+1000], filter_shape=(5,5))
    filters = Filters(n_filters=3,
                      maxpool=(3,3),
                      device='cpu',
                      filters_generator=filter_gen)
    weak_learner = Ridge
    # weak_learner = MulticlassDecisionStump

    print('Complete')
    wl = RandomCompleteConvolution(filters=filters,
                                   weak_learner=weak_learner,
                                   encoder=encoder,
                                   ).fit(Xtr[:m], Ytr[:m])
    # print('Local')
    # wl = RandomLocalConvolution(n_filters=3,
    #                             weak_learner=weak_learner,
    #                             encoder=encoder,
    #                             filter_shape=(5,5),
    #                             maxpool_size=(2,2),
    #                             init_filters=init_filters,
    #                             filter_normalization='c',
    #                             filter_bank=Xtr[m:m+1000],
    #                             filter_transform=transform_filter(15, (0.9,1.1)),
    #                             locality=3,
    #                             ).fit(Xtr[:m], Ytr[:m])

    print('Train acc', wl.evaluate(Xtr[:m], Ytr[:m]))
    print('All train acc', wl.evaluate(Xtr, Ytr))
    print('Test acc', wl.evaluate(Xts, Yts))


if __name__ == '__main__':
    from mnist_dataset import MNISTDataset
    from label_encoder import OneHotEncoder
    from weak_learner import MulticlassDecisionStump
    from torchvision.transforms.functional import to_pil_image, to_tensor
    from torchvision.transforms import RandomAffine

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    main()
