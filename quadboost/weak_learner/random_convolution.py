import numpy as np
from sklearn.linear_model import Ridge
import inspect
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import RandomAffine
import torchvision.transforms.functional as tf
from PIL.Image import BICUBIC
import warnings

import sys, os
sys.path.append(os.getcwd())

try:
    from quadboost.weak_learner import _WeakLearnerBase, _Cloner
    from quadboost.utils import timed, identity_func
except ModuleNotFoundError:
    from weak_learner import _WeakLearnerBase, _Cloner
    from utils import timed, identity_func


class Filters(_Cloner):
    """
    Class that encapsulates a number of filters and convolutes them with a dataset when 'apply' is called.

    Each call to an instance of the class will yield a new Filters object instanciated with the same arguments (see _Cloner). This allows to pick different filters from the weights_generator every new instance.
    """
    def __init__(self, n_filters, weights_generator, maxpool_shape=(3,3), activation=None):
        """
        Args:
            n_filters (int): Number of filters.
            weights_generator (iterable, optional): Iterable that yields filters weight.
            maxpool_shape (tuple of 2 or 3 integers or None, optional): Shape of the maxpool kernel layer. If None, no maxpool is done. If one of the dim is set to -1, the maxpool is done over all components of this dim.
            activation (Callable or None, optional): Activation function to apply which returns transformed data.
        """
        self.n_filters = n_filters
        if maxpool_shape:
            # Expanding maxpool_shape to account for n_transforms if needed
            self.maxpool_shape = (1, *maxpool_shape) if len(maxpool_shape) == 2 else maxpool_shape
        else:
            self.maxpool_shape = maxpool_shape

        self.activation = activation or identity_func

        self.weights, self.positions = [], []
        self._generate_filters(weights_generator, n_filters)

    def _generate_filters(self, weights_generator, n_filters):
        for _, (weight, position) in zip(range(n_filters), weights_generator):
            self.weights.append(torch.unsqueeze(weight, dim=1))

        if weights_generator.filters_shape_high is None:
            self.n_transforms_first = True
            self.weights = torch.cat(self.weights, dim=1)
            # self.weights.shape -> (n_transforms, n_filters, n_channels, height, width)
        else:
            self.n_transforms_first = False
            self.weights = [torch.squeeze(w, dim=1) for w in self.weights]
            # len(self.weights) -> n_filters
            # weight.shape -> (n_transforms, n_channels, height, width)

    def _send_weights_to_device(self, X):
        if isinstance(self.weights, torch.Tensor):
            self.weights = self.weights.to(device=X.device)
        else:
            self.weights = [w.to(device=X.device) for w in self.weights]

    def apply(self, X):
        self._send_weights_to_device(X)
        n_examples, *_ = X.shape

        output = []

        if self.n_transforms_first: # This is more efficient, but can't be done if filters are of different filter_shape (i.e. filters_shape_high is not None)
            for weights in self.weights:
                # weights.shape -> (n_filters, n_channels, height, width)
                output.append(torch.unsqueeze(F.conv2d(X, weights), dim=2))
                # output[0].shape -> (n_examples, n_filters, conv_height, conv_width)
            output = torch.cat(output, dim=2)

            # output.shape -> (n_examples, n_filters, n_transforms, conv_height, conv_width)
            if self.maxpool_shape:
                maxpool_shape = self._compute_maxpool_shape(output)
                output = F.max_pool3d(output, maxpool_shape, ceil_mode=True)

        else:
            for weights in self.weights:
                # weights.shape -> (n_transforms, n_channels, height, width)
                output_ = torch.unsqueeze(F.conv2d(X, weights), dim=1)
                # output_.shape -> (n_examples, 1, n_transforms, conv_height, conv_width)

                if self.maxpool_shape:
                    maxpool_shape = self._compute_maxpool_shape(output_)
                    output_ = F.max_pool3d(output_, maxpool_shape, ceil_mode=True)

                output.append(output_.reshape((n_examples, -1)))
            output = torch.cat(output, dim=1)

        self.activation(output)
        return output.reshape((n_examples, -1))

    def _compute_maxpool_shape(self, output):
        return tuple(ms if ms != -1 else output.shape[i+2] for i, ms in enumerate(self.maxpool_shape))  # Finding actual shape if -1 was used.


class LocalFilters(Filters):
    __doc__ = Filters.__doc__ + """\n
    The convolution is made locally only around the area the filter was drawn.
    """
    def __init__(self, *args, locality=5, **kwargs):
        """
        Args:
            locality (int, optional): Applies the filters locally around the place where the filter was taken from the picture in the filter bank. For example, for a filters_shape=(N,N) and a locality=L, the convolution will be made on a square of side N+2L centered around the original position. It will yield an array of side 2L+1. No padding is made in case the square exceeds the size of the examples.
        """
        super().__init__(*args, **kwargs)
        self.locality = locality

    def _generate_filters(self, weights_generator, n_filters):
        for _, (weight, position) in zip(range(n_filters), weights_generator):
            # self.weights.append(torch.unsqueeze(weight, dim=0))
            self.weights.append(weight)
            self.positions.append(position)

    def _send_weights_to_device(self, X):
        self.weights = [weight.to(device=X.device) for weight in self.weights]

    def apply(self, X):
        self._send_weights_to_device(X)

        n_examples, n_channels, height, width = X.shape
        random_feat = []
        for (i, j), weight in zip(self.positions, self.weights):
            i_min = max(i - self.locality, 0)
            j_min = max(j - self.locality, 0)
            i_max = min(i + weight.shape[-2] + self.locality, height)
            j_max = min(j + weight.shape[-1] + self.locality, width)

            output = F.conv2d(X[:,:,i_min:i_max, j_min:j_max], weight)
            # output.shape -> (n_examples, n_transforms, height, array)
            if self.maxpool_shape:
                output = torch.unsqueeze(output, dim=1)
                # output.shape -> (n_examples, 1, n_transforms, height, array)
                maxpool_shape = self._compute_maxpool_shape(output)
                output = F.max_pool3d(output, maxpool_shape, ceil_mode=True)

            random_feat.append(output.reshape(n_examples, -1))

        random_feat = torch.cat(random_feat, dim=1)
        self.activation(output)

        return random_feat


class WeightFromBankGenerator:
    """
    Infinite generator of weights.
    """
    def __init__(self, filter_bank, filters_shape=(5,5), filters_shape_high=None, margin=0, filter_processing=None, degrees=0, scale=None, shear=None, padding=2, n_transforms=1):
        """
        Args:
            filter_bank (tensor or array of shape (n_examples, n_channels, height, width)): Bank of images for filters to be drawn.
            filters_shape (sequence of 2 integers, optional): Shape of the filters.
            filters_shape_high (sequence of 2 integers or None, optional): If not None, the shape of the filters will be randomly drawn from a uniform distribution between filters_shape (inclusive) and filters_shape_high (exclusive).
            margin (int, optional): Number of pixels from the sides that are excluded from the pool of possible filters.
            filter_processing (callable or iterable of callables or None, optional): Callable or iterable of callables that execute (sequentially) some process on one weight and returns the result.
            degrees (int or tuple of int, optional): Maximum number of degrees the image drawn will be rotated before a filter in drawn. The actual degree is drawn from random. See torchvision.transforms.RandomAffine for more info.
            scale (tuple of float or None, optional): Scale factor the image drawn will be rescaled before a filter in drawn. The actual factor is drawn from random. See torchvision.transforms.RandomAffine for more info.
            shear (float or None, optional): Shear degree the image drawn will be sheared before a filter in drawn. The actual degree is drawn from random. See torchvision.transforms.RandomAffine for more info.
            n_transforms (int, optional): Number of filters made from the same image (at the same position) but with a different random transformation applied each time.
        """
        self.filter_bank = RandomConvolution.format_data(filter_bank)
        self.filters_shape = filters_shape
        self.filters_shape_high = filters_shape_high
        self.margin = margin
        if callable(filter_processing): filter_processing = [filter_processing]
        self.filter_processing = filter_processing or []

        self.n_examples, n_channels, self.bank_height, self.bank_width = self.filter_bank.shape
        self.i_max = self.bank_height - filters_shape[0]
        self.j_max = self.bank_width - filters_shape[1]

        self.degrees, self.scale, self.shear = degrees, scale, shear
        self.affine_transform = RandomAffine(degrees=degrees, scale=scale, shear=shear,
                                             resample=BICUBIC)
        self.padding = padding
        self.n_transforms = n_transforms

    def _draw_filter_shape(self):
        if not self.filters_shape_high:
            return self.filters_shape
        else:
            return (np.random.randint(self.filters_shape[0], self.filters_shape_high[0]),
                    np.random.randint(self.filters_shape[1], self.filters_shape_high[1]))

    def __iter__(self):
        while True:
            height, width = self._draw_filter_shape()
            i_max = self.bank_height - height
            j_max = self.bank_width - width
            yield self._draw_from_bank(height, width, i_max, j_max)

    def _draw_from_bank(self, height, width, i_max, j_max):
        # (i, j) is the top left corner where the filter position was taken
        i, j = (np.random.randint(self.margin, i_max-self.margin),
                np.random.randint(self.margin, j_max-self.margin))

        x = torch.tensor(self.filter_bank[np.random.randint(self.n_examples)], requires_grad=False).cpu()

        weight = []
        for _ in range(self.n_transforms):
            if self.degrees or self.scale or self.shear:
                x_transformed = self._transform_image(x)
                # plot_images([x_transformed.numpy().reshape(28,28)])
            else:
                x_transformed = x

            w = torch.tensor(x_transformed[:, i:i+height, j:j+width], requires_grad=False)
            for process in self.filter_processing:
                w = process(w)
            w = torch.unsqueeze(w, dim=0)
            weight.append(w)

        weight = torch.cat(weight, dim=0)
        return weight, (i, j)

    def _transform_image(self, x):
        # PIL images must be in format float 0-1 gray scale:
        min_x = torch.min(x)
        x_transformed = x - min_x
        max_x = torch.max(x)
        x_transformed /= max_x

        fillcolor = int(-min_x/max_x * 255) # Value to use to fill so that when reconverted to tensor, fill value is 0.
        self.affine_transform.fillcolor = fillcolor

        x_pil = tf.to_pil_image(x_transformed) # Conversion to PIl image looses quality because it is converted to 0-255 gray scale.
        x_pil = tf.crop(self.affine_transform(tf.pad(x_pil, self.padding, fill=fillcolor)),
                        self.padding, self.padding, self.bank_height, self.bank_width)
        x_transformed = tf.to_tensor(x_pil)
        x_transformed *= max_x
        x_transformed += min_x

        return x_transformed


class RandomConvolution(_WeakLearnerBase):
    """
    This weak learner is takes random filters and convolutes the dataset with them. It then applies a non-linearity on the resulting random features and uses an other weak regressor as final classifier.
    """
    def __init__(self, filters, encoder=None, weak_learner=Ridge):
        """
        Args:
            filters (callable): Callable that creates and returns a Filters object. A Filters object should define an 'apply' method that receives the examples (torch array of shape (n_examples, n_channels, width, height)) and outputs extracted features (torch array of shape (n_examples, n_features)).
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
        """
        Formats a data array to the right format accepted by this class, which is a torch.Tensor of shape (n_examples, n_channels, height, width).
        """
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


def center_weight(weight):
    mean = torch.mean(weight)
    weight -= mean
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
    (Xtr, Ytr), (Xts, Yts) = mnist.get_train_valid_test(valid=3000, center=True, reduce=True)
    # cifar = CIFAR10Dataset.load()
    # (Xtr, Ytr), (Xts, Yts) = cifar.get_train_test(center=True, reduce=True)
    # Xtr = torch.unsqueeze(torch.from_numpy(Xtr), dim=1)
    # Xts = torch.unsqueeze(torch.from_numpy(Xts), dim=1)

    encoder = OneHotEncoder(Ytr)

    # m = 60_000
    # bank = 1_000

    # print('CPU')
    # print('CUDA')
    # Xtr = Xtr[:m+bank].to(device='cuda:0')
    # Xts = Xts.to(device='cuda:0')
    scale = (0.9, 1.1)
    shear = 15
    filter_gen = WeightFromBankGenerator(filter_bank=Xtr,#[m:m+bank],
                                         filters_shape=(5,5),
                                         filters_shape_high=(16,16),
                                         margin=2,
                                         filter_processing=[center_weight],
                                         degrees=15,
                                         scale=scale if scale != 1 else None,
                                         shear=shear if shear != 0 else None,
                                         n_transforms=40,
                                         )
    filters = LocalFilters(n_filters=5,
                      maxpool_shape=(-1,-1,-1),
                    #   activation=torch.sigmoid,
                      weights_generator=filter_gen,
                      locality=3,
                      )
    weak_learner = Ridge
    # weak_learner = MulticlassDecisionStump

    print('Starting fit')
    wl = RandomConvolution(filters=filters,
                           weak_learner=weak_learner,
                           encoder=encoder,
                           )
    wl.fit(Xtr, Ytr)
    print('Train acc', wl.evaluate(Xtr, Ytr))
    print('Test acc', wl.evaluate(Xts, Yts))
    # wl.fit(Xtr[:m], Ytr[:m])
    # print('Train acc', wl.evaluate(Xtr[:m], Ytr[:m]))
    # print('Test acc', wl.evaluate(Xts[:m], Yts[:m]))


def plot_images(images, titles=None, block=True):
    import matplotlib.pyplot as plt
    from quadboost.utils import make_fig_axes

    fig, axes = make_fig_axes(len(images))

    vmax = min(np.max(np.abs(im)) for im in images)
    # vmax = 1
    if not titles:
        titles = (f'{n}: {vmax}' for n in range(len(images)))
    for im, title, ax in zip(images, titles, axes):
        # ax.imshow(im, cmap='gray_r')
        cax = ax.imshow(im, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_title(title)

    fig.colorbar(cax)
    # plt.get_current_fig_manager().window.showMaximized()
    plt.show(block=block)


if __name__ == '__main__':

    from quadboost.datasets import MNISTDataset, CIFAR10Dataset
    from quadboost.label_encoder import OneHotEncoder
    from quadboost.weak_learner import MulticlassDecisionStump

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    main()
