import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import conv2d
from mnist_dataset import MNISTDataset
from utils import *
from label_encoder import OneHotEncoder


class NoListModule(nn.Module):
    def __init__(self, n_filters=10, kernel_size=(5,5)):
        super().__init__()
        self.conv_filters = nn.Conv2d(1, n_filters, kernel_size)

    def forward(self, X):
        return self.conv_filters(X)


class InternalListModule(nn.Module):
    def __init__(self, n_filters=10, kernel_size=(5,5)):
        super().__init__()
        self.conv_filters = [nn.Conv2d(1, 1, kernel_size) for _ in range(n_filters)]
        for i, conv in enumerate(self.conv_filters):
            setattr(self, f'conv{i}', conv)

    def forward(self, X):
        return torch.cat([conv(X) for conv in self.conv_filters], dim=1)


class ExternalListModule(nn.Module):
    def __init__(self, n_filters=10, kernel_size=(5,5)):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size)

    def forward(self, X):
        return self.conv(X)

class FunctionalConv:
    def __init__(self, n_filters, kernel_size=(5,5)):
        shape = (n_filters, 1, *kernel_size)
        self.weight = torch.rand(shape)

    def cuda(self):
        self.weight = self.weight.cuda()

    def __call__(self, X):
        return conv2d(X, self.weight)


class NChannelsConv(FunctionalConv):
    def __init__(self, n_filters, kernel_size=(5,5)):
        shape = (1, n_filters, *kernel_size)
        self.weight = torch.rand(shape)


def no_list(X, n_filters=10, kernel_size=(5,5), use_gpu=False):
    m = NoListModule(n_filters, kernel_size)
    if use_gpu: m.cuda()
    return m(X)


def internal_list(X, n_filters=10, kernel_size=(5,5), use_gpu=False):
    m = InternalListModule(n_filters, kernel_size)
    if use_gpu: m.cuda()
    return m(X)


def external_list(X, n_filters=10, kernel_size=(5,5), use_gpu=False):
    ms = [ExternalListModule(n_filters, kernel_size) for _ in range(n_filters)]
    if use_gpu:
        for m in ms:
            m.cuda()
    return torch.cat([m(X) for m in ms], dim=1)


def functional_conv(X, n_filters=10, kernel_size=(5,5), use_gpu=False):
    m = FunctionalConv(n_filters, kernel_size)
    if use_gpu: m.cuda()
    return m(X)


def n_channels_conv(X, n_filters=10, kernel_size=(5,5), use_gpu=False):
    m = NChannelsConv(n_filters, kernel_size)
    if use_gpu: m.cuda()
    new_X = []
    for i, j in np.random.randint(1,18, (n_filters, 2)):
        new_X.append(X[:, :, i:i+2*kernel_size[0], j:j+2*kernel_size[1]])
        print(i, j)
    X = torch.cat(new_X, dim=1)
    print(X.shape)
    return m(X)


def timeit(func, *args, N=10):
    tot = 0
    for i in range(N):
        t = time()
        func(*args).cpu()
        tot += time() - t
    return tot/N


def main():
    mnist = MNISTDataset.load()
    (Xtr, Ytr), (Xts, Yts) = mnist.get_train_test(center=True, reduce=True)

    n_filters = 10
    kernel_size = (5,5)
    use_gpu = False
    # use_gpu = True

    with torch.no_grad():
        X = torch.unsqueeze(torch.from_numpy(Xtr[:8_000]), dim=1).float()
        if use_gpu:
            X = timed(X.cuda)()

        no_list(X, n_filters, kernel_size, use_gpu)
        internal_list(X, n_filters, kernel_size, use_gpu)
        external_list(X, n_filters, kernel_size, use_gpu)
        functional_conv(X, n_filters, kernel_size, use_gpu)
        n_channels_conv(X, n_filters, kernel_size, use_gpu)

        print('no list time:', timeit(no_list, X, n_filters, kernel_size, use_gpu))
        # print('internal list time:', timeit(internal_list, X, n_filters, kernel_size, use_gpu))
        # print('external list time:', timeit(external_list, X, n_filters, kernel_size, use_gpu))
        print('functional conv time:', timeit(functional_conv, X, n_filters, kernel_size, use_gpu))
        print('n channels conv time:', timeit(n_channels_conv, X, n_filters, kernel_size, use_gpu))


if __name__ == '__main__':
    from time import sleep, time
    main()
