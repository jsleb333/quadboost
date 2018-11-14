import numpy as np
import torch
import torch.nn as nn
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
    # use_gpu = False
    use_gpu = True

    with torch.no_grad():
        X = torch.unsqueeze(torch.from_numpy(Xtr[:8_000]), dim=1).float()
        if use_gpu:
            X = timed(X.cuda)()

        no_list(X, n_filters, kernel_size, use_gpu)
        internal_list(X, n_filters, kernel_size, use_gpu)
        external_list(X, n_filters, kernel_size, use_gpu)

        print('no list time:', timeit(no_list, X, n_filters, kernel_size, use_gpu))
        print('internal list time:', timeit(internal_list, X, n_filters, kernel_size, use_gpu))
        print('external list time:', timeit(external_list, X, n_filters, kernel_size, use_gpu))


if __name__ == '__main__':
    from time import sleep, time
    main()
