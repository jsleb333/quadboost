import numpy as np
from sklearn.linear_model import Ridge
import torch
from torch import nn
from torch.nn import functional as F

import sys, os
sys.path.append(os.getcwd())

from weak_learner import WeakLearnerBase
from utils import timed


class Filters(nn.Module):
    def __init__(self, n_filters, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(1, n_filters, kernel_size)

        for param in self.conv.parameters():
            nn.init.normal_(param)
            param.requires_grad = False

    def forward(self, x):
        return F.relu(self.conv(x))


class RandomFilters(WeakLearnerBase):
    """
    """
    def __init__(self, encoder=None, n_filters=2, kernel_size=(5,5)):
        self.encoder = encoder
        self.filters = Filters(n_filters, kernel_size)

    def fit(self, X, Y, W=None, **kwargs):
        """
        """
        if self.encoder is not None:
            Y, W = self.encoder.encode_labels(Y)

        tensor_shape = (X.shape[0], 1, *X.shape[1:]) # Insert the number of input channels
        X_tensor = torch.from_numpy(X).reshape(tensor_shape).float()
        random_feat = self.filters(X_tensor).numpy().reshape((X.shape[0], -1))

        self._classifier = Ridge().fit(random_feat, Y)

        return self

    def predict(self, X, **kwargs):
        tensor_shape = (X.shape[0], 1, *X.shape[1:]) # Insert the number of input channels
        X = torch.from_numpy(X).reshape(tensor_shape).float()
        random_feat = self.filters(X).numpy().reshape((X.shape[0], -1))

        return self._classifier.predict(random_feat)


@timed
def main():
    mnist = MNISTDataset.load()
    (Xtr, Ytr), (Xts, Yts) = mnist.get_train_test(center=True, reduce=True)

    encoder = OneHotEncoder(Ytr)

    m = 60_000
    Xtr, Ytr = Xtr[:m], Ytr[:m]

    wl = RandomFilters(n_filters=20, encoder=encoder).fit(Xtr, Ytr)
    print(wl.evaluate(Xtr, Ytr))
    print(wl.evaluate(Xts, Yts))


if __name__ == '__main__':
    from mnist_dataset import MNISTDataset
    from label_encoder import OneHotEncoder

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    main()
