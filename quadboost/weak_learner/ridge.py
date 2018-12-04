import numpy as np
from sklearn.linear_model import Ridge

import sys, os
sys.path.append(os.getcwd())

try:
    from quadboost.weak_learner import _WeakLearnerBase
    from quadboost.utils import timed
except ModuleNotFoundError:
    from weak_learner import _WeakLearnerBase
    from utils import timed


class WLRidge(_WeakLearnerBase, Ridge):
    """
    Confidence rated Ridge classification based on a Ridge regression.
    Inherits from Ridge of the scikit-learn package.
    In this implementation, the method 'fit' does not support encoding weights of the QuadBoost algorithm.
    """
    def __init__(self, alpha=1, encoder=None, fit_intercept=False, **kwargs):
        super().__init__(alpha=alpha, fit_intercept=fit_intercept, encoder=encoder, **kwargs)

    def fit(self, X, Y, W=None, **kwargs):
        """
        NB: this method supports encoding weights of the QuadBoost algorithm by multiplying Y by the square root of the weights W. This should be taken into account for continous predictions.
        """
        X = X.reshape((X.shape[0], -1))
        if self.encoder != None:
            Y, W = self.encoder.encode_labels(Y)
        if W is not None:
            Y *= np.sqrt(W)
        return super(Ridge, self).fit(X, Y, **kwargs)

    def predict(self, X, **kwargs):
        X = X.reshape((X.shape[0], -1))
        return super(Ridge, self).predict(X, **kwargs)


class WLThresholdedRidge(_WeakLearnerBase, Ridge):
    """
    Ridge classification based on a ternary vote (1, 0, -1) of a Ridge regression based on a threshold. For a threshold of 0, it is equivalent to take the sign of the prediction.
    Inherits from Ridge of the scikit-learn package.
    In this implementation, the method 'fit' does not support encoding weights of the QuadBoost algorithm.
    """
    def __init__(self, alpha=1, encoder=None, threshold=0.5, fit_intercept=False, **kwargs):
        super().__init__(alpha=alpha, fit_intercept=fit_intercept, encoder=encoder, **kwargs)
        self.threshold = threshold

    def fit(self, X, Y, W=None, **kwargs):
        """
        Note: this method does not support encoding weights of the QuadBoost algorithm.
        """
        X = X.reshape((X.shape[0], -1))
        if self.encoder != None:
            Y, W = self.encoder.encode_labels(Y)
        return super(Ridge, self).fit(X, Y, **kwargs)

    def predict(self, X, **kwargs):
        X = X.reshape((X.shape[0], -1))
        Y = super(Ridge, self).predict(X, **kwargs)
        Y = np.where(Y >= self.threshold, 1.0, Y)
        Y = np.where(np.logical_and(Y < self.threshold, Y > -self.threshold), 0, Y)
        Y = np.where(Y < -self.threshold, -1, Y)
        return Y


@timed
def main():
    mnist = MNISTDataset.load()
    (Xtr, Ytr), (Xts, Yts) = mnist.get_train_test(center=True, reduce=False)

    encoder = OneHotEncoder(Ytr)

    m = 6_000
    Xtr, Ytr = Xtr[:m], Ytr[:m]

    # wl = WLThresholdedRidge(encoder=encoder)
    wl = WLRidge(encoder=encoder)
    wl.fit(Xtr, Ytr)
    print(wl.evaluate(Xtr, Ytr))
    print(wl.evaluate(Xts, Yts))

if __name__ == '__main__':
    from quadboost.datasets import MNISTDataset
    from quadboost.label_encoder import *
    main()
