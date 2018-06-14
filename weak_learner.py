import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVR
import functools

from label_encoder import LabelEncoder, OneHotEncoder, AllPairsEncoder
from mnist_dataset import MNISTDataset
from utils import *


def cloner(cls):
    """
    This function decorator makes any weak learners clonable by setting the __call__ function as a constructor using the initialization parameters.
    """
    @functools.wraps(cls)
    def wrapper(*args, **kwargs):
        def clone(self):
            return cls(*args, **kwargs)
        cls.__call__ = clone
        return cls(*args, **kwargs)
    return wrapper


@cloner
class WLRidge(Ridge):
    """
    Confidence rated Ridge classification based on a Ridge regression.
    Inherits from Ridge of the scikit-learn package.
    In this implementation, the method 'fit' does not support encoding weights of the QuadBoost algorithm.
    """
    def __init__(self, alpha=1, encoder=None, fit_intercept=False, **kwargs):
        super().__init__(alpha=alpha, fit_intercept=fit_intercept, **kwargs)
        self.encoder = encoder
    
    def fit(self, X, Y, W=None, **kwargs):
        """
        NB: this method supports encoding weights of the QuadBoost algorithm by multiplying Y by the square root of the weights W. This should be taken into account for continous predictions.
        """
        X = X.reshape((X.shape[0], -1))
        if self.encoder != None:
            Y, W = self.encoder.encode_labels(Y)
        if W is not None:
            Y *= np.sqrt(W)
        return super().fit(X, Y, **kwargs)
    
    def predict(self, X, **kwargs):
        X = X.reshape((X.shape[0], -1))
        return super().predict(X, **kwargs)

    def evaluate(self, X, Y):
        Y_pred = self.predict(X)
        if self.encoder != None:
            Y_pred = self.encoder.decode_labels(Y_pred)
        return accuracy_score(y_true=Y, y_pred=Y_pred)


@cloner
class WLThresholdedRidge(Ridge):
    """
    Ridge classification based on a ternary vote (1, 0, -1) of a Ridge regression based on a threshold. For a threshold of 0, it is equivalent to take the sign of the prediction.
    Inherits from Ridge of the scikit-learn package.
    In this implementation, the method 'fit' does not support encoding weights of the QuadBoost algorithm.
    """
    def __init__(self, alpha=1, encoder=None, threshold=0.5, fit_intercept=False, **kwargs):
        super().__init__(alpha=alpha, fit_intercept=fit_intercept, **kwargs)
        self.encoder = encoder
        self.threshold = threshold

    def fit(self, X, Y, W=None, **kwargs):
        """
        Note: this method does not support encoding weights of the QuadBoost algorithm.
        """
        X = X.reshape((X.shape[0], -1))
        if self.encoder != None:
            Y, W = self.encoder.encode_labels(Y)
        return super().fit(X, Y, **kwargs)
    
    def predict(self, X, **kwargs):
        X = X.reshape((X.shape[0], -1))
        Y = super().predict(X, **kwargs)
        Y = np.where(Y >= self.threshold, 1.0, Y)
        Y = np.where(np.logical_and(Y < self.threshold, Y > -self.threshold), 0, Y)
        Y = np.where(Y < -self.threshold, -1, Y)
        return Y

    def evaluate(self, X, Y):
        Y_pred = self.predict(X)
        if self.encoder != None:
            Y_pred = self.encoder.decode_labels(Y_pred)
        return accuracy_score(y_true=Y, y_pred=Y_pred)


@cloner
class MultidimSVR:
    """
    Implements a non-coupled multidimensional output SVM regressor based on the LinearSVR of sci-kit learn. This is highly non-efficient for large dataset. 
    """
    def __init__(self, *args, encoder=None, **kwargs):
        self.encoder = encoder
        self.predictors = []
        self.svm = lambda: LinearSVR(*args, **kwargs)
    
    def fit(self, X, Y, W=None, **kwargs):
        X = X.reshape((X.shape[0], -1))
        if self.encoder != None:
            Y, W = self.encoder.encode_labels(Y)
        for i, y in enumerate(Y.T):
            print('Fitting dim ' + str(i) + ' of Y...')
            self.predictors.append(self.svm().fit(X, y, **kwargs))
            print('Finished fit')
        return self

    def predict(self, X, **kwargs):
        n_samples = X.shape[0]
        X = X.reshape((n_samples, -1))
        Y = np.zeros((n_samples, len(self.predictors)))
        for i, predictor in enumerate(self.predictors):
            Y[:,i] = predictor.predict(X, **kwargs)
        return Y

    def evaluate(self, X, Y):
        Y_pred = self.predict(X)
        if self.encoder != None:
            Y_pred = self.encoder.decode_labels(Y_pred)
        return accuracy_score(y_true=Y, y_pred=Y_pred)    


@timed
def main():
    mnist = MNISTDataset.load()
    (Xtr, Ytr), (Xts, Yts) = mnist.get_train_test(center=True, reduce=True)

    # encoder = LabelEncoder.load_encodings('js_without_0', convert_to_int=True)
    # encoder = LabelEncoder.load_encodings('mario')
    encoder = OneHotEncoder(Ytr)
    # encoder = AllPairsEncoder(Ytr)

    # wl = WLRidgeMH(encoder=encoder)
    # wl = WLRidgeMHCR(encoder=encoder)
    # wl = WLThresholdedRidge(encoder=encoder, threshold=.5)
    wl = MultidimSVR(encoder=encoder)
    wl.fit(Xtr[:5000], Ytr[:5000])
    print('WL train acc:', wl.evaluate(Xtr[:5000], Ytr[:5000]))
    print('WL test acc:', wl.evaluate(Xts, Yts))

if __name__ == '__main__':
    main()