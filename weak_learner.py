import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Ridge

from label_encoder import LabelEncoder, OneHotEncoder, AllPairsEncoder
from mnist_dataset import MNISTDataset


class WLRidgeMH(Ridge):
    """
    Ridge classification based on the sign of a Ridge regression.
    Inherits from Ridge of the scikit-learn package.
    In this implementation, the method 'fit' does not support encoding weights of the QuadBoost algorithm.
    """
    def __init__(self, *args, alpha=1, encoder=None, fit_intercept=False, **kwargs):
        super().__init__(*args, alpha=alpha, fit_intercept=fit_intercept, **kwargs)
        self.encoder = encoder
    
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
        return np.sign(super().predict(X, **kwargs))

    def evaluate(self, X, Y):
        Y_pred = self.predict(X)
        if self.encoder != None:
            Y_pred = self.encoder.decode_labels(Y_pred)
        return accuracy_score(y_true=Y, y_pred=Y_pred)


def main():
    mnist = MNISTDataset.load()
    (Xtr, Ytr), (Xts, Yts) = mnist.get_train_test(center=True, reduce=True)

    # encoder = LabelEncoder.load_encodings('js_without_0', convert_to_int=True)
    # encoder = LabelEncoder.load_encodings('mario')
    # encoder = OneHotEncoder(Ytr)
    encoder = AllPairsEncoder(Ytr)
    
    wl = WLRidgeMH(encoder=encoder)
    wl.fit(Xtr, Ytr)
    print('WL train acc:', wl.evaluate(Xtr, Ytr))
    print('WL test acc:', wl.evaluate(Xts, Yts))

if __name__ == '__main__':
    from time import time
    t = time()
    try:
        main()
    except:
        print('\nExecution terminated after {:.2f} seconds.\n'.format(time()-t))
        raise
    print('\nExecution completed in {:.2f} seconds.\n'.format(time()-t))