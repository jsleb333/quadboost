import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVR

try:
    from quadboost.weak_learner import _WeakLearnerBase
    from quadboost.utils import timed
except ModuleNotFoundError:
    from weak_learner import _WeakLearnerBase
    from utils import timed

class MultidimSVR(_WeakLearnerBase):
    """
    Implements a non-coupled multidimensional output SVM regressor based on the LinearSVR of sci-kit learn. This is highly non-efficient for large dataset.
    """
    def __init__(self, *args, encoder=None, **kwargs):
        super().__init__(encoder=encoder)
        self.predictors = []
        self.svm = lambda: LinearSVR(*args, **kwargs)

    def fit(self, X, Y, W=None, **kwargs):
        X = X.reshape((X.shape[0], -1))
        if self.encoder != None:
            Y, W = self.encoder.encode_labels(Y)
        print(self.encoder)
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


@timed
def main():
    mnist = MNISTDataset.load()
    (Xtr, Ytr), (Xts, Yts) = mnist.get_train_test(center=True, reduce=False)

    encoder = OneHotEncoder(Ytr)

    wl = MultidimSVR(encoder=encoder)
    wl.fit(Xtr[:100], Ytr[:100])
    print(wl.evaluate(Xts, Yts))


if __name__ == '__main__':
    from quadboost.mnist_dataset import MNISTDataset
    from quadboost.label_encoder import *
    main()
