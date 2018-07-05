import numpy as np
from sklearn.metrics import accuracy_score
from multiprocessing import Pool
import joblib as jl
from functools import partial

import sys, os
sys.path.append(os.getcwd())

from weak_learner import cloner
from weak_learner.decision_stump import Stump
from utils import *


@cloner
class MulticlassDecisionStump:
    def __init__(self, encoder=None):
        self.encoder = encoder

    def fit(self, X, Y, W=None, n_jobs=4):
        if self.encoder != None:
            Y, W = self.encoder.encode_labels(Y)
        X = X.reshape((X.shape[0], -1))

        # stump = self.find_stump(X, Y, W, n_jobs=n_jobs)
        stump = self._find_stump(X, Y, W)

        self.feature = stump.feature
        self.confidence_rates = stump.compute_confidence_rates()
        self.stump = stump.stump

        return self

    def find_stump(self, X, Y, W, n_jobs):
        n_features = X.shape[1]
        # self.pool = Pool(n_jobs)
        find_stump_XYW = partial(self._find_stump, X, Y, W)
        # stumps = self.pool.map(find_stump_XYW, split_int(n_features, n_jobs))

        parallelizer = jl.Parallel(n_jobs=n_jobs)
        stumps = parallelizer(jl.delayed(find_stump_XYW)(sub_idx) for sub_idx in split_int(n_features, n_jobs))
        stump = min(stumps)
        return stump

    def _find_stump(self, X, Y, W, sub_idx=(None,)):
        n_examples, n_classes = Y.shape
        _, n_features = X[:,slice(*sub_idx)].shape
        n_partitions = 2
        n_moments = 3

        sorted_X_idx = X[:,slice(*sub_idx)].argsort(axis=0)
        sorted_X = X[:,slice(*sub_idx)][sorted_X_idx, range(n_features)]

        moments = np.zeros((n_moments, n_partitions, n_features, n_classes))
        moments_update = np.zeros((n_moments, n_features, n_classes))

        # At first, all examples are in partition 1
        # Moments are not normalized so they can be computed cumulatively
        moments[0,1] = np.sum(W, axis=0)
        moments[1,1] = np.sum(W*Y, axis=0)
        moments[2,1] = np.sum(W*Y**2, axis=0)

        risk = self._compute_risk(moments)
        best_stump = Stump(risk, moments)

        for i, row in enumerate(sorted_X_idx[:-1]):
            self._update_moments(moments, moments_update, W[row], Y[row])
            possible_stumps = ~np.isclose(sorted_X[i+1] - sorted_X[i], 0)

            if possible_stumps.any():
                risk = self._compute_risk(moments[:,:,possible_stumps,:])
                best_stump.update(risk, moments, possible_stumps, stump_idx=i+1)

        best_stump.compute_stump_value(sorted_X)
        best_stump.feature += sub_idx[0] if sub_idx[0] is not None else 0
        return best_stump

    def _update_moments(self, moments, moments_update, weights_update, labels_update):
        moments_update[0] = weights_update
        moments_update[1] = weights_update*labels_update
        moments_update[2] = weights_update*labels_update**2

        moments[:,0] += moments_update
        moments[:,1] -= moments_update

    def _compute_risk(self, moments):
        moments[np.isclose(moments,0)] = 0
        with np.errstate(divide='ignore', invalid='ignore'):
            # We could use
            # np.divide(moments[1]**2, moments[0], where=~np.isclose(moments[0]))
            # However, the buffer size is not big enough for several examples and the resulting division is not done correctly
            normalized_m1 = np.nan_to_num(moments[1]**2/moments[0])
        risk = np.sum(np.sum(moments[2] - normalized_m1, axis=2), axis=0)
        return risk

    def predict(self, X):
        n_partitions, n_classes = self.confidence_rates.shape
        n_examples = X.shape[0]
        X = X.reshape((n_examples, -1))
        Y_pred = np.zeros((n_examples, n_classes))
        for i, x in enumerate(X):
            if x[self.feature] < self.stump:
                Y_pred[i] = self.confidence_rates[0]
            else:
                Y_pred[i] = self.confidence_rates[1]
        return Y_pred

    def evaluate(self, X, Y):
        Y_pred = self.predict(X)
        if self.encoder != None:
            Y_pred = self.encoder.decode_labels(Y_pred)
        return accuracy_score(y_true=Y, y_pred=Y_pred)

    # def __getstate__(self):
    #     self_dict = self.__dict__.copy()
    #     del self_dict['pool']
    #     return self_dict

@timed
def main():
    mnist = MNISTDataset.load()
    (Xtr, Ytr), (Xts, Yts) = mnist.get_train_test(center=False, reduce=False)

    # encoder = LabelEncoder.load_encodings('js_without_0', convert_to_int=True)
    # encoder = LabelEncoder.load_encodings('mario')
    encoder = OneHotEncoder(Ytr)
    # encoder = AllPairsEncoder(Ytr)

    m = 60_000
    X = Xtr[:m].reshape((m,-1))
    Y = Ytr[:m]
    # X, Y = Xtr, Ytr
    wl = MulticlassDecisionStump(encoder=encoder)
    wl.fit(X, Y, n_jobs=4)
    print('WL train acc:', wl.evaluate(X, Y))
    # print('WL test acc:', wl.evaluate(Xts, Yts))


if __name__ == '__main__':
    from mnist_dataset import MNISTDataset
    from label_encoder import *
    main()
