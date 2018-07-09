import numpy as np
from sklearn.metrics import accuracy_score
from multiprocessing import Pool
from functools import partial

import sys, os
sys.path.append(os.getcwd())

from weak_learner import cloner
from weak_learner.decision_stump import Stump
from utils import *


@cloner
class MulticlassDecisionStump:
    """
    Decision stump classifier with innate multiclass algorithm.
    It finds a stump to partition examples into 2 parts which minimizes the quadratic multiclass risk.
    It assigns a confidence rates (scalar) for each class for each partition.
    Parallelization is implemented in the 'fit' method.
    """
    def __init__(self, encoder=None):
        """
        encoder (LabelEncoder object, optional, default=None): Encoder to encode labels. If None, no encoding will be made before fitting.
        """
        self.encoder = encoder

    def fit(self, X, Y, W=None, n_jobs=1):
        """
        Fits the model by finding the best decision stump using the algorithm implemented in the StumpFinder class.

        X (Array of shape (n_examples, ...)): Examples
        Y (Array of shape (n_examples,) or (n_examples, n_classes)): Labels for the examples. If an encoder was provided at construction, Y should be a vector to be encoded.
        W (Array of shape (n_examples, n_classes)): Weights of each examples according to their class. Should be None if Y is not encoded.
        n_jobs (int, optional, default=1): Number of processes to execute in parallel to find the stump.

        Returns self
        """
        if self.encoder != None:
            Y, W = self.encoder.encode_labels(Y)
        X = X.reshape((X.shape[0], -1))
        _, n_features = X.shape

        stump = self.parallel_find_stump(X, Y, W, n_jobs)

        self.feature = stump.feature
        self.confidence_rates = stump.compute_confidence_rates()
        self.stump = stump.stump

        return self

    def parallel_find_stump(self, X, Y, W, n_jobs):
        """
        Parallelizes the processes.
        """
        n_features = X.shape[1]
        stump_finder = StumpFinder(X, Y, W)
        if n_jobs > 1:
            pool = Pool(n_jobs)
            stumps = pool.map(stump_finder.find_stump, split_int(n_features, n_jobs))
            stump = min(stumps)
        else:
            stump = stump_finder.find_stump()

        return stump

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

class StumpFinder:
    """
    Implements the algorithm to find the stump. It is separated from the class MulticlassDecisionStump so that it can be pickled when parallelized with 'multiprocessing' (which uses pickle).
    """
    def __init__(self, X, Y, W):
        self.X = X
        self.zeroth_moments = W
        self.first_moments = W*Y
        self.second_moments = W*Y**2

    def find_stump(self, sub_idx=(None,)):
        """
        Algorithm to the best stump within the sub array of X specfied by the bounds 'sub_idx'.
        """
        n_examples, n_classes = self.zeroth_moments.shape
        _, n_features = self.X[:,slice(*sub_idx)].shape
        n_partitions = 2
        n_moments = 3

        sorted_X_idx = self.X[:,slice(*sub_idx)].argsort(axis=0)
        sorted_X = self.X[:,slice(*sub_idx)][sorted_X_idx, range(n_features)]

        moments = np.zeros((n_moments, n_partitions, n_features, n_classes))
        # moments_update = np.zeros((n_moments, n_features, n_classes))

        # At first, all examples are in partition 1
        # Moments are not normalized so they can be computed cumulatively
        moments[0,1] = np.sum(self.zeroth_moments, axis=0)
        moments[1,1] = np.sum(self.first_moments, axis=0)
        moments[2,1] = np.sum(self.second_moments, axis=0)

        risk = self.compute_risk(moments)
        best_stump = Stump(risk, moments)

        for i, row in enumerate(sorted_X_idx[:-1]):
            self.update_moments(moments, row)
            possible_stumps = ~np.isclose(sorted_X[i+1] - sorted_X[i], 0)

            if possible_stumps.any():
                risk = self.compute_risk(moments[:,:,possible_stumps,:])
                best_stump.update(risk, moments, possible_stumps, stump_idx=i+1)

        best_stump.compute_stump_value(sorted_X)
        best_stump.feature += sub_idx[0] if sub_idx[0] is not None else 0
        return best_stump

    def update_moments(self, moments, row_idx):
        # moments_update[0] = self.zeroth_moments[row_idx]
        # moments_update[1] = self.first_moments[row_idx]
        # moments_update[2] = self.second_moments[row_idx]
        # moments[:,0] += moments_update
        # moments[:,1] -= moments_update

        moments[0,0] += self.zeroth_moments[row_idx]
        moments[1,0] += self.first_moments[row_idx]
        moments[2,0] += self.second_moments[row_idx]

        moments[0,1] -= self.zeroth_moments[row_idx]
        moments[1,1] -= self.first_moments[row_idx]
        moments[2,1] -= self.second_moments[row_idx]

    def compute_risk(self, moments):
        moments[np.isclose(moments,0)] = 0
        with np.errstate(divide='ignore', invalid='ignore'):
            # We could use
            # np.divide(moments[1]**2, moments[0], where=~np.isclose(moments[0]))
            # However, the buffer size is not big enough for several examples and the resulting division is not done correctly
            normalized_m1 = np.nan_to_num(moments[1]**2/moments[0])
        risk = np.sum(np.sum(moments[2] - normalized_m1, axis=2), axis=0)
        return risk


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
    import cProfile
    cProfile.run('main()', sort='tottime')
    # main()
