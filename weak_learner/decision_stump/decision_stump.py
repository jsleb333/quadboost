import numpy as np
from sklearn.metrics import accuracy_score
import multiprocessing as mp
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
    Parallelization is implemented for the 'fit' method.
    """
    def __init__(self, encoder=None):
        """
        encoder (LabelEncoder object, optional, default=None): Encoder to encode labels. If None, no encoding will be made before fitting.
        """
        self.encoder = encoder

    def fit(self, X, Y, W=None, n_jobs=1, sorted_X=None, sorted_X_idx=None):
        """
        Fits the model by finding the best decision stump using the algorithm implemented in the StumpFinder class.

        X (Array of shape (n_examples, ...)): Examples
        Y (Array of shape (n_examples,) or (n_examples, n_classes)): Labels for the examples. If an encoder was provided at construction, Y should be a vector to be encoded.
        W (Array of shape (n_examples, n_classes)): Weights of each examples according to their class. Should be None if Y is not encoded.
        n_jobs (int, optional, default=1): Number of processes to execute in parallel to find the stump.
        sorted_X (Array of shape (n_examples, ...), optional, default=None): Sorted examples along axis 0. If None, 'X' will be sorted, else it will not.
        sorted_X_idx (Array of shape (n_examples, ...), optional, default=None): Indices of the sorted examples along axis 0 (corresponds to argsort). If None, 'X' will be argsorted, else it will not.

        Returns self
        """
        if self.encoder != None:
            Y, W = self.encoder.encode_labels(Y)
        if sorted_X is None or sorted_X_idx is None:
            sorted_X, sorted_X_idx = self.sort_data(X)

        stump = self.parallel_find_stump(sorted_X, sorted_X_idx, Y, W, n_jobs)

        self.feature = stump.feature
        self.confidence_rates = stump.compute_confidence_rates()
        self.stump = stump.stump

        return self

    def parallel_find_stump(self, sorted_X, sorted_X_idx, Y, W, n_jobs):
        """
        Parallelizes the processes.
        """
        stump_finder = StumpFinder(sorted_X, sorted_X_idx, Y, W)
        n_features = stump_finder.sorted_X.shape[1]
        stumps_queue = mp.Queue()
        processes = []
        for sub_idx in split_int(n_features, n_jobs):
            process = mp.Process(target=stump_finder.find_stump, args=(stumps_queue, sub_idx))
            processes.append(process)

        for process in processes: process.start()
        for process in processes: process.join()

        stump = min(stumps_queue.get() for _ in processes)

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

    @staticmethod
    def sort_data(X):
        """
        Necessary sorting operations on the data to find the optimal stump. It is useful to sort the data prior to boost to speed up the algorithm, since the sorting step will not be made at each round.

        'sorted_X' and 'sorted_X_idx' should be passed as keyword arguments to the 'fit' method to speed up the algorithm.
        """
        X = X.reshape((X.shape[0],-1))
        n_examples, n_features = X.shape
        sorted_X_idx = np.argsort(X, axis=0)
        sorted_X = X[sorted_X_idx, range(n_features)]

        return sorted_X, sorted_X_idx

class StumpFinder:
    """
    Implements the algorithm to find the stump. It is separated from the class MulticlassDecisionStump so that it can be pickled when parallelized with 'multiprocessing' (which uses pickle).
    """
    def __init__(self, sorted_X, sorted_X_idx, Y, W):
        self.sorted_X = sorted_X
        self.sorted_X_idx = sorted_X_idx

        self.zeroth_moments = W
        self.first_moments = W*Y
        self.second_moments = self.first_moments*Y

    def find_stump(self, stumps_queue, sub_idx=(None,)):
        """
        Algorithm to the best stump within the sub array of X specfied by the bounds 'sub_idx'.
        """
        X = self.sorted_X[:,slice(*sub_idx)]
        X_idx = self.sorted_X_idx[:,slice(*sub_idx)]

        n_examples, n_classes = self.zeroth_moments.shape
        _, n_features = X.shape
        n_partitions = 2
        n_moments = 3

        moments = np.zeros((n_moments, n_partitions, n_features, n_classes))

        # At first, all examples are in partition 1
        # Moments are not normalized so they can be computed cumulatively
        moments[0,1] = np.sum(self.zeroth_moments, axis=0)
        moments[1,1] = np.sum(self.first_moments, axis=0)
        moments[2,1] = np.sum(self.second_moments, axis=0)

        risk = self.compute_risk(moments)
        best_stump = Stump(risk, moments)

        for i, row in enumerate(X_idx[:-1]):
            self.update_moments(moments, row)
            possible_stumps = ~np.isclose(X[i+1] - X[i], 0)

            if possible_stumps.any():
                risk = self.compute_risk(moments[:,:,possible_stumps,:])
                best_stump.update(risk, moments, possible_stumps, stump_idx=i+1)

        best_stump.compute_stump_value(X)
        best_stump.feature += sub_idx[0] if sub_idx[0] is not None else 0
        stumps_queue.put(best_stump)

    def update_moments(self, moments, row_idx):
        moments_update = np.array([self.zeroth_moments[row_idx],
                                      self.first_moments[row_idx],
                                      self.second_moments[row_idx]])
        moments[:,0] += moments_update
        moments[:,1] -= moments_update

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
    sorted_X, sorted_X_idx = wl.sort_data(X)
    wl.fit(X, Y, n_jobs=4, sorted_X=sorted_X, sorted_X_idx=sorted_X_idx)
    print('WL train acc:', wl.evaluate(X, Y))
    # print('WL test acc:', wl.evaluate(Xts, Yts))


if __name__ == '__main__':
    from mnist_dataset import MNISTDataset
    from label_encoder import *
    # import cProfile
    # cProfile.run('main()', sort='tottime')
    main()
