import numpy as np
from sklearn.metrics import accuracy_score
import multiprocessing as mp

import sys, os
sys.path.append(os.getcwd())

try:
    from quadboost.weak_learner import _WeakLearnerBase
    from quadboost.utils import split_int, timed, ComparableMixin
    from quadboost.utils.multiprocessing_utils import PicklableExceptionWrapper, SafeQueue, parallel_processes
except ModuleNotFoundError:
    from weak_learner import _WeakLearnerBase
    from utils import split_int, timed, ComparableMixin


class MulticlassDecisionStump(_WeakLearnerBase):
    """
    Decision stump classifier with innate multiclass algorithm.
    It finds a stump to partition examples into 2 parts which minimizes the quadratic multiclass risk.
    It assigns a confidence rates (scalar) for each class for each partition.
    Parallelization is implemented for the 'fit' method.
    """
    def fit(self, X, Y, W=None, n_jobs=1, sorted_X=None, sorted_X_idx=None):
        """
        Fits the model by finding the best decision stump using the algorithm implemented in the StumpFinder class.

        Args:
            X (Array of shape (n_examples, ...)): Examples
            Y (Array of shape (n_examples,) or (n_examples, n_classes)): Labels for the examples. If an encoder was provided at construction, Y should be a vector to be encoded.
            W (Array of shape (n_examples, n_classes)): Weights of each examples according to their class. Should be None if Y is not encoded.
            n_jobs (int, optional, default=1): Number of processes to execute in parallel to find the stump.
            sorted_X (Array of shape (n_examples, ...), optional, default=None): Sorted examples along axis 0. If None, 'X' will be sorted, else it will not.
            sorted_X_idx (Array of shape (n_examples, ...), optional, default=None): Indices of the sorted examples along axis 0 (corresponds to argsort). If None, 'X' will be argsorted, else it will not.

        Returns self
        """
        if self.encoder is not None:
            Y, W = self.encoder.encode_labels(Y)
        if sorted_X is None or sorted_X_idx is None:
            sorted_X, sorted_X_idx = self.sort_data(X)

        stump = self.find_stump(sorted_X, sorted_X_idx, Y, W, n_jobs)

        for attr in ['feature', 'confidence_rates', 'stump', 'stump_idx', 'risks', 'risk']:
            setattr(self, attr, getattr(stump, attr))

        return self

    @staticmethod
    def find_stump(sorted_X, sorted_X_idx, Y, W):
        stump_finder = StumpFinder(sorted_X, sorted_X_idx, Y, W)
        stumps_queue = []
        stump_finder.find_stump(stumps_queue)

        return min(stump for stump in stumps_queue)

    def predict(self, X):
        n_partitions, n_classes = self.confidence_rates.shape
        n_examples = X.shape[0]
        Y_pred = np.zeros((n_examples, n_classes))
        for i, partition in enumerate(self.partition_generator(X)):
            Y_pred[i] = self.confidence_rates[partition]
        return Y_pred

    def partition_generator(self, X):
        """
        Partition examples into 2 sets denoted by 0 and 1 in an lazy iterator fashion.
        """
        n_examples = X.shape[0]
        for x in X.reshape((n_examples, -1)):
            yield int(x[self.feature] > self.stump)

    def partition(self, X, dtype=bool):
        return np.array([p for p in self.partition_generator(X)], dtype=dtype)

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

        # multiprocessing Arrays are shared between processed to alleviate pickling
        self.sorted_X = np.ctypeslib.as_array(mp.RawArray('d', sorted_X.size)).reshape(sorted_X.shape)
        self.sorted_X[:] = sorted_X
        self.sorted_X_idx = np.ctypeslib.as_array(mp.RawArray('i', sorted_X_idx.size)).reshape(sorted_X_idx.shape)
        self.sorted_X_idx[:] = sorted_X_idx

        self.zeroth_moments = np.ctypeslib.as_array(mp.RawArray('d', W.size)).reshape(W.shape)
        self.zeroth_moments[:] = W
        self.first_moments = np.ctypeslib.as_array(mp.RawArray('d', W.size)).reshape(W.shape)
        self.first_moments[:] = W*Y
        self.second_moments = np.ctypeslib.as_array(mp.RawArray('d', W.size)).reshape(W.shape)
        self.second_moments[:] = self.first_moments*Y

        # # multiprocessing Arrays are shared between processed to alleviate pickling
        # self.X_shape = sorted_X.shape
        # self.X_idx_shape = sorted_X_idx.shape
        # self.moments_shape = W.shape
        # self.sorted_X = mp.Array('d', sorted_X.reshape(-1))
        # self.sorted_X_idx = mp.Array('i', sorted_X_idx.reshape(-1))

        # self.zeroth_moments = mp.Array('d', W.reshape(-1))
        # self.first_moments = mp.Array('d', (W*Y).reshape(-1))
        # self.second_moments = mp.Array('d', (W*Y*Y).reshape(-1))

    def safe_find_stump(self, stumps_queue, sub_idx=(None,)):
        """
        Handles exception raised in a subprocess so the script will not hang indefinitely.

        This is basically a decorator for find_stump, but parallelizing requires pickling, and decorators cannot be pickled.
        """
        with stumps_queue: # Context manager handles exceptions
            self.find_stump(stumps_queue, sub_idx)

    def find_stump(self, stumps_queue, sub_idx=(None,)):
        """
        Algorithm to the best stump within the sub array of X specified by the bounds 'sub_idx'.
        """
        X = self.sorted_X[:,slice(*sub_idx)]
        X_idx = self.sorted_X_idx[:,slice(*sub_idx)]

        _, n_classes = self.zeroth_moments.shape
        n_examples, n_features = X.shape
        n_partitions = 2
        n_moments = 3

        moments = np.zeros((n_moments, n_partitions, n_features, n_classes))

        # At first, all examples are in partition 1
        # Moments are not normalized so they can be computed cumulatively
        moments[0,1] = np.sum(self.zeroth_moments[X_idx[:,0]], axis=0)
        moments[1,1] = np.sum(self.first_moments[X_idx[:,0]], axis=0)
        moments[2,1] = np.sum(self.second_moments[X_idx[:,0]], axis=0)

        risks = self.compute_risks(moments) # Shape (n_partitions, n_features)
        best_stump = Stump(risks, moments)

        for i, row in enumerate(X_idx[:-1]):
            self.update_moments(moments, row)
            possible_stumps = ~np.isclose(X[i+1] - X[i], 0)

            if possible_stumps.any():
                risk = self.compute_risks(moments[:,:,possible_stumps,:])
                best_stump.update(risk, moments, possible_stumps, stump_idx=i+1)

        best_stump.compute_stump_value(X)
        best_stump.feature += sub_idx[0] if sub_idx[0] is not None else 0
        stumps_queue.append(best_stump)

    def update_moments(self, moments, row_idx):
        moments_update = np.array([self.zeroth_moments[row_idx],
                                   self.first_moments[row_idx],
                                   self.second_moments[row_idx]])
        moments[:,0] += moments_update
        moments[:,1] -= moments_update

    def compute_risks(self, moments):
        """
        Computes the risks for each partitions for every features.
        """
        moments[np.isclose(moments,0)] = 0
        with np.errstate(divide='ignore', invalid='ignore'):
            # We could use
            # np.divide(moments[1]**2, moments[0], where=~np.isclose(moments[0]))
            # However, the buffer size is not big enough for several examples and the resulting division is not done correctly
            normalized_m1 = np.nan_to_num(moments[1]**2/moments[0])
        risks = np.sum(moments[2] - normalized_m1, axis=2) # Shape (n_partitions, n_features)
        return risks


class Stump(ComparableMixin, cmp_attr='risk'):
    """
    Stump is a simple class that stores the variables used by the MulticlassDecisionStump algorithm. It provides a method 'update' that changes the values only if the new stump is better than the previous one. Easy comparison between the stumps is provided with the ComparableMixin parent class, which is useful to determine the best stump among many.
    """
    def __init__(self, risks, moments):
        super().__init__()
        risk = np.sum(risks, axis=0)
        self.feature = risk.argmin()
        self.risks = risks[:,self.feature]
        self.stump_idx = 0
        self.moment_0 = moments[0,:,self.feature,:].copy()
        self.moment_1 = moments[1,:,self.feature,:].copy()

    @property
    def risk(self):
        return np.sum(self.risks)

    def update(self, risks, moments, possible_stumps, stump_idx):
        """
        Updates the current stump with the new stumps only if the new risk is lower than the previous one.

        To optimize the algorithm, the risks are computed only for the acceptable stumps, which happen to be represented as the non zero entries of the variable 'possible_stumps'.
        """
        risk = np.sum(risks, axis=0)
        sparse_feature_idx = risk.argmin()
        if risk[sparse_feature_idx] < self.risk:
            self.feature = possible_stumps.nonzero()[0][sparse_feature_idx] # Retrieves the actual index of the feature
            self.risks = risks[:,sparse_feature_idx]
            self.moment_0 = moments[0,:,self.feature,:].copy()
            self.moment_1 = moments[1,:,self.feature,:].copy()
            self.stump_idx = stump_idx

    @property
    def confidence_rates(self):
        return np.divide(self.moment_1, self.moment_0, where=self.moment_0!=0)

    def compute_stump_value(self, sorted_X):
        feat_val = lambda idx: sorted_X[idx, self.feature]
        if self.stump_idx != 0:
            self.stump = (feat_val(self.stump_idx) + feat_val(self.stump_idx-1))/2
        else:
            self.stump = feat_val(self.stump_idx) - 1
        return self.stump


cdef int n_classes
cdef int n_partitions = 2
cdef int n_features
cdef int n_examples

cdef struct Stump:
    float risk
    float risks[n_partitions * n_classes]
    int feature
    int stump_idx
    float moment_0[n_partitions * n_classes]
    float moment_1[n_partitions * n_classes]
    float confidence_rates[n_partitions * n_classes]

cdef void update_stump(Stump* stump):
    pass


