import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVR
import functools
from collections import defaultdict

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


@cloner
class MulticlassDecisionStump:
    def __init__(self, encoder=None, bins=-1):
        self.encoder = encoder
        self.bins = bins
    
    def fit(self, X, Y, W=None):
        if self.encoder != None:
            Y, W = self.encoder.encode_labels(Y)
        X = X.reshape((X.shape[0], -1))

        feature_sorted_X_idx, stump_idx_ptr = self.stump_sort(X)

        # confidence, variance, mass = self._compute_confidence_variance_mass(sorted_X_idx, Y, W)

        # risk = np.sum(np.sum(variance, axis=3), axis=1)
        # stump_idx, feature_idx = self._find_best_stump(risk, X, sorted_X_idx)

        # self.confidence_rates = np.divide(confidence[stump_idx,:,feature_idx], mass[stump_idx,:,feature_idx], where=mass[stump_idx,:,feature_idx]!=0)
        
        return self
    
    def stump_sort(self, X):
        """
        Returns indices to sorted examples by features, and a stump pointer index on these indices. The stump pointer index contains the indices where the features change in the sorted examples. Zeros in the stump pointer index are filling and should be ignored.
        """
        feature_sorted_X_idx = X.argsort(axis=0)

        idx_to_features, indices = np.unique(X, return_inverse=True)
        sorted_indices = np.sort(indices.reshape(X.shape), axis=0)

        n_values = len(idx_to_features)
        n_examples, n_features = X.shape
         
        stump_idx_ptr = np.zeros((n_values, n_features))
        for row_idx, row in enumerate(sorted_indices):
            stump_idx_ptr[row,range(len(row))] = row_idx + 1

        # print(sorted_indices)
        # print(stump_idx_ptr)
        # print(X.argsort(axis=0))
        # print(np.sort(X, axis=0))
        # print(idx_to_features)

        return feature_sorted_X_idx, stump_idx_ptr
    
    def _find_best_stump(self, risk, X, sorted_X_idx):
        risk_idx = (np.unravel_index(idx, risk.shape) for idx in np.argsort(risk, axis=None))

        for stump_idx, feature_idx in risk_idx:
            feature_value = X[sorted_X_idx[stump_idx, feature_idx], feature_idx]
            if stump_idx == 0:
                self.stump = feature_value -1
                self.feature = feature_idx
                return stump_idx, feature_idx
            neighbour_value = X[sorted_X_idx[stump_idx-1, feature_idx], feature_idx]

            if feature_value != neighbour_value: # We cannot have a stump between 2 examples with the same feature value.
                self.stump = (feature_value + neighbour_value)/2
                self.feature = feature_idx
                return stump_idx, feature_idx
        
        raise ValueError('All examples are identical.')

    def _find_stump(self, feature_sorted_X_idx, stump_idx_ptr, Y, W):
        n_examples, n_classes = Y.shape
        n_partitions = 2

        0th_moment = np.zeros((n_partitions))
        1st_moment
        2nd_moment





        n_stumps, n_features = feature_decomposed_X.shape
        n_partitions = 2 # Decision stumps partition space into 2 (partition 0 on the left and partition 1 on the right)

        confidence = np.zeros((n_stumps, n_partitions, n_features, n_classes))
        mass = np.zeros_like(confidence)
        variance = np.zeros_like(confidence)

        Y_broad = Y[:,np.newaxis,:]
        W_broad = W[:,np.newaxis,:]

        # At first, all examples are at the right side of the stump (partition 1)
        mass[0,1] = np.sum(W, axis=0)
        confidence[0,1] = np.sum(W*Y, axis=0)
        c = np.divide(confidence[0,1], mass[0,1], where=mass[0,1]!=0)
        variance[0,1] = np.sum(W_broad*(Y_broad-c[np.newaxis])**2, axis=0)

        mass_update = np.zeros_like(mass[0,0])
        confidence_update = np.zeros_like(mass_update)
        variance_update = np.zeros_like(mass_update)

        for stump_idx, x_indices in enumerate(feature_decomposed_X):
            for feature, indices in enumerate(x_indices):
                mass_update[feature] = np.sum(W[indices], axis=0)
                confidence_update[feature] = np.sum(W[indices]*Y[indices], axis=0)

            mass[stump_idx,0] = mass[stump_idx-1,0] + mass_update # Example is added to partition 0
            mass[stump_idx,1] = mass[stump_idx-1,1] - mass_update # Example is removed from partition 1

            confidence[stump_idx,0] = confidence[stump_idx-1,0] + confidence_update
            confidence[stump_idx,1] = confidence[stump_idx-1,1] - confidence_update

            c0 = np.divide(confidence[stump,0], mass[stump,0], where=mass[stump,0]!=0)
            c1 = np.divide(confidence[stump-1,1], mass[stump-1,1], where=mass[stump-1,1]!=0)

            variance[stump,0] = variance[stump-1,0] + W[x_idx]*(Y[x_idx] - c0)**2
            variance[stump,1] = variance[stump-1,1] - W[x_idx]*(Y[x_idx] - c1)**2




        for x_idx, stump in zip(sorted_X_idx, range(1, n_stumps)):
            mass[stump,0] = mass[stump-1,0] + W[x_idx] # Example is added to partition 0
            mass[stump,1] = mass[stump-1,1] - W[x_idx] # Example is removed from partition 1

            confidence[stump,0] = confidence[stump-1,0] + W[x_idx]*Y[x_idx]
            confidence[stump,1] = confidence[stump-1,1] - W[x_idx]*Y[x_idx]

            c0 = np.divide(confidence[stump,0], mass[stump,0], where=mass[stump,0]!=0)
            c1 = np.divide(confidence[stump-1,1], mass[stump-1,1], where=mass[stump-1,1]!=0)

            variance[stump,0] = variance[stump-1,0] + W[x_idx]*(Y[x_idx] - c0)**2
            variance[stump,1] = variance[stump-1,1] - W[x_idx]*(Y[x_idx] - c1)**2
        
        return confidence, variance, mass
    
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
        

@timed
def main():
    mnist = MNISTDataset.load()
    (Xtr, Ytr), (Xts, Yts) = mnist.get_train_test(center=False, reduce=False)

    # encoder = LabelEncoder.load_encodings('js_without_0', convert_to_int=True)
    # encoder = LabelEncoder.load_encodings('mario')
    encoder = OneHotEncoder(Ytr)
    # encoder = AllPairsEncoder(Ytr)

    # wl = WLRidgeMH(encoder=encoder)
    # wl = WLRidgeMHCR(encoder=encoder)
    # wl = WLThresholdedRidge(encoder=encoder, threshold=.5)
    m = 5
    X = Xtr.reshape((m,-1))[:m,520:523]
    Y = Ytr[:m]
    # X, Y = Xtr, Ytr
    wl = MulticlassDecisionStump(encoder=encoder)
    wl.fit(X, Y)
    # print('WL train acc:', wl.evaluate(X, Y))
    # print('WL test acc:', wl.evaluate(Xts, Yts))




if __name__ == '__main__':
    main()