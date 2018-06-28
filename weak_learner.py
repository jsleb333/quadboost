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
    def __init__(self, encoder=None):
        self.encoder = encoder
    
    def fit(self, X, Y, W=None):
        if self.encoder != None:
            Y, W = self.encoder.encode_labels(Y)
        X = X.reshape((X.shape[0], -1))

        feature_sorted_X_idx, stump_idx_ptr = self.stump_sort(X)
        stump_idx, self.feature, self.confidence_rates = self._find_stump(feature_sorted_X_idx[:,:784], stump_idx_ptr[:,:784], Y, W)

        feature_value = lambda stump_idx: X[feature_sorted_X_idx[stump_idx,self.feature],self.feature]
        self.stump = (feature_value(stump_idx) + feature_value(stump_idx-1))/2 if stump_idx != 0 else feature_value(stump_idx) - 1
        
        return self

    def stump_sort(self, X):
        """
        Returns indices of examples sorted by features, and a stump pointer index on these indices. The stump pointer index contains the indices where the features change in the sorted examples. Zeros in the stump pointer index are filling/padding and should be ignored.
        """
        feature_sorted_X_idx = X.argsort(axis=0)

        idx_to_features, indices = np.unique(X, return_inverse=True)
        sorted_indices = np.sort(indices.reshape(X.shape), axis=0)

        n_values = len(idx_to_features)
        n_examples, n_features = X.shape
         
        stump_idx_ptr = np.zeros((n_values, n_features), dtype=int)
        for row_idx, row in enumerate(sorted_indices):
            stump_idx_ptr[row,range(len(row))] = row_idx + 1

        return feature_sorted_X_idx, stump_idx_ptr

    @timed
    def _find_stump(self, feature_sorted_X_idx, stump_idx_ptr, Y, W):
        n_examples, n_classes = Y.shape
        n_values, n_features = stump_idx_ptr.shape
        n_partitions = 2
        n_moments = 3

        moments = np.zeros((n_moments, n_partitions, n_features, n_classes))

        # At first, all examples are in partition 1
        # All moments are not normalized so they can be computed cumulatively
        moments[0,1] = np.sum(W, axis=0)
        moments[1,1] = np.sum(W*Y, axis=0)
        moments[2,1] = np.sum(W*Y**2, axis=0)
        self.moments_update = np.zeros((n_moments, n_features, n_classes))

        risk = np.sum(np.sum(moments[2] - np.divide(moments[1]**2, moments[0], where=moments[0]!=0), axis=0), axis=1)

        best_feature = risk.argmin()
        best_risk = risk[best_feature]
        best_moment_0 = moments[0,:,best_feature,:].copy()
        best_moment_1 = moments[1,:,best_feature,:].copy()
        best_stump_idx = 0

        stump_idx = np.zeros(n_features, dtype=int)
        for stump_update in stump_idx_ptr:
            prev_stump_idx = stump_idx.copy()
            stump_idx = np.where(stump_update!=0, stump_update, stump_idx)

            self._update_moments(prev_stump_idx, stump_idx, feature_sorted_X_idx, Y, W, moments)
            self._compute_risk(moments)

            feature = risk.argmin()
            if risk[feature] < best_risk:
                best_feature = feature
                best_risk = risk[best_feature]
                best_moment_0 = moments[0,:,best_feature,:].copy()
                best_moment_1 = moments[1,:,best_feature,:].copy()
                best_stump_idx = stump_idx[best_feature]
        
        confidence_rates = np.divide(best_moment_1, best_moment_0, where=best_moment_0!=0)
        delattr(self, 'moments_update')

        return best_stump_idx, best_feature, confidence_rates
    
    def _update_moments(self, prev_stump_idx, stump_idx, feature_sorted_X_idx, Y, W, moments):
        for feature, (ps, s) in enumerate(zip(prev_stump_idx, stump_idx)):
            x_idx = feature_sorted_X_idx[ps:s,feature]
            self.moments_update[0,feature] = np.sum(W[x_idx], axis=0)
            self.moments_update[1,feature] = np.sum(W[x_idx]*Y[x_idx], axis=0)
            self.moments_update[2,feature] = np.sum(W[x_idx]*Y[x_idx]**2, axis=0)        
        moments[:,0] = moments[:,0] + self.moments_update
        moments[:,1] = moments[:,1] - self.moments_update

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
    m = 60
    # X = Xtr.reshape((m,-1))[:m,520:523]
    X = Xtr[:m]
    Y = Ytr[:m]
    # X, Y = Xtr, Ytr
    wl = MulticlassDecisionStump(encoder=encoder)
    wl.fit(X, Y)
    print('WL train acc:', wl.evaluate(X, Y))
    # print('WL test acc:', wl.evaluate(Xts, Yts))




if __name__ == '__main__':
    main()