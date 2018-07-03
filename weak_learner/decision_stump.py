import numpy as np
from sklearn.metrics import accuracy_score
import sys, os
print(os.getcwd())
sys.path.append(os.getcwd())
from weak_learner import cloner
from utils import *


@cloner
class MulticlassDecisionStump:
    def __init__(self, encoder=None):
        self.encoder = encoder

    def fit(self, X, Y, W=None):
        if self.encoder != None:
            Y, W = self.encoder.encode_labels(Y)
        X = X.reshape((X.shape[0], -1))

        sorted_X_idx = X.argsort(axis=0)
        sorted_X = X[sorted_X_idx, range(X.shape[1])]

        batch_size = X.shape[1]
        stump = self.find_stump(sorted_X[:,:batch_size], sorted_X_idx[:,:batch_size], Y, W)

        self.feature = stump.feature
        self.confidence_rates = stump.compute_confidence_rates()
        idx = stump.stump_idx

        feature_value = lambda idx: X[sorted_X_idx[idx, self.feature], self.feature]
        if idx != 0:
            self.stump = (feature_value(idx) + feature_value(idx-1))/2
        else:
            self.stump = feature_value(idx) - 1

        return self

    @timed
    def find_stump(self, sorted_X, sorted_X_idx, Y, W):
        n_examples, n_classes = Y.shape
        _, n_features = sorted_X.shape
        n_partitions = 2
        n_moments = 3

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


class Stump:
    """
    Stump is a simple class that stores the variables used by the MulticlassDecisionStump algorithm. It provides a method 'update' that changes the values only if the new stump is better than the previous one. It also provides a method 'compute_confidence_rates' for the stored stump.
    """
    def __init__(self, risk, moments):
        self.feature = risk.argmin()
        self.risk = risk[self.feature]
        self.stump_idx = 0
        self.moment_0 = moments[0,:,self.feature,:].copy()
        self.moment_1 = moments[1,:,self.feature,:].copy()

    def update(self, risk, moments, possible_stumps, stump_idx):
        """
        Updates the current stump with the new stumps only if the new risk is lower than the previous one.

        To optimize the algorithm, the risks are computed only for the acceptable stumps, which happen to be represented as the non zero entries of the variable 'possible_stumps'.
        """
        sparse_feature_idx = risk.argmin()
        if risk[sparse_feature_idx] < self.risk:
            self.feature = possible_stumps.nonzero()[0][sparse_feature_idx] # Retrieves the actual index of the feature
            self.risk = risk[sparse_feature_idx]
            self.moment_0 = moments[0,:,self.feature,:].copy()
            self.moment_1 = moments[1,:,self.feature,:].copy()
            self.stump_idx = stump_idx

    def compute_confidence_rates(self):
        return np.divide(self.moment_1, self.moment_0, where=self.moment_0!=0)


@timed
def main():
    mnist = MNISTDataset.load()
    (Xtr, Ytr), (Xts, Yts) = mnist.get_train_test(center=False, reduce=False)

    # encoder = LabelEncoder.load_encodings('js_without_0', convert_to_int=True)
    # encoder = LabelEncoder.load_encodings('mario')
    encoder = OneHotEncoder(Ytr)
    # encoder = AllPairsEncoder(Ytr)

    m = 10_000
    X = Xtr[:m].reshape((m,-1))
    Y = Ytr[:m]
    # X, Y = Xtr, Ytr
    wl = MulticlassDecisionStump(encoder=encoder)
    wl.fit(X, Y)
    print('WL train acc:', wl.evaluate(X, Y))
    # print('WL test acc:', wl.evaluate(Xts, Yts))


if __name__ == '__main__':
    from mnist_dataset import MNISTDataset
    from label_encoder import *
    import cProfile
    cProfile.run('main()', sort='tottime')
