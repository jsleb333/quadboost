import numpy as np
from sklearn.metrics import accuracy_score
import torch
import sys, os
print(os.getcwd())
sys.path.append(os.getcwd())

from weak_learner import Cloner
from utils import *


class MulticlassDecisionStump(Cloner):
    def __init__(self, encoder=None):
        self.encoder = encoder

    def fit(self, X, Y, W=None):
        if self.encoder != None:
            Y, W = self.encoder.encode_labels(Y)
        X = X.reshape((X.shape[0], -1))
        torch.no_grad()
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        X, Y, W = map(torch.from_numpy, [X,Y,W]) # Convert to torch tensors with shared memory
        X_gpu, Y_gpu, W_gpu = [Z.cuda() for Z in [X,Y,W]] # Copied to GPU

        sorted_X, sorted_X_idx = torch.sort(X_gpu, dim=0)

        batch_size = X.shape[1]
        stump_idx, self.feature = self.find_stump(sorted_X[:,:batch_size],
                                                  sorted_X_idx[:,:batch_size],
                                                  Y_gpu, W_gpu)

        feature_value = lambda stump_idx: X[sorted_X_idx[stump_idx,self.feature],self.feature]
        self.stump = (feature_value(stump_idx) + feature_value(stump_idx-1))/2 if stump_idx != 0 else feature_value(stump_idx) - 1

        return self

    def find_stump(self, sorted_X, sorted_X_idx, Y, W):
        n_examples, n_classes = Y.shape
        _, n_features = sorted_X.shape
        n_partitions = 2
        n_moments = 3

        moments = torch.cuda.FloatTensor(n_moments, n_partitions, n_features, n_classes).fill_(0)
        moments_update = torch.cuda.FloatTensor(n_moments, n_features, n_classes).fill_(0)

        # At first, all examples are in partition 1
        # Moments are not normalized so they can be computed cumulatively
        moments[0,1] = torch.sum(W, dim=0)
        moments[1,1] = torch.sum(W*Y, dim=0)
        moments[2,1] = torch.sum(W*Y**2, dim=0)

        risk = self._compute_risk(moments)

        best_feature = risk.argmin()
        best_risk = risk[best_feature]
        best_moment_0 = moments[0,:,best_feature,:].cpu()
        best_moment_1 = moments[1,:,best_feature,:].cpu()
        best_stump_idx = 0

        for i, row in enumerate(sorted_X_idx[:-1]):

            self._update_moments(moments, moments_update, W[row], Y[row])

            possible_stumps = ~near_zero(sorted_X[i+1] - sorted_X[i])
            if possible_stumps.any():

                risk = self._compute_risk(moments[:,:,possible_stumps,:])
                feature = risk.argmin()
                if risk[feature] < best_risk:
                    best_feature = possible_stumps.nonzero().reshape((-1))[feature]
                    best_risk = risk[feature]
                    best_moment_0 = moments[0,:,best_feature,:].cpu()
                    best_moment_1 = moments[1,:,best_feature,:].cpu()
                    best_stump_idx = i + 1

        self.confidence_rates = np.divide(best_moment_1, best_moment_0, where=best_moment_0!=0)

        return best_stump_idx, best_feature

    def _update_moments(self, moments, moments_update, weights_update, labels_update):
        moments_update[0] = weights_update
        moments_update[1] = weights_update*labels_update
        moments_update[2] = weights_update*labels_update**2

        moments[:,0] += moments_update
        moments[:,1] -= moments_update

    def _compute_risk(self, moments):
        valid_idx = ~near_zero(moments[0])
        normalized_m1 = torch.zeros_like(moments[0])
        normalized_m1[valid_idx] = moments[1][valid_idx]**2/moments[0][valid_idx]
        risk = torch.sum(torch.sum(moments[2] - normalized_m1, dim=2), dim=0)
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


def near_zero(X, atol=1E-5):
    return torch.abs(X) <= atol


@timed
def main():
    mnist = MNISTDataset.load()
    (Xtr, Ytr), (Xts, Yts) = mnist.get_train_test(center=False, reduce=False)

    # encoder = LabelEncoder.load_encodings('js_without_0', convert_to_int=True)
    # encoder = LabelEncoder.load_encodings('mario')
    encoder = OneHotEncoder(Ytr)
    # encoder = AllPairsEncoder(Ytr)

    m = 1_000
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
    # import cProfile
    # cProfile.run('main()', sort='tottime')
    main()
