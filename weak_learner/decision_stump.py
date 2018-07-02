import numpy as np
from sklearn.metrics import accuracy_score

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
        # torch.no_grad()
        # X, Y, W = map(torch.from_numpy, [X,Y,W]) # Convert to torch tensors with shared memory
        # X_gpu, Y_gpu, W_gpu = map(lambda Z: Z.cuda(), [X,Y,W]) # Copied to GPU

        # sorted_X, sorted_X_idx = torch.sort(X_gpu, dim=0)

        sorted_X_idx = X.argsort(axis=0)
        sorted_X = X[sorted_X_idx, range(X.shape[1])]

        batch_size = X.shape[1]
        stump_idx, self.feature = self.find_stump(sorted_X[:,:batch_size],
                                                  sorted_X_idx[:,:batch_size],
                                                  Y, W)

        feature_value = lambda stump_idx: X[sorted_X_idx[stump_idx,self.feature],self.feature]
        self.stump = (feature_value(stump_idx) + feature_value(stump_idx-1))/2 if stump_idx != 0 else feature_value(stump_idx) - 1
        
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
        # All moments are not normalized so they can be computed cumulatively
        moments[0,1] = np.sum(W, axis=0)
        moments[1,1] = np.sum(W*Y, axis=0)
        moments[2,1] = np.sum(W*Y**2, axis=0)

        risk = np.sum(np.sum(moments[2] - np.divide(moments[1]**2, moments[0], where=moments[0]!=0), axis=0), axis=1)

        best_feature = risk.argmin()
        best_risk = risk[best_feature]
        best_moment_0 = moments[0,:,best_feature,:].copy()
        best_moment_1 = moments[1,:,best_feature,:].copy()
        best_stump_idx = 0

        for i, row in enumerate(sorted_X_idx[:-1]):

            # Update moments
            weights_update = W[row]
            labels_update = Y[row]
            moments_update[0] = weights_update
            moments_update[1] = weights_update*labels_update
            moments_update[2] = weights_update*labels_update**2

            moments[:,0] = moments[:,0] + moments_update
            moments[:,1] = moments[:,1] - moments_update

            possible_stumps = ~np.isclose(sorted_X[i+1] - sorted_X[i], 0)

            if possible_stumps.any():

                risk = self._compute_risk(moments[:,:,possible_stumps,:])
                feature = risk.argmin()
                if risk[feature] < best_risk:
                    best_feature = possible_stumps.nonzero()[0][feature]
                    best_risk = risk[feature]
                    best_moment_0 = moments[0,:,best_feature,:].copy()
                    best_moment_1 = moments[1,:,best_feature,:].copy()
                    best_stump_idx = i + 1
        
        self.confidence_rates = np.divide(best_moment_1, best_moment_0, where=best_moment_0!=0)

        return best_stump_idx, best_feature
    
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

    m = 60000
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
    main()