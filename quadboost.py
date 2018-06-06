import numpy as np
import sklearn as sk
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Ridge

from label_encoder import LabelEncoder, OneHotEncoder, AllPairsEncoder
from mnist_dataset import MNISTDataset

class QuadBoostMH:
    def __init__(self, weak_learner, encoder=None):
        """
        weak_learner (Object that defines the 'fit' method and the 'predict' method): Weak learner that generates weak predictors to be boosted on.
        encoder (LabelEncoder object, optional, default=None): Object that encodes the labels to provide an easier separation problem. If None, a one-hot encoding is used.
        """
        self.weak_learner = weak_learner
        self.encoder = encoder


    def fit(self, X, Y, T, f0=None):
        """
        X (Array of shape (n_examples, ...)): Examples.
        Y (Iterable of length 'n_examples'): Labels for the examples X. Y is encoded with the encode_labels method if one is provided, else it is transformed as one-hot vectors.
        T (int): Number of boosting rounds.
        f0 (Array of shape (encoding_dim,), optional, default=None): Initial prediction function. If None, f0 is set to 0.
        """
        # Encodes the labels
        if self.encoder == None:
            self.encoder = OneHotEncoder(Y)
        encoded_Y, weights = self.encoder.encode_labels(Y)
        n_examples = len(Y)

        # Initialization
        if f0 == None:
            self.f0 = np.zeros(self.encoder.encoding_dim)
        else:
            self.f0 = f0

        residue = encoded_Y - self.f0
        self.alphas = []
        self.weak_predictors = []

        # Boosting algorithm
        for t in range(T):
            weak_predictor = self.weak_learner().fit(X, residue, weights)
            weak_prediction = np.sign(weak_predictor.predict(X))

            alpha = np.sum(weights * residue * weak_prediction, axis=0)/n_examples/np.mean(weights, axis=0)
            residue -= alpha * weak_prediction

            self.alphas.append(alpha)
            self.weak_predictors.append(weak_predictor)
            print('Boosting round ' + str(t+1) + ' - train accuracy: ' + str(self.evaluate(X, Y)))
            wp_acc = accuracy_score(y_true=Y, y_pred=self.encoder.decode_labels(weak_prediction))
            print('weak predictor accuracy:' + str(wp_acc))


    def predict(self, X):
        encoded_Y_pred = np.zeros((X.shape[0], self.encoder.encoding_dim)) + self.f0

        for alpha, weak_predictor in zip(self.alphas, self.weak_predictors):
            encoded_Y_pred += alpha * np.sign(weak_predictor.predict(X))

        return self.encoder.decode_labels(encoded_Y_pred)


    def evaluate(self, X, Y):
        Y_pred = self.predict(X)
        return accuracy_score(y_true=Y, y_pred=Y_pred)


class WeakLearner(Ridge):
    def __init__(self, *args, alpha=1, encoder=None, fit_intercept=False, **kwargs):
        super().__init__(*args, alpha=alpha, fit_intercept=fit_intercept, **kwargs)
        self.encoder = encoder
    
    def fit(self, X, Y, W=None, **kwargs):
        X = X.reshape((X.shape[0], -1))
        if self.encoder != None:
            Y, W = self.encoder.encode_labels(Y)
        return super().fit(X, Y, **kwargs)
    
    def predict(self, X, **kwargs):
        X = X.reshape((X.shape[0], -1))
        return super().predict(X, **kwargs)

    def evaluate(self, X, Y):
        Y_pred = self.predict(X)
        if self.encoder != None:
            Y_pred = self.encoder.decode_labels(Y_pred)
        return accuracy_score(y_true=Y, y_pred=Y_pred)


def main():
    mnist = MNISTDataset.load()
    (Xtr, Ytr), (Xts, Yts) = mnist.get_train_test(center=True, reduce=True)

    # encoder = LabelEncoder.load_encodings('js_without_0', convert_to_int=True)
    encoder = LabelEncoder.load_encodings('mario')
    # encoder = OneHotEncoder(Ytr)
    # encoder = AllPairsEncoder(Ytr)


    qb = QuadBoostMH(WeakLearner, encoder=encoder)
    qb.fit(Xtr, Ytr, T=3)
    acc = qb.evaluate(Xts, Yts)
    print('QB test acc:', acc)
    
    # wl = WeakLearner(encoder=encoder)
    # wl.fit(Xtr, Ytr)
    # print('WL train acc:', wl.evaluate(Xtr, Ytr))
    # print('WL test acc:', wl.evaluate(Xts, Yts))
    # print(wl.classifier.intercept_)

if __name__ == '__main__':
    from time import time
    t = time()
    try:
        main()
    except:
        print('\nExecution terminated after {:.2f} seconds.\n'.format(time()-t))
        raise
    print('\nExecution completed in {:.2f} seconds.\n'.format(time()-t))