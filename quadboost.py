import numpy as np
import sklearn as sk
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Ridge

from label_encoder import LabelEncoder, OneHotEncoder
from utils import load_mnist, to_one_hot, load_encodings, load_verbal_encodings


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
        if self.encoder == None:
            self.encoder = OneHotEncoder(Y)
        encoded_Y, weights = self.encoder.encode_labels(Y)
        n_examples = len(Y)

        if f0 == None:
            self.f0 = np.zeros(self.encoder.encoding_dim)
        else:
            self.f0 = f0

        residue = encoded_Y - self.f0
        self.alphas = []
        self.weak_predictors = []
        alpha = np.zeros(self.encoder.encoding_dim)
        for t in range(T):
            weak_predictor = self.weak_learner()
            weak_predictor.fit(X, residue, weights)
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


class WeakLearner:
    def __init__(self):
        self.classifier = Ridge(alpha=1)
    
    def fit(self, X, Y, W=None):
        X = X.reshape((X.shape[0], -1))
        self.classifier.fit(X, Y)
    
    def predict(self, X):
        X = X.reshape((X.shape[0], -1))
        return self.classifier.predict(X)


if __name__ == '__main__':
    (Xtr, Ytr), (Xts, Yts) = load_mnist(60000, 10000)
    # encoder = LabelEncoder.load_encodings('js_without_0', convert_to_int=True)
    # encoder = LabelEncoder.load_encodings('mario')
    encoder = None

    qb = QuadBoostMH(WeakLearner, encoder=encoder)

    qb.fit(Xtr, Ytr, T=3)
    acc = qb.evaluate(Xts, Yts)
    print('test accuracy', acc)
    
    # wl = WeakLearner()
    # Ytr_encoded = qb._encode_labels(Ytr)
    # wl.fit(Xtr, Ytr_encoded)
    # tr_pred = qb._decode_labels(wl.predict(Xtr))
    # ts_pred = qb._decode_labels(wl.predict(Xts))
    # print('WL train acc', accuracy_score(tr_pred, Ytr))
    # print('WL test acc', accuracy_score(ts_pred, Yts))
