import numpy as np
import sklearn as sk
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Ridge

from utils import load_mnist, to_one_hot, load_encodings, load_verbal_encodings


class QuadBoostMH:
    def __init__(self, weak_learner, labels_encoding=None, encoding_score=None):
        """
        weak_learner (Custom object that defines the 'fit' method and the 'predict' method): Weak learner that generates weak predictors to be boosted on.
        labels_encoding (Dictionary, optional, default=None): Dictionary of {label:encoding} used to encode and decode labels. 'encoding' should be an array of shape (encoding_dim,). If None, one-hot vector encoding is used.
        encoding_score (Callable, optional, default=None): Function that computes a score between a predicted encoding and a real encoding. Should support ndarrays operations. Class predictions will be made by taking the class with the highest score. If None, the score is simply the predicted encoding.
        """
        self.weak_learner = weak_learner
        self.labels_encoding = labels_encoding
        self.encoding_score = encoding_score
        if self.encoding_score == None:
            self.encoding_score = self._encoding_score


    def fit(self, X, Y, T, f0=None):
        """
        X (Array of shape (n_examples, ...)): Examples.
        Y (Iterable of length 'n_examples'): Labels for the examples X. Y is encoded with the encode_labels method if one is provided, else it is transformed as one-hot vectors.
        T (int): Number of boosting rounds.
        f0 (Array of shape (encoding_dim,), optional, default=None): Initial prediction function. If None, f0 is set to 0.
        """
        self._make_labels_to_idx(Y)
        self._make_encoding_matrix()
        self._make_weights_matrix()
        encoded_Y = self._encode_labels(Y)
        weights = self._map_weights(Y)
        n_examples = len(Y)

        if f0 == None:
            self.f0 = np.zeros(self.encoding_dim)
        else:
            self.f0 = f0

        residue = encoded_Y - self.f0
        self.alphas = []
        self.weak_predictors = []
        alpha = np.zeros(self.encoding_dim)
        for t in range(T):
            weak_predictor = self.weak_learner()
            weak_predictor.fit(X, residue, self.weights_matrix)
            weak_prediction = np.sign(weak_predictor.predict(X))
            alpha = np.sum(weights * residue * weak_prediction, axis=0)/n_examples/np.mean(weights, axis=0)
            residue -= alpha * weak_prediction

            self.alphas.append(alpha)
            self.weak_predictors.append(weak_predictor)
            print('Boosting round ' + str(t+1) + ' - train accuracy: ' + str(self.evaluate(X, Y)))
            wp_acc = accuracy_score(y_true=Y, y_pred=self._decode_labels(weak_prediction))
            print('weak predictor accuracy:' + str(wp_acc))


    def predict(self, X):
        encoded_Y_pred = np.zeros((X.shape[0], self.encoding_dim)) + self.f0
        for alpha, weak_predictor in zip(self.alphas, self.weak_predictors):
            encoded_Y_pred += alpha * np.sign(weak_predictor.predict(X))

        return self._decode_labels(encoded_Y_pred)


    def evaluate(self, X, Y):
        Y_pred = self.predict(X)
        return accuracy_score(y_true=Y, y_pred=Y_pred)


    def _make_labels_to_idx(self, Y):
        self.labels = sorted(set(Y)) # Keeps only one copy of each label.
        self.labels_to_idx = {}
        for idx, label in enumerate(self.labels):
            self.labels_to_idx[label] = idx
        
        self.n_classes = len(self.labels)


    def _make_encoding_matrix(self):
        if self.labels_encoding == None:
            self.labels_encoding = {label:2*np.eye(1, self.n_classes, k=idx)[0]-1 for label, idx in self.labels_to_idx.items()}

        self.encoding_dim = len(self.labels_encoding[self.labels[0]])
        self.encoding_matrix = np.zeros((self.n_classes, self.encoding_dim))
        for label, idx in self.labels_to_idx.items():
            self.encoding_matrix[idx] = self.labels_encoding[label]


    def _encode_labels(self, Y):
        """
        Y (Iterable of length 'n_examples'): labels of the examples.

        Returns an array of shape (n_examples, encoding_dim).
        """
        encoded_Y = -np.ones((len(Y), self.encoding_dim))
        for i, label in enumerate(Y):
            encoded_Y[i] = self.labels_encoding[label]
        
        return encoded_Y


    def _decode_labels(self, encoded_Y):
        """
        encoded_Y (Array of shape (n_examples, encoding_dim)): Encoded labels to be decoded.

        Decodes the labels by computing a score between encoded labels and the encoding matrix and taking the argmax as the class.

        Returns a list of decoded labels.
        """
        n_examples = encoded_Y.shape[0]
        scored_Y = self.encoding_score(encoded_Y) # Shape: (n_examples, n_classes)
        decoded_Y_idx = np.argmax(scored_Y, axis=1) # Shape: (n_examples,)
        decoded_Y = []
        for idx in decoded_Y_idx:
            decoded_Y.append(self.labels[idx])

        return decoded_Y


    def _encoding_score(self, encoded_Y):
        """
        encoded_Y (Array of shape (n_examples, encoding_dim))

        By default, computes the scalar product of the encoded labels with each encoding from the weighted encoding matrix.
        For one-hot vectors encoding, it is equivalent to do nothing since the encoding matrix is the identity.
        """
        weighted_encoding = self.encoding_matrix * self.weights_matrix
        score = encoded_Y.dot(weighted_encoding.T)
        return score


    def _make_weights_matrix(self):
        # if self.encoding_weights == None:
        #     self.encoding_weights = {label: np.ones(self.encoding_dim)/self.encoding_dim for label in self.labels}

        # self.weights_matrix = np.ones_like(self.encoding_matrix)
        # for label, idx in self.labels_to_idx.items():
        #     self.weights_matrix[idx] = self.encoding_weights[label]

        weights = np.abs(self.encoding_matrix)
        self.weights_matrix = weights/np.sum(weights, axis=1).reshape(-1,1)


    def _map_weights(self, Y):
        """
        Returns an array of shape (n_examples, encoding_dim) that explicits the encoding weights for each label.
        """        
        weights = np.zeros((len(Y), self.encoding_dim))
        for i, label in enumerate(Y):
            label_idx = self.labels_to_idx[label]
            weights[i] = self.weights_matrix[label_idx]
        
        return weights


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
    encodings = load_encodings('js_without_0', convert_to_int=True)
    # encodings = load_verbal_encodings('mario')
    # encodings = None

    qb = QuadBoostMH(WeakLearner, labels_encoding=encodings)

    qb.fit(Xtr, Ytr, T=3)
    acc = qb.evaluate(Xts, Yts)
    print('test accuracy', acc)
    
    wl = WeakLearner()
    Ytr_encoded = qb._encode_labels(Ytr)
    wl.fit(Xtr, Ytr_encoded)
    tr_pred = qb._decode_labels(wl.predict(Xtr))
    ts_pred = qb._decode_labels(wl.predict(Xts))
    print('WL train acc', accuracy_score(tr_pred, Ytr))
    print('WL test acc', accuracy_score(ts_pred, Yts))
