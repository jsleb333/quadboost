import numpy as np
from sklearn.metrics import accuracy_score
from weak_learner import WLRidge, WLThresholdedRidge, MulticlassDecisionStump
import matplotlib.pyplot as plt

from label_encoder import LabelEncoder, OneHotEncoder, AllPairsEncoder
from mnist_dataset import MNISTDataset
from boost_iterator import BoostIterator
from utils import *


class QuadBoost:
    def __init__(self, weak_learner, encoder=None):
        """
        weak_learner (Object that defines the 'fit' method and the 'predict' method): Weak learner that generates weak predictors to be boosted on.
        encoder (LabelEncoder object, optional, default=None): Object that encodes the labels to provide an easier separation problem. If None, a one-hot encoding is used.
        """
        self.weak_learner = weak_learner
        self.encoder = encoder
        self.weak_predictors = []
        self.weak_predictors_weights = []


    def fit(self, X, Y, T=-1, f0=None, X_val=None, Y_val=None, patience=10):
        """
        X (Array of shape (n_examples, ...)): Examples.
        Y (Iterable of 'n_examples' elements): Labels for the examples X. Y is encoded with the encode_labels method if one is provided, else it is transformed as one-hot vectors.
        T (int, optional, default=-1): Number of boosting rounds. If T=-1, the algorithm will boost indefinitely, until reaching a training accuracy of 1.0, or until the training accuracy does not improve for 'patience' consecutive boosting rounds.
        f0 (Array of shape (encoding_dim,), optional, default=None): Initial prediction function. If None, f0 is set to 0.
        X_val (Array of shape (n_val, ...), optional, default=None): Validation examples. If not None, the validation accuracy will be evaluated at each boosting round.
        Y_val (Iterable of 'n_val' elements, optional, default=None): Validation labels for the examples X_val. If not None, the validation accuracy will be evaluated at each boosting round.
        patience (int, optional, default=10): Number of boosting rounds before terminating the algorithm when the training accuracy shows no improvements. If patience=None, the boosting rounds will continue until T iterations.
        """
        # Encodes the labels
        if self.encoder == None:
            self.encoder = OneHotEncoder(Y)
        encoded_Y, weights = self.encoder.encode_labels(Y)

        # Initialization
        if f0 == None:
            self.f0 = np.zeros(self.encoder.encoding_dim)
        else:
            self.f0 = f0

        residue = encoded_Y - self.f0

        train_acc = best_train_acc = 0
        rounds_since_no_improvements = 0

        # Boosting algorithm
        for boosting_round in BoostIterator(T, patience):
            residue, weak_prediction = self._boost(X, residue, weights)

            # wp_acc = acc_score(y_true=Y, y_pred=self.encoder.decode_labels(weak_prediction))
            # print('weak predictor acc:' + str(wp_acc))
            boosting_round.train_acc = self.evaluate(X, Y)
            if X_val is not None and Y_val is not None:
                boosting_round = valid_acc = self.evaluate(X_val, Y_val)
            
            print(boosting_round)
                
        # If the boosting algorithm uses the confidence of the WeakLearner as a weights instead of computing one, we set a weight of 1 for every weak predictor.
        if self.weak_predictors_weights == []:
            self.weak_predictors_weights = [np.array([1])]*len(self.weak_predictors)


    def _boost(self, X, residue, weights):
        """
        Should implements one round of boosting.
        Must return the calculated residue and the weak_prediction.
        Should append self.weak_predictors with a fitted self.weak_learner.
        """
        raise NotImplementedError


    def predict(self, X):
        encoded_Y_pred = np.zeros((X.shape[0], self.encoder.encoding_dim)) + self.f0

        for wp_weight, wp in zip(self.weak_predictors_weights, self.weak_predictors):
            encoded_Y_pred += wp_weight * wp.predict(X)

        return self.encoder.decode_labels(encoded_Y_pred)


    def evaluate(self, X, Y):
        Y_pred = self.predict(X)
        return accuracy_score(y_true=Y, y_pred=Y_pred)
    

    def visualize_coef(self):
        fig, axes = make_fig_axes(self.encoder.encoding_dim)
        coefs = self.coef_
        
        for i, ax in enumerate(axes):
            ax.imshow(coefs[i,:,:], cmap='gray_r')

        plt.get_current_fig_manager().window.showMaximized()
        plt.show()
    

    @property
    def coef_(self):
        return self._compute_coef()
    

    def _compute_coef(self):
        coefs = [wp_w.reshape(-1,1)*wp.coef_ for wp_w, wp in zip(self.weak_predictors_weights, self.weak_predictors)]
        coefs = np.sum(coefs, axis=0).reshape((self.encoder.encoding_dim,28,28))
        return coefs


class QuadBoostMH(QuadBoost):
    def __init__(self, weak_learner, encoder=None):
        """
        weak_learner (Object that defines the 'fit' method and the 'predict' method): Weak learner that generates weak predictors to be boosted on.
        encoder (LabelEncoder object, optional, default=None): Object that encodes the labels to provide an easier separation problem. If None, a one-hot encoding is used.
        """
        super().__init__(weak_learner, encoder)


    def _boost(self, X, residue, weights):
        """
        Implements one round of boosting. 
        Appends the lists of weak_predictors and of weak_predictors_weights with the fitted weak learner and its computed weight.

        X (Array of shape (n_examples, ...)): Examples.
        residue (Array of shape (n_examples, encoding_dim)): Residues to fit for the examples X.
        weights (Array of shape (n_examples, encoding_dim)): Weights of the examples X for each encoding.

        Returns the calculated residue and the weak_prediction.
        """
        weak_predictor = self.weak_learner().fit(X, residue, weights)
        weak_prediction = weak_predictor.predict(X)

        n_examples = X.shape[0]
        alpha = np.sum(weights * residue * weak_prediction, axis=0)/n_examples/np.mean(weights, axis=0)
        residue -= alpha * weak_prediction

        self.weak_predictors_weights.append(alpha)
        self.weak_predictors.append(weak_predictor)

        return residue, weak_prediction


class QuadBoostMHCR(QuadBoost):
    def __init__(self, confidence_rated_weak_learner, encoder=None):
        """
        confidence_rated_weak_learner (Object that defines the 'fit' method and the 'predict' method): Weak learner that generates confidence rated weak predictors to be boosted on.
        encoder (LabelEncoder object, optional, default=None): Object that encodes the labels to provide an easier separation problem. If None, a one-hot encoding is used.
        """
        super().__init__(confidence_rated_weak_learner, encoder)


    def _boost(self, X, residue, weights):
        """
        Implements one round of boosting. 
        Appends the lists of weak_predictors with the fitted weak learner.

        X (Array of shape (n_examples, ...)): Examples.
        residue (Array of shape (n_examples, encoding_dim)): Residues to fit for the examples X.
        weights (Array of shape (n_examples, encoding_dim)): Weights of the examples X for each encoding.

        Returns the calculated residue and the confidence_rated_weak_prediction.
        """
        confidence_rated_weak_predictor = self.weak_learner().fit(X, residue, weights)
        confidence_rated_weak_prediction = confidence_rated_weak_predictor.predict(X)

        residue -= confidence_rated_weak_prediction

        self.weak_predictors.append(confidence_rated_weak_predictor)
        self.weak_predictors_weights.append(np.array([1]))

        return residue, confidence_rated_weak_prediction

@timed
def main():
    mnist = MNISTDataset.load()
    (Xtr, Ytr), (Xts, Yts) = mnist.get_train_test(center=False, reduce=False)

    # encoder = LabelEncoder.load_encodings('js_without_0', convert_to_int=True)
    # encoder = LabelEncoder.load_encodings('mario')
    # encoder = OneHotEncoder(Ytr)
    encoder = AllPairsEncoder(Ytr)

    # weak_learner = WLThresholdedRidge(threshold=.5)
    # weak_learner = WLRidge
    weak_learner = MulticlassDecisionStump

    qb = QuadBoostMHCR(weak_learner, encoder=encoder)
    m = 100
    qb.fit(Xtr[:m], Ytr[:m], T=1000, X_val=Xts, Y_val=Yts, patience=2)
    # qb.visualize_coef()
    

if __name__ == '__main__':
    main()