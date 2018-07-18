import numpy as np
from sklearn.metrics import accuracy_score
from weak_learner import WLRidge, WLThresholdedRidge, MulticlassDecisionStump
import matplotlib.pyplot as plt

from label_encoder import LabelEncoder, OneHotEncoder, AllPairsEncoder
from mnist_dataset import MNISTDataset
from boost_manager import BoostManager
from callbacks import ModelCheckpoint
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

    def fit(self, X, Y, f0=None,
            max_round_number=None, patience=None, break_on_perfect_train_acc=False,
            X_val=None, Y_val=None,
            callbacks=None,
            **weak_learner_fit_kwargs):
        """
        X (Array of shape (n_examples, ...)): Examples.
        Y (Iterable of 'n_examples' elements): Labels for the examples X. Y is encoded with the encode_labels method if one is provided, else it is transformed as one-hot vectors.
        f0 (Array of shape (encoding_dim,), optional, default=None): Initial prediction function. If None, f0 is set to 0.
        max_round_number (int, optional, default=-1): Maximum number of boosting rounds. If None, the algorithm will boost indefinitely, until reaching a perfect training accuracy (if True), or until the training accuracy does not improve for 'patience' consecutive boosting rounds (if not None).
        patience (int, optional, default=None): Number of boosting rounds before terminating the algorithm when the training accuracy shows no improvements. If None, the boosting rounds will continue until max_round_number iterations (if not None).
        break_on_perfect_train_acc (Boolean, optional, default=False): If True, it will stop the iterations if a perfect train accuracy of 1.0 is achieved.
        X_val (Array of shape (n_val, ...), optional, default=None): Validation examples. If not None, the validation accuracy will be evaluated at each boosting round.
        Y_val (Iterable of 'n_val' elements, optional, default=None): Validation labels for the examples X_val. If not None, the validation accuracy will be evaluated at each boosting round.
        callbacks (Iterable of Callback objects, optional, default=None): Callbacks objects to be called at some specific step of the training procedure to execute something. Ending conditions of the boosting iteration are handled with BreakCallbacks. If callbacks contains BreakCallbacks and terminating conditions (max_round_number, patience, break_on_perfect_train_acc) are not None, all conditions will be checked at each round and the first that is not verified will stop the iteration.
        weak_learner_fit_kwargs: Keyword arguments to pass to the fit method of the weak learner.
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

        boost_manager = BoostManager(self, callbacks)

        # Boosting algorithm
        for boosting_round in boost_manager.iterate(max_round_number,
                                                    patience,
                                                    break_on_perfect_train_acc):

            residue, weak_prediction = self._boost(X, residue, weights, **weak_learner_fit_kwargs)

            boosting_round.train_acc = self.evaluate(X, Y)
            if X_val is not None and Y_val is not None:
                boosting_round.valid_acc = self.evaluate(X_val, Y_val)

            print(boosting_round)

        # If the boosting algorithm uses the confidence of the WeakLearner as a weights instead of computing one, we set a weight of 1 for every weak predictor.
        if self.weak_predictors_weights == []:
            self.weak_predictors_weights = [np.array([1])]*len(self.weak_predictors)

    def _boost(self, X, residue, weights, **kwargs):
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

    def _boost(self, X, residue, weights, **kwargs):
        """
        Implements one round of boosting.
        Appends the lists of weak_predictors and of weak_predictors_weights with the fitted weak learner and its computed weight.

        X (Array of shape (n_examples, ...)): Examples.
        residue (Array of shape (n_examples, encoding_dim)): Residues to fit for the examples X.
        weights (Array of shape (n_examples, encoding_dim)): Weights of the examples X for each encoding.

        Returns the calculated residue and the weak_prediction.
        """
        weak_predictor = self.weak_learner().fit(X, residue, weights, **kwargs)
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

    def _boost(self, X, residue, weights, **kwargs):
        """
        Implements one round of boosting.
        Appends the lists of weak_predictors with the fitted weak learner.

        X (Array of shape (n_examples, ...)): Examples.
        residue (Array of shape (n_examples, encoding_dim)): Residues to fit for the examples X.
        weights (Array of shape (n_examples, encoding_dim)): Weights of the examples X for each encoding.

        Returns the calculated residue and the confidence_rated_weak_prediction.
        """
        confidence_rated_weak_predictor = self.weak_learner().fit(X, residue, weights, **kwargs)
        confidence_rated_weak_prediction = confidence_rated_weak_predictor.predict(X)

        residue -= confidence_rated_weak_prediction

        self.weak_predictors.append(confidence_rated_weak_predictor)
        self.weak_predictors_weights.append(np.array([1]))

        return residue, confidence_rated_weak_prediction

@timed
def main():
    ### Data loading
    mnist = MNISTDataset.load('haar_mnist.pkl')
    # mnist = MNISTDataset.load()
    (Xtr, Ytr), (Xts, Yts) = mnist.get_train_test(center=False, reduce=False)
    m = 100

    ### Choice of encoder
    # encoder = LabelEncoder.load_encodings('js_without_0', convert_to_int=True)
    # encoder = LabelEncoder.load_encodings('mario')
    encoder = OneHotEncoder(Ytr)
    # encoder = AllPairsEncoder(Ytr)

    ### Choice of weak learner
    # weak_learner = WLThresholdedRidge(threshold=.5)
    # weak_learner = WLRidge
    weak_learner = MulticlassDecisionStump()
    sorted_X, sorted_X_idx = weak_learner.sort_data(Xtr[:m])

    ### Callbacks
    ckpt = ModelCheckpoint(filename='best_test{step}.ckpt', dirname='./results', period=2,
                           save_last=True,
                           save_best_only=True)
    callbacks = [ckpt]


    qb = QuadBoostMHCR(weak_learner, encoder=encoder)
    qb.fit(Xtr[:m], Ytr[:m], max_round_number=3, patience=10,
           X_val=Xts, Y_val=Yts,
           callbacks=callbacks,
           n_jobs=1, sorted_X=sorted_X, sorted_X_idx=sorted_X_idx)
    # qb.visualize_coef()


if __name__ == '__main__':
    main()
