import numpy as np
import pickle as pkl
from sklearn.metrics import accuracy_score
from weak_learner import WLRidge, WLThresholdedRidge, MulticlassDecisionStump
import matplotlib.pyplot as plt
import logging

from label_encoder import LabelEncoder, OneHotEncoder, AllPairsEncoder
from mnist_dataset import MNISTDataset
from callbacks import BoostManager, ModelCheckpoint, CSVLogger
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
            starting_round_number=0,
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
        starting_round_number (int, optional): Number of the round the iteration should start. This is mainly used to resume a fit process that was interrupted.
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

        with BoostManager(self, callbacks) as boost_manager:
            # Boosting algorithm
            for boosting_round in boost_manager.iterate(max_round_number,
                                                        patience,
                                                        break_on_perfect_train_acc,
                                                        starting_round_number):

                residue = self._boost(X, residue, weights, **weak_learner_fit_kwargs)

                boosting_round.train_acc = self.evaluate(X, Y)
                if X_val is not None and Y_val is not None:
                    boosting_round.valid_acc = self.evaluate(X_val, Y_val)

        return self
    
    def _boost(self, X, residue, weights, **kwargs):
        """
        Implements one round of boosting.
        Appends the lists of weak_predictors and of weak_predictors_weights with the fitted weak learner and its computed weight.

        Args:
            X (Array of shape (n_examples, ...)): Examples.
            residue (Array of shape (n_examples, encoding_dim)): Residues to fit for the examples X.
            weights (Array of shape (n_examples, encoding_dim)): Weights of the examples X for each encoding.
            kwargs: Keyword arguments to be passed to the weak learner fit method.

        Returns the calculated residue.
        """
        weak_predictor = self.weak_learner().fit(X, residue, weights, **kwargs)
        weak_prediction = weak_predictor.predict(X)

        weak_predictor_weight = self._compute_weak_predictor_weight(weights, residue, weak_prediction)
        residue -= weak_predictor_weight * weak_prediction

        self.weak_predictors_weights.append(weak_predictor_weight)
        self.weak_predictors.append(weak_predictor)

        return residue
    
    def _compute_weak_predictor_weight(self, weights, residue, weak_prediction):
        raise NotImplementedError

    def resume_fit(self, X, Y, f0=None, X_val=None, Y_val=None, **weak_learner_fit_kwargs):
        try:
<<<<<<< HEAD
            self.callbacks
=======
            return self.fit(X, Y, f0,
                            X_val=X_val, Y_val=Y_val,
                            callbacks=self.callbacks,
                            starting_round_number=len(self.weak_predictors),
                            **weak_learner_fit_kwargs)
>>>>>>> fc7342d043797ff09e05de8bd7f36a02a6190f0a
        except AttributeError:
            logging.error("Can't resume fit if previous training did not end on an exception. Use 'fit' instead.")
            return
        
        if self.encoder == None:
            self.encoder = OneHotEncoder(Y)
        encoded_Y, weights = self.encoder.encode_labels(Y)

        # Initialization
        if f0 == None:
            self.f0 = np.zeros(self.encoder.encoding_dim)
        else:
            self.f0 = f0

        residue = encoded_Y - self.f0
        for predictor, predictor_weight in zip(self.weak_predictors, self.weak_predictors_weights):
            residue -= predictor_weight * predictor.predict(X)

        n_pred = len(self.weak_predictors)
        with BoostManager(self, self.callbacks) as boost_manager:
            for boosting_round in boost_manager.iterate(starting_round_number=n_pred):
                residue = self._boost(X, residue, weights, **weak_learner_fit_kwargs)
                
                boosting_round.train_acc = self.evaluate(X, Y)
                if X_val is not None and Y_val is not None:
                    boosting_round.valid_acc = self.evaluate(X_val, Y_val)

        return self

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

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as file:
            model = pkl.load(file)
        return model


class QuadBoostMH(QuadBoost):
    def __init__(self, weak_learner, encoder=None):
        """
        weak_learner (Object that defines the 'fit' method and the 'predict' method): Weak learner that generates weak predictors to be boosted on.
        encoder (LabelEncoder object, optional, default=None): Object that encodes the labels to provide an easier separation problem. If None, a one-hot encoding is used.
        """
        super().__init__(weak_learner, encoder)
    
    def _compute_weak_predictor_weight(self, weights, residue, weak_prediction):
        n_examples = weights.shape[0]
        return np.sum(weights*residue*weak_prediction, axis=0)/n_examples/np.mean(weights, axis=0)


class QuadBoostMHCR(QuadBoost):
    def __init__(self, confidence_rated_weak_learner, encoder=None):
        """
        confidence_rated_weak_learner (Object that defines the 'fit' method and the 'predict' method): Weak learner that generates confidence rated weak predictors to be boosted on.
        encoder (LabelEncoder object, optional, default=None): Object that encodes the labels to provide an easier separation problem. If None, a one-hot encoding is used.
        """
        super().__init__(confidence_rated_weak_learner, encoder)

    def _compute_weak_predictor_weight(self, weights, residue, weak_prediction):
        return np.array([1])

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
    # filename = 'haar_onehot_ds_'
    filename = 'test'
    ckpt = ModelCheckpoint(filename=filename+'{step}.ckpt', dirname='./results', save_last=True)
    logger = CSVLogger(filename=filename+'log.csv', dirname='./results/log')
    callbacks = [ckpt,
                logger,
                ]

    ### Fitting the model
    qb = QuadBoostMHCR(weak_learner, encoder=encoder)
    qb.fit(Xtr[:m], Ytr[:m], max_round_number=3, patience=10,
            X_val=Xts, Y_val=Yts,
            callbacks=callbacks,
<<<<<<< HEAD
            n_jobs=1, sorted_X=sorted_X, sorted_X_idx=sorted_X_idx)
    ### Or resume fitting a model
    # qb = QuadBoostMHCR.load('results/test1_exception_exit.ckpt')
    # qb.resume_fit(Xtr[:m], Ytr[:m],
    #               X_val=Xts, Y_val=Yts,
    #               n_jobs=1, sorted_X=sorted_X, sorted_X_idx=sorted_X_idx)
=======
            n_jobs=4, sorted_X=sorted_X, sorted_X_idx=sorted_X_idx)
    # qb = QuadBoostMHCR.load('results/haar_onehot_ds_126_exception_exit.ckpt')
    # qb.resume_fit(Xtr, Ytr, X_val=Xts, Y_val=Yts, n_jobs=4, sorted_X=sorted_X, sorted_X_idx=sorted_X_idx)
>>>>>>> fc7342d043797ff09e05de8bd7f36a02a6190f0a

if __name__ == '__main__':
    import logging
    logging.basicConfig(level=30, style='{', format='[{levelname}] {message}')
    main()
