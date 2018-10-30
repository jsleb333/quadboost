import numpy as np
import pickle as pkl
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import logging

from weak_learner import *
from label_encoder import LabelEncoder, OneHotEncoder, AllPairsEncoder
from mnist_dataset import MNISTDataset
from callbacks import CallbacksManagerIterator, Step
from callbacks import ModelCheckpoint, CSVLogger, Progression
from callbacks import BreakOnMaxStepCallback, BreakOnPerfectTrainAccuracyCallback, BreakOnZeroRiskCallback, BreakOnPlateauCallback
from utils import *


class _QuadBoost:
    """
    QuadBoost is a boosting algorithm based on the squared loss. Provided with a (weak) learner, the model builds upon a collection of them to be able to make strong predictions. The algorithm has strong guarantees and a quadratic convergence.
    
    AdaBoost is another known algorithm which rely on boosting weak learners. As opposed to QuadBoost, it uses the exponential loss. Indeed, using the squared loss provides many advantages, such as having an exact, solvable minimum at each iteration.
    """
    def __init__(self, weak_learner, encoder=None):
        """
        Args:
            weak_learner (Object that defines the 'fit' method and the 'predict' method): Weak learner that generates weak predictors to be boosted on.
            encoder (LabelEncoder object, optional): Object that encodes the labels to provide an easier separation problem. If None, a one-hot encoding is used.
        """
        self.weak_learner = weak_learner
        self.encoder = encoder

    def algorithm(self, *args, **kwargs):
        return NotImplementedError 

    def fit(self, X, Y, f0=None,
            max_round_number=None, patience=None, break_on_perfect_train_acc=False,
            X_val=None, Y_val=None,
            callbacks=None,
            **weak_learner_fit_kwargs):
        """
        Function that fits the model to the data.

        The function is split into two parts: the first prepare the data and the callbacks, the second, done in _fit, actually executes the algorithm. The iteration and the callbacks are handled by a CallbacksManagerIterator.

        Args:
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

        Returns self.
        """
        # Encodes the labels
        if self.encoder == None:
            self.encoder = OneHotEncoder(Y)
        encoded_Y, weights = self.encoder.encode_labels(Y)

        # Initialization
        self.weak_predictors = []
        self.weak_predictors_weights = []

        if f0 == None:
            self.f0 = np.zeros(self.encoder.encoding_dim)
        else:
            self.f0 = f0

        residue = encoded_Y - self.f0

        # Callbacks
        if callbacks is None:
            callbacks = [Progression()]
        elif not any(isinstance(callback, Progression) for callback in callbacks):
            callbacks.append(Progression())

        if break_on_perfect_train_acc:
            callbacks.append(BreakOnPerfectTrainAccuracyCallback())
        if max_round_number:
            callbacks.append(BreakOnMaxStepCallback(max_step_number=max_round_number))
        if patience:
            callbacks.append(BreakOnPlateauCallback(patience=patience))

        self.callbacks = callbacks
        self._fit(X, Y, residue, weights, X_val, Y_val, **weak_learner_fit_kwargs)

        return self
    
    def _fit(self, X, Y, residue, weights, X_val, Y_val, **weak_learner_fit_kwargs):

        encoded_Y_pred = self.predict_encoded(X)
        encoded_Y_val_pred = self.predict_encoded(X_val) if X_val is not None else None

        starting_round = BoostingRound(len(self.weak_predictors))
        boost_manager = CallbacksManagerIterator(self, self.callbacks, starting_round)

        qb_algo = self.algorithm(boost_manager, self.encoder, self.weak_learner,
                                 X, Y, residue, weights, encoded_Y_pred,
                                 X_val, Y_val, encoded_Y_val_pred)
        qb_algo.fit(self.weak_predictors, self.weak_predictors_weights, **weak_learner_fit_kwargs)

    def resume_fit(self, X, Y, X_val=None, Y_val=None, max_round_number=None, **weak_learner_fit_kwargs):
        """
        Function to resume a previous fit uncompleted, with the same callbacks and ending conditions. See 'fit' for a description of the arguments.

        The condition on the maximum number of round can be modified or added by specifying max_round_number.

        Returns self.
        """
        if not hasattr(self, 'weak_predictors'):
            logging.error("Can't resume fit if no previous fitting made. Use 'fit' instead.")
            return self

        if max_round_number:
            for callback in self.callbacks:
                if isinstance(callback, BreakOnMaxStepCallback):
                    callback.max_step_number = max_round_number
                    break
            else:
                self.callbacks.append(BreakOnMaxStepCallback(max_round_number))

        encoded_Y, weights = self.encoder.encode_labels(Y)

        residue = encoded_Y - self.f0 - self.predict_encoded(X)

        self._fit(X, Y, residue, weights, X_val, Y_val, **weak_learner_fit_kwargs)

        return self

    def predict(self, X):
        """
        Returns the predicted labels of the given sample.

        Args:
            X (Array of shape(n_examples, ...)): Examples to predict.

        Returns Y_pred (Array of shape (n_examples))
        """
        return self.encoder.decode_labels(self.predict_encoded(X))

    def predict_encoded(self, X):
        """
        Returns the predicted encoded labels of the given sample. Can be decoded with the LabelEncoder.decode_labels() method.

        Args:
            X (Array of shape(n_examples, ...)): Examples to predict.

        Returns encoded_Y_pred (Array of shape (n_examples, encoding_dim))
        """
        encoded_Y_pred = np.zeros((X.shape[0], self.encoder.encoding_dim)) + self.f0

        for wp_weight, wp in zip(self.weak_predictors_weights, self.weak_predictors):
            encoded_Y_pred += wp_weight * wp.predict(X)
        
        return encoded_Y_pred

    def evaluate(self, X, Y, return_risk=False):
        """
        Evaluates the accuracy of the classifier given a sample and its labels.

        Args:
            X (Array of shape(n_examples, ...)): Examples to predict.
            Y (Array of shape (n_examples)): True labels.
            return_risk (bool, optional): If True, additionally returns the (non normalized) risk of the examples.

        Returns the accuracy (float) or a tuple of (accuracy (float), risk (float))
        """
        encoded_Y_pred = self.predict_encoded(X)
        Y_pred = self.encoder.decode_labels(encoded_Y_pred)

        accuracy = accuracy_score(y_true=Y, y_pred=Y_pred)

        if return_risk:
            encoded_Y, W = self.encoder.encode_labels(Y)
            risk = np.sum(W * (encoded_Y - self.f0 - encoded_Y_pred)**2)

        return accuracy if not return_risk else (accuracy, risk)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as file:
            model = pkl.load(file)
        return model


class QuadBoostMH(_QuadBoost):
    __doc__ = _QuadBoost.__doc__

    def algorithm(self, *args, **kwargs):
        return QuadBoostMHAlgorithm(*args, **kwargs)


class QuadBoostMHCR(_QuadBoost):
    __doc__ = _QuadBoost.__doc__

    def __init__(self, confidence_rated_weak_learner, encoder=None, dampening=1):
        """
        Args:
            confidence_rated_weak_learner (Object that defines the 'fit' method and the 'predict' method): Weak learner that generates confidence rated weak predictors to be boosted on.
            encoder (LabelEncoder object, optional, default=None): Object that encodes the labels to provide an easier separation problem. If None, a one-hot encoding is used.
            dampening (float in ]0,1] ): Dampening factor to weight the weak predictors. Serves to slow the convergence of the algorithm so it can boost longer.
        """
        super().__init__(confidence_rated_weak_learner, encoder)
        self.dampening = dampening

    def algorithm(self, *args, **kwargs):
        return QuadBoostMHCRAlgorithm(*args, dampening=self.dampening, **kwargs)


class _QuadBoostAlgorithm:
    """
    This is an implementation of the QuadBoost algorithm. It is intended to be used inside the QuadBoost class API and not as is.
    """
    def __init__(self, boost_manager, encoder, weak_learner,
                 X, Y, residue, weights, encoded_Y_pred,
                 X_val, Y_val, encoded_Y_val_pred):
        self.boost_manager = boost_manager
        self.encoder = encoder
        self.weak_learner = weak_learner

        self.X, self.Y, self.residue, self.weights = X, Y, residue, weights
        self.X_val, self.Y_val = X_val, Y_val
        self.encoded_Y_pred = encoded_Y_pred
        self.encoded_Y_val_pred = encoded_Y_val_pred

    def fit(self, weak_predictors, weak_predictors_weights, **weak_learner_fit_kwargs):
        """
        Execute the algorithm.
        Appends the weak_predictors and weak_predictors_weights lists with the fitted weak learners.

        Args:
            weak_predictors (list): Reference to the list of weak_predictors of the model.
            weak_predictors_weights (list): Reference to the list of weak_predictors_weights of the model.
            **weak_learner_fit_kwargs: Keyword arguments needed to fit the weak learner.
        
        Returns None.
        """
        with self.boost_manager:
            for boosting_round in self.boost_manager:

                weak_predictor = self.weak_learner().fit(self.X, self.residue, self.weights,                                                 **weak_learner_fit_kwargs)
                weak_prediction = weak_predictor.predict(self.X)

                weak_predictor_weight = self._compute_weak_predictor_weight(weak_prediction)
                weighted_weak_prediction = weak_predictor_weight * weak_prediction
                self.residue -= weighted_weak_prediction

                weak_predictors_weights.append(weak_predictor_weight)
                weak_predictors.append(weak_predictor)

                self._evaluate_round(boosting_round, weighted_weak_prediction, weak_predictor, weak_predictor_weight)

    def _compute_weak_predictor_weight(self, weights, residue, weak_prediction):
        raise NotImplementedError
    
    def _evaluate_round(self, boosting_round, weighted_weak_prediction, weak_predictor, weak_predictor_weight):
        self.encoded_Y_pred += weighted_weak_prediction
        Y_pred = self.encoder.decode_labels(self.encoded_Y_pred)
        boosting_round.train_acc = accuracy_score(y_true=self.Y, y_pred=Y_pred)
        boosting_round.risk = np.sum(self.weights * self.residue**2)

        if not (self.X_val is None or self.Y_val is None or self.encoded_Y_val_pred is None):
            self.encoded_Y_val_pred += weak_predictor_weight * weak_predictor.predict(self.X_val)
            Y_val_pred = self.encoder.decode_labels(self.encoded_Y_val_pred)
            boosting_round.valid_acc = accuracy_score(y_true=self.Y_val, y_pred=Y_val_pred)


class QuadBoostMHAlgorithm(_QuadBoostAlgorithm):
    def _compute_weak_predictor_weight(self, weak_prediction):
        n_examples = self.weights.shape[0]
        return np.sum(self.weights*self.residue*weak_prediction, axis=0) / (n_examples*np.mean(self.weights, axis=0))


class QuadBoostMHCRAlgorithm(_QuadBoostAlgorithm):
    def __init__(self, *args, dampening=1, **kwargs):

        """
        Args:
            dampening (float in ]0,1] ): Dampening factor to weight the weak predictors. Serves to slow the convergence of the algorithm so it can boost longer.
        """
        super().__init__(*args, **kwargs)
        self.dampening = np.array([dampening])

    def _compute_weak_predictor_weight(self, weak_prediction):
        return self.dampening


class BoostingRound(Step):
    """
    Class that stores information about the current boosting round like the the round number and the training and validation accuracies. Used by the CallbacksManagerIterator in the _QuadBoostAlgorithm.fit method.
    """
    def __init__(self, round_number=0):
        super().__init__(step_number=round_number)
        self.train_acc = None
        self.valid_acc = None
        self.risk = None


@timed
def main():
    import torch
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    ### Data loading
    # mnist = MNISTDataset.load('haar_mnist.pkl')
    # mnist = MNISTDataset.load('filtered_mnist.pkl')
    mnist = MNISTDataset.load()
    (Xtr, Ytr), (Xts, Yts) = mnist.get_train_test(center=True, reduce=True)
    m = 60_000

    ### Choice of encoder
    # encoder = LabelEncoder.load_encodings('js_without_0', convert_to_int=True)
    # encoder = LabelEncoder.load_encodings('mario')
    # encoder = LabelEncoder.load_encodings('ideal_mnist', convert_to_int=True)
    encoder = OneHotEncoder(Ytr)
    # encoder = AllPairsEncoder(Ytr)

    ### Choice of weak learner
    # weak_learner = WLThresholdedRidge(threshold=.5)
    # weak_learner = WLRidge
    weak_learner = RandomFilters(n_filters=1, kernel_size=(5,5), init_filters='from_data')
    # weak_learner = MulticlassDecisionTree(max_n_leaves=4)
    # weak_learner = MulticlassDecisionStump
    # sorted_X, sorted_X_idx = weak_learner.sort_data(Xtr[:m])

    ### Callbacks
    # filename = 'haar_onehot_ds_'
    # filename = 'ideal_mnist_ds_'
    filename = 'test_dampening=.9'
    ckpt = ModelCheckpoint(filename=filename+'_{round}.ckpt', dirname='./results', save_last=True)
    logger = CSVLogger(filename=filename+'_log.csv', dirname='./results/log')
    zero_risk = BreakOnZeroRiskCallback()
    callbacks = [ckpt,
                logger,
                zero_risk,
                ]

    ### Fitting the model
    qb = QuadBoostMHCR(weak_learner, encoder=encoder, dampening=.9)
    qb.fit(Xtr[:m], Ytr[:m], max_round_number=None, patience=10,
            X_val=Xts, Y_val=Yts,
            callbacks=callbacks,
            # n_jobs=4, sorted_X=sorted_X, sorted_X_idx=sorted_X_idx,
            )
    ### Or resume fitting a model
    # qb = QuadBoostMHCR.load('results/test_dampening=.9_12.ckpt')
    # qb.resume_fit(Xtr[:m], Ytr[:m],
    #               X_val=Xts, Y_val=Yts,
    #               max_round_number=40,
    #             #   n_jobs=1, sorted_X=sorted_X, sorted_X_idx=sorted_X_idx,
    #               )
if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO, style='{', format='[{levelname}] {message}')
    main()
