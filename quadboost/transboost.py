import numpy as np
import pickle as pkl
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import logging

import sys, os
sys.path.append(os.getcwd())

try:
    from weak_learner import *
    from label_encoder import LabelEncoder, OneHotEncoder, AllPairsEncoder
    from callbacks import CallbacksManagerIterator, Step
    from callbacks import ModelCheckpoint, CSVLogger, Progression, BestRoundTrackerCallback
    from callbacks import (BreakOnMaxStepCallback, BreakOnPerfectTrainAccuracyCallback,
                        BreakOnPlateauCallback, BreakOnZeroRiskCallback)
    from utils import *

except ModuleNotFoundError:
    from .weak_learner import *
    from .label_encoder import LabelEncoder, OneHotEncoder, AllPairsEncoder
    from .callbacks import CallbacksManagerIterator, Step
    from .callbacks import ModelCheckpoint, CSVLogger, Progression, BestRoundTrackerCallback
    from .callbacks import (BreakOnMaxStepCallback, BreakOnPerfectTrainAccuracyCallback,
                        BreakOnPlateauCallback, BreakOnZeroRiskCallback)
    from .utils import *

from quadboost import QuadBoostMH, QuadBoostMHCR


class TransBoost(QuadBoostMHCR):
    """
    QuadBoostMHCR, but with a twist: every τ steps, the previous weak learners must provide a set of convolutional filters to apply to X before resuming the training.
    """
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

        if not any(isinstance(callback, BestRoundTrackerCallback) for callback in callbacks):
            if X_val is not None and Y_val is not None:
                callbacks.append(BestRoundTrackerCallback(quantity='valid_acc'))
            else:
                callbacks.append(BestRoundTrackerCallback(quantity='train_acc'))

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


