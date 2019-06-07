import numpy as np
import pickle as pkl
from sklearn.metrics import accuracy_score
import torch
import torch.nn.functional as F
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
    from quadboost import BoostingRound

except ModuleNotFoundError:
    from .weak_learner import *
    from .label_encoder import LabelEncoder, OneHotEncoder, AllPairsEncoder
    from .callbacks import CallbacksManagerIterator, Step
    from .callbacks import ModelCheckpoint, CSVLogger, Progression, BestRoundTrackerCallback
    from .callbacks import (BreakOnMaxStepCallback, BreakOnPerfectTrainAccuracyCallback,
                        BreakOnPlateauCallback, BreakOnZeroRiskCallback)
    from .utils import *
    from .quadboost import BoostingRound

from quadboost import QuadBoostMH, QuadBoostMHCR


class TransBoost(QuadBoostMHCR):
    """
    QuadBoostMHCR, but with a twist: every τ steps, the previous weak learners must provide a set of convolutional filters to apply to X before resuming the training.

    The weak learner should be able to choose a number of good filters to give to TransBoost. To do so, the weak learner should define a 'select_filters()' method which returns a torch.Tensor of shape (some_number_of_filters, n_channels, width, height).
    """
    def algorithm(self, *args, **kwargs):
        return TransBoostAlgorithm(*args, **kwargs)

    def fit(self, X, Y, f0=None,
            patience=None, break_on_perfect_train_acc=False,
            n_filters_per_layer=100, n_layers=3,
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

            n_filters_per_layer (int or sequence of ints, optional): Number of filters (i.e. boosting rounds) to produce before starting the next layer. If it is a sequence of ints, it corresponds to the number of filters on each layer.

            n_layers (int, optional): Number of layers before stopping the training. Is ignored if n_filters_per_layer is a sequence.

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

        self.n_filters_per_layer = n_filters_per_layer
        if isinstance(n_filters_per_layer, int):
            self.n_filters_per_layer = [n_filters_per_layer]*n_layers

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

        self.filters = []
        bank = self.weak_learner.filters.weights_generator.filter_bank
        filters_of_the_layer = None
        for n_filters in self.n_filters_per_layer:
            print('X shape:', X.shape)
            X = self._advance_to_next_layer(X, filters_of_the_layer)
            X_val = self._advance_to_next_layer(X_val, filters_of_the_layer)
            bank = self._advance_to_next_layer(bank, filters_of_the_layer)

            x_size = X.element_size() * X.nelement() / 1e9
            x_val_size = X_val.element_size() * X_val.nelement() / 1e9
            bank_size = bank.element_size() * bank.nelement() / 1e9
            # Y_size = Y.element_size() * Y.nelement() / 1e9
            # Y_val_size = Y_val.element_size() * Y_val.nelement() / 1e9
            # print(Y_val_size)
            # weights_size = weights.element_size() * weights.nelement() / 1e9
            # print(weights_size)
            # residue_size = residue.element_size() * residue.nelement() / 1e9
            # print(residue_size)
            # print(Y_size)
            print(x_size)
            print(x_val_size)
            print(bank_size)
            print(bank_size+x_val_size+x_size)

            self.weak_learner.filters.weights_generator.filter_bank = bank
            qb_algo = self.algorithm(boost_manager, self.encoder, self.weak_learner,
                                    X, Y, residue, weights, encoded_Y_pred,
                                    X_val, Y_val, encoded_Y_val_pred)
            filters_of_the_layer = qb_algo.fit(self.weak_predictors, self.weak_predictors_weights, n_filters, **weak_learner_fit_kwargs)
            print('Going to next layer...')
            filters_of_the_layer = torch.cat(filters_of_the_layer, dim=0)

            self.filters.append(filters_of_the_layer)

    def _advance_to_next_layer(self, X, filters_weights):
        if filters_weights is not None:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            nf, ch, width, height = filters_weights.shape
            padding = ((width-1)//2, (height-1)//2)
            output = F.conv2d(X, filters_weights, padding=padding)
            # output.shape -> (n_examples, n_filters, conv_height, conv_width)
            # output = F.max_pool2d(output, (2,2), ceil_mode=True)
            # F.relu(output, inplace=True)
            output = torch.tanh(output)#, inplace=True)
            return output
        else:
            return X

class TransBoostAlgorithm:
    """
    This is an implementation of the QuadBoost algorithm. It is intended to be used inside the QuadBoost class API and not as is.
    """
    def __init__(self, boost_manager, encoder, weak_learner,
                 X, Y, residue, weights, encoded_Y_pred,
                 X_val, Y_val, encoded_Y_val_pred,
                 dampening=1):
        """
        Args:
            dampening (float in ]0,1] ): Dampening factor to weight the weak predictors. Serves to slow the convergence of the algorithm so it can boost longer.
        """
        self.boost_manager = boost_manager
        self.encoder = encoder
        self.weak_learner = weak_learner

        self.X, self.Y, self.residue, self.weights = X, Y, residue, weights
        self.X_val, self.Y_val = X_val, Y_val
        self.encoded_Y_pred = encoded_Y_pred
        self.encoded_Y_val_pred = encoded_Y_val_pred
        self.dampening = np.array([dampening])

    def fit(self, weak_predictors, weak_predictors_weights, n_filters, **weak_learner_fit_kwargs):
        """
        Execute the algorithm.
        Appends the weak_predictors and weak_predictors_weights lists with the fitted weak learners.

        Args:
            weak_predictors (list): Reference to the list of weak_predictors of the model.
            weak_predictors_weights (list): Reference to the list of weak_predictors_weights of the model.
            **weak_learner_fit_kwargs: Keyword arguments needed to fit the weak learner.

        Returns filters, a list of torch.Tensors.
        """
        filters = []
        with self.boost_manager: # boost_manager handles callbacks and terminating conditions
            for boosting_round in self.boost_manager:

                weak_predictor = self.weak_learner().fit(self.X, self.residue, self.weights,                                                 **weak_learner_fit_kwargs)
                weak_prediction = weak_predictor.predict(self.X)

                weak_predictor_weight = self._compute_weak_predictor_weight(weak_prediction)
                weighted_weak_prediction = weak_predictor_weight * weak_prediction
                self.residue -= weighted_weak_prediction

                weak_predictors_weights.append(weak_predictor_weight)
                weak_predictors.append(weak_predictor)

                self._evaluate_round(boosting_round, weighted_weak_prediction, weak_predictor, weak_predictor_weight)

                filters.append(weak_predictor.select_filters())
                if np.sum([f.shape[0] for f in filters]) >= n_filters:
                    self.boost_manager.callbacks.on_step_end()
                    break
        return filters

    def _compute_weak_predictor_weight(self, weak_prediction):
        return self.dampening

    def _evaluate_round(self, boosting_round, weighted_weak_prediction, weak_predictor, weak_predictor_weight):
        self.encoded_Y_pred += weighted_weak_prediction
        Y_pred = self.encoder.decode_labels(self.encoded_Y_pred)
        boosting_round.train_acc = accuracy_score(y_true=self.Y, y_pred=Y_pred)
        boosting_round.risk = np.sum(self.weights * self.residue**2)

        if not (self.X_val is None or self.Y_val is None or self.encoded_Y_val_pred is None):
            self.encoded_Y_val_pred += weak_predictor_weight * weak_predictor.predict(self.X_val)
            Y_val_pred = self.encoder.decode_labels(self.encoded_Y_val_pred)
            boosting_round.valid_acc = accuracy_score(y_true=self.Y_val, y_pred=Y_val_pred)


@timed
def main():
    import torch
    seed = 97
    torch.manual_seed(seed)
    np.random.seed(seed)

    ### Data loading
    data = MNISTDataset.load()
    # data = CIFAR10Dataset.load()
    (Xtr, Ytr), (Xts, Yts) = data.get_train_test(center=True, reduce=True)
    m = 2000
    m_val = 100

    X_val, Y_val = Xtr[m:m+m_val], Ytr[m:m+m_val]
    bank = Xtr[m+m_val:2*m+m_val]
    Xtr, X_val, Xts = SparseRidgeRC.format_data(Xtr), SparseRidgeRC.format_data(X_val),SparseRidgeRC.format_data(Xts)
    X, Y = Xtr[:m], Ytr[:m]

    ### Choice of encoder
    encoder = OneHotEncoder(Ytr)

    ### Choice of weak learner
    f_gen = WeightFromBankGenerator(filter_bank=bank,
                                    filters_shape=(5,5),
                                    filter_processing=center_weight)
    filters = Filters(n_filters=10,
                      weights_generator=f_gen,
                      maxpool_shape=(-1,7,7))
    # Xtr, X_val, Xts = Xtr.to('cuda'), X_val.to('cuda'), Xts.to('cuda')
    weak_learner = SparseRidgeRC(filters=filters, top_k_filters=1)

    ### Callbacks
    zero_risk = BreakOnZeroRiskCallback()
    callbacks = [zero_risk]

    ### Fitting the model
    tb = TransBoost(weak_learner, encoder=encoder, dampening=1)
    # tb = QuadBoostMHCR(weak_learner, encoder=encoder, dampening=1)
    tb.fit(X, Y, patience=10,
            # n_filters_per_layer=(30,),
            n_filters_per_layer=(10,10,10),
            # max_round_number=9,
            X_val=X_val, Y_val=Y_val,
            callbacks=callbacks,
            )
    print(f'Best round recap:\nBoosting round {tb.best_round.step_number+1:03d} | Train acc: {tb.best_round.train_acc:.3%} | Valid acc: {tb.best_round.valid_acc:.3%} | Risk: {tb.best_round.risk:.3f}')
    # print(f'Test accuracy on best model: {tb.evaluate(Xts, Yts):.3%}')
    # print(f'Test accuracy on last model: {tb.evaluate(Xts, Yts, mode="last"):.3%}')


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO, style='{', format='[{levelname}] {message}')

    from datasets import MNISTDataset, CIFAR10Dataset
    main()
