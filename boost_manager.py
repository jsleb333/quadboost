import numpy as np
from warnings import warn
from time import time
from callbacks import CallbackList
from callbacks import BreakOnMaxRound, BreakOnPlateau, BreakOnPerfectTrainAccuracy
from callbacks import UpdateTrainAcc


class BoostingRound:
    """
    Class that stores information about the current boosting round, such as the number of the iteration 'round_number' and the training and validation accuracies. The class contains a flag to indicate if the training accuracy have been updated for the current boosting round, and provides a formatting of the informations in the __str__ method.
    """
    def __init__(self):
        self.round_log = {'round':0,
                          'train_acc':0,
                          'valid_acc':None}
        self.train_acc_was_set_this_round = True
        self.start_time = None
        self.end_time = None

    def __str__(self):

        output = ['Boosting round {round:03d}']

        if self.train_acc_was_set_this_round:
            output.append('Train acc: {train_acc:.3f}')
        else:
            self.warn_train_acc_was_not_updated()
            output.append('Train acc: ?.???')

        if self.valid_acc is not None:
            output.append('Valid acc: {valid_acc:.3f}')

        if self.end_time is not None:
            output.append(f'Time: {self.end_time-self.start_time:.2f}s')

        return ' | '.join(output).format(**self.round_log)

    def warn_train_acc_was_not_updated(self):
        if not self.train_acc_was_set_this_round:
            warn("The train_acc attribute should be set for the iterator to work properly. Otherwise, the 'patience' mechanism will not work, and the iteration will not stop if the training accuracy reaches 1.0.")

    @property
    def train_acc(self):
        return self.round_log['train_acc']

    @train_acc.setter
    def train_acc(self, train_acc):
        self.train_acc_was_set_this_round = True
        self.round_log['train_acc'] = train_acc
        self.end_time = time()

    @property
    def valid_acc(self):
        return self.round_log['valid_acc']

    @valid_acc.setter
    def valid_acc(self, valid_acc):
        self.round_log['valid_acc'] = valid_acc

    @property
    def round_number(self):
        return self.round_log['round']

    @round_number.setter
    def round_number(self, round_number):
        self.train_acc_was_set_this_round = False # On a new round, train_acc is not yet updated.
        self.round_log['round'] = round_number
        self.start_time = time()

    def log(self):
        output = [self.round_number]
        if self.train_acc_was_set_this_round: output.append(self.train_acc)
        if self.valid_acc is not None: output.append(self.valid_acc)
        return output


class BoostManager:
    """
    Class that implements an iterator to boost in QuadBoost. The iterator yields a BoostingRound object that should be updated at each boosting round with the training accuracy. The iterator stops the iteration when:
        - the maximum number of boosting rounds has been reached (if 'max_round_number' is not -1)
        - the training accuracy did not improve for 'patience' rounds (if patience is not None)
        - the training accuracy has reached 1.0
    """
    def __init__(self, boost_model):
        """
        boost_model (QuadBoost object): Reference to the QuadBoost object to manage.
        max_round_number (int, optional, default=-1): Number of boosting rounds. If max_round_number=-1, the algorithm will boost indefinitely, until reaching a training accuracy of 1.0, or until the training accuracy does not improve for 'patience' consecutive boosting rounds.
        patience (int, optional, default=10): Number of boosting rounds before terminating the algorithm when the training accuracy shows no improvements. If patience=None, the boosting rounds will continue until max_round_number iterations.
        """
        self.boost_model = boost_model
        self.best_train_acc = -1
        self.callbacks = CallbackList([UpdateTrainAcc(self)])
        self.boosting_round = BoostingRound()

    def __iter__(self):
        return self

    def iterate(self, max_round_number=None, patience=None, break_on_perfect_train_acc=True):
        self.boosting_round.round_number = 0

        if max_round_number == None and patience is None:
            warn("Beware that the values of 'max_round_number=None' and 'patience=None' may result in an infinite loop if the algorithm does not converge to a training accuracy of 1.0.")

        if max_round_number:
            self.callbacks.append(BreakOnMaxRound(self, max_round_number=max_round_number))
        if patience:
            self.callbacks.append(BreakOnPlateau(self, patience=patience))
        if break_on_perfect_train_acc:
            self.callbacks.append(BreakOnPerfectTrainAccuracy(self))

        self.callbacks.on_fit_begin()

        return self

    def __next__(self):
        try:
            if self.boosting_round.round_number != 0:
                self.callbacks.on_boosting_round_end()

            self.callbacks.on_boosting_round_begin()

        except StopIteration:
            self.callbacks.on_fit_end()
            raise

        self.boosting_round.round_number += 1
        return self.boosting_round


if __name__ == '__main__':
    a = 0
    safe = 0
    max_round_number = 20
    patience = 0
    bi = BoostManager(max_round_number, patience)
    for br in bi:
        # if safe == 0:
        a += .1
        br.train_acc = a
        # br.valid_acc = a
        print(br)

        safe += 1
        if safe >= 100:
            break
