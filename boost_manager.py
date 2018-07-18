import numpy as np
from warnings import warn
from time import time
from callbacks import CallbackList
from callbacks import BreakOnMaxRound, BreakOnPlateau, BreakOnPerfectTrainAccuracy
from callbacks import UpdateTrainAcc


class BoostingRound:
    """
    Class that stores information about the current boosting round, such as the number of the iteration 'round_number' and the training and validation accuracies. The class contains a flag to indicate if the training accuracy have been updated for the current boosting round, and provides a formatting of the informations in the __str__ method. It also provides a 'log' method to easily follow the progression of the boosting procedure.
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
    Class that manages the QuadBoost algorithm. Its goal is 2 sided:
        - Handles callbacks at 4 moments in the procedures: on_fit_begin, on_fit_end, on_boosting_round_begin, on_boosting_round_end;
        - Handles iteration of the boosting algorithm since these are closely related to callbacks.

    The iteration is managed with BreakCallbacks which raise a StopIteration exception on_boosting_round_begin or on_boosting_round_end when a condition is not satisfied.
    An iteration procedure can be launched by calling the 'iterate' method.

    Boosting rounds info (training accuracy, validation accuracy, round number and time) are handle in a BoostingRound object which is returned by the iterator.
    """
    def __init__(self, model=None, callbacks=None):
        """
        model (QuadBoost object, optional): Reference to the QuadBoost object to manage. If the callbacks do not use the model, it can be omitted.
        callbacks (Iterable of Callbacks objects, optional): Callbacks handles functions to call at specific time in the program. Usage examples: stop the iteration or save the model or the logs.
        """
        self.model = model
        self.callbacks = CallbackList(manager=self, callbacks=callbacks or [])
        self.boosting_round = BoostingRound()

    def __iter__(self):
        return self

    def iterate(self, max_round_number=None, patience=None, break_on_perfect_train_acc=True):
        """
        Initialize an iteration procedure to boost in QuadBoost. The iterator is itself and yields a BoostingRound object that should be updated at each boosting round with the training accuracy. The iterator stops the iteration when:
            - the maximum number of boosting rounds has been reached (if 'max_round_number' is not None)
            - the training accuracy did not improve for 'patience' rounds (if patience is not None)
            - the training accuracy has reached 1.0
        Break conditions are handled through callbacks.

        The callback 'on_fit_begin' is called here.

        Args:
            max_round_number (int, optional, default=-1): Number of boosting rounds. If max_round_number=-1, the algorithm will boost indefinitely, until reaching a training accuracy of 1.0, or until the training accuracy does not improve for 'patience' consecutive boosting rounds.
            patience (int, optional, default=10): Number of boosting rounds before terminating the algorithm when the training accuracy shows no improvements. If patience=None, the boosting rounds will continue until max_round_number iterations.
            break_on_perfect_train_acc (Boolean, optional, default=True): If True, iteration will stop when a perfect training accuracy is reached.
        """
        self.boosting_round.round_number = 0

        if max_round_number:
            self.callbacks.append(BreakOnMaxRound(max_round_number=max_round_number))
        if patience:
            self.callbacks.append(BreakOnPlateau(patience=patience))
        if break_on_perfect_train_acc:
            self.callbacks.append(BreakOnPerfectTrainAccuracy())

        if self.callbacks.break_callbacks == []:
            warn("The algorithm will loop indefinitelty since no break conditions were given.")

        self.callbacks.on_fit_begin()

        return self

    def __next__(self):
        """
        Steps to next iteration. Callbacks 'on_boosting_round_begin', 'on_boosting_round_end' and 'on_fit_end' are called here.

        Returns a BoostingRound object to be updated with train_acc and valid_acc.
        """
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
    patience = 3
    bi = BoostManager()
    for br in bi.iterate(max_round_number, patience, True):
        # if safe == 0:
        a += .1
        br.train_acc = a
        br.valid_acc = a
        print(br)

        safe += 1
        if safe >= 100:
            break
