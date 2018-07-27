import sys, os
sys.path.append(os.getcwd())

import numpy as np
from warnings import warn
from time import time

from callbacks import IteratorManager, Step
from callbacks import BreakOnMaxStep, BreakOnPlateau, BreakOnPerfectTrainAccuracy
from callbacks import Progression


class BoostingRound(Step):
    """
    Class that stores information about the current boosting round, storing the round number and the training and validation accuracies.
    """
    def __init__(self, round_number=0):
        self.round = round_number
        self.train_acc = None
        self.valid_acc = None

    def __next__(self):
        self.round += 1
        return self


class BoostManager(IteratorManager):
    """
    Class that manages the boosting in the QuadBoost algorithm by handling callbacks.

    Boosting rounds info (training accuracy, validation accuracy) are handled in a BoostingRound object which is returned by the iterator.

    The class provides stock callbacks which are Progression (always present), BreakOnMaxStep, BreakOnPlateau and BreakOnPerfectTrainAccuracy (optional in 'iterate')
    """
    def __init__(self, model=None, callbacks=None):
        """
        model (QuadBoost object, optional): Reference to the QuadBoost object to manage. If the callbacks do not use the model, it can be omitted.
        callbacks (Iterable of Callbacks objects, optional): Callbacks handles functions to call at specific time in the program. Usage examples: stop the iteration or save the model or the logs.
        """
        if callbacks is None:
            callbacks = [Progression()]
        elif not any(isinstance(callback, Progression) for callback in callbacks):
            callbacks.append(Progression())

        super().__init__(caller=model, callbacks=callbacks, step=BoostingRound())

    def iterate(self, max_round_number=None,
                      patience=None,
                      break_on_perfect_train_acc=True,
                      starting_round_number=0
                      ):
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
            starting_round_number (int, optional): Indicates to which round number the iteration should start.
        """
        if max_round_number:
            self.callbacks.append(BreakOnMaxStep(max_step_number=max_round_number))
        if patience:
            self.callbacks.append(BreakOnPlateau(patience=patience))
        if break_on_perfect_train_acc:
            self.callbacks.append(BreakOnPerfectTrainAccuracy())

        super().iterate(starting_step_number=starting_round_number)
        return self


if __name__ == '__main__':
    from time import sleep
    a = 0
    safe = 0
    max_round_number = 20
    patience = 3
    bi = BoostManager()
    for br in bi.iterate(max_round_number, patience, True):
        # a += .1
        a = 1
        br.train_acc = a
        br.valid_acc = a
        # sleep(.1)
        safe += 1
        if safe >= 100:
            break
