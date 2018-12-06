import numpy as np
import logging
import sys, os
sys.path.append(os.getcwd())

try:
    from quadboost.callbacks import Callback, BestRoundTrackerCallback
except ModuleNotFoundError:
    from callbacks import Callback, BestRoundTrackerCallback


class BreakCallback(Callback):
    """
    This abstract class implements a callback with purpose to stop the iteration according to a condition. Hence, all subclasses of BreakCallback should eventually raise a StopIteration exception at on_step_begin or on_step_end.

    N.B.: All callbacks that raise such an exception should inherit from BreakCallback, or else other callbacks are not guarenteed to be called, since a StopIteration will prevent it. The CallbackList object correctly manages these BreakCallbacks by calling them only after ordinary Callbacks.
    """
    pass


class BreakOnMaxStepCallback(BreakCallback):
    def __init__(self, max_step_number=None, manager=None):
        super().__init__(manager)
        self.max_step_number = max_step_number

    def on_step_begin(self):
        if self.max_step_number is not None:
            if self.manager.step.step_number + 1 >= self.max_step_number:
                logging.info('Terminating iteration due to maximum round number reached.')
                raise StopIteration


class BreakOnPlateauCallback(BreakCallback, BestRoundTrackerCallback):
    def __init__(self, patience=None, manager=None, quantity='valid_acc', monitor='max'):
        super().__init__(quantity=quantity, monitor=monitor, manager=manager)
        self.patience = patience

    def on_step_end(self):
        if self.patience is not None:
            super().on_step_end() # Updates self.best_round
            if self.manager.step.step_number - self.best_round.step_number - 1 > self.patience:
                logging.info('Terminating iteration due to maximum round number without improvement reached.')
                raise StopIteration


class BreakOnPerfectTrainAccuracyCallback(BreakCallback):
    def __init__(self, manager=None):
        super().__init__(manager)

    def on_step_end(self):
        if self.manager.step.train_acc is not None:
            if np.isclose(self.manager.step.train_acc, 1.0):
                logging.info('Terminating iteration due to maximum accuracy reached.')
                raise StopIteration


class BreakOnZeroRiskCallback(BreakCallback):
    def __init__(self, manager=None):
        super().__init__(manager)

    def on_step_end(self):
        if hasattr(self.manager.step, 'risk') and self.manager.step.risk is not None:
            if np.isclose(self.manager.step.risk, 0.0):
                logging.info('Terminating iteration due to risk being zero.')
                raise StopIteration
