import logging
from copy import deepcopy

try:
    from quadboost.callbacks import Callback
except ModuleNotFoundError:
    from callbacks import Callback


class BestRoundTrackerCallback(Callback):
    """
    This callback keeps track of the best round according to the quantity to monitor and updates the model with it.
    """
    def __init__(self, *args, quantity='valid_acc', monitor='max', verbose=False, **kwargs):
        """
        Args:
            quantity (str, either 'valid_acc', 'train_acc', 'risk', optional): Monitored quantity to choose the best model from.
            monitor (str, either 'max' or 'min', optional): Monitor the quantity for the max or the min value to restore the best model.
            verbose (bool, optional): Whether or not to log new best found.
        """
        super().__init__(*args, **kwargs)
        self.quantity = quantity
        self.monitor = monitor
        self.verbose = verbose
        self.best_round = None

    @property
    def best_value(self):
        if self.best_round is None:
            return float('-inf') if self.monitor == 'max' else float('inf')
        else:
            return getattr(self.best_round, self.quantity)

    def on_step_end(self):
        if self._new_round_is_better_than_current():
            self.manager.caller.best_round = self.best_round = deepcopy(self.manager.step)
            if self.verbose:
                logging.info('New best model found.')

    def _new_round_is_better_than_current(self):
        new_value = getattr(self.manager.step, self.quantity)
        if self.monitor == 'max':
            return new_value >= self.best_value
        elif self.monitor == 'min':
            return new_value <= self.best_value
