from callbacks import Callback
import logging


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
        self.best_round = 0
        self.best_value = float('-inf') if self.monitor == 'max' else float('inf')

    def on_step_end(self):
        value = getattr(self.manager.step, self.quantity)
        if self._value_is_better_than_current(value):
            self.best_value = value
            self.manager.caller.best_round = self.best_round = self.manager.step.step_number + 1
            if self.verbose:
                logging.info('New best model found.')

    def _value_is_better_than_current(self, value):
        if self.monitor == 'max':
            return value >= self.best_value
        elif self.monitor == 'min':
            return value <= self.best_value
