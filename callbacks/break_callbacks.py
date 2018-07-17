import numpy as np
import sys, os
sys.path.append(os.getcwd())

from callbacks import Callback


class BreakCallback(Callback):
    pass


class BreakOnMaxRound(BreakCallback):
    def __init__(self, manager, max_round_number=None):
        super().__init__(manager)
        self.max_round_number = max_round_number

    def on_boosting_round_begin(self):
        if self.max_round_number is not None and self.manager.round_number >= self.max_round_number:
            raise StopIteration


class BreakOnPlateau(BreakCallback):
    def __init__(self, manager, patience=None):
        super().__init__(manager)
        self.patience = patience
        self.rounds_since_no_improvements = 0
        self.best_train_acc = 0

    def on_boosting_round_end(self):
        if self.manager.boosting_round.train_acc_was_set_this_round:
            self._update_best_train_acc()
            if self.patience is not None and self.rounds_since_no_improvements > self.patience:
                raise StopIteration
        else:
            self.manager.boosting_round.warn_train_acc_was_not_updated()

    def _update_best_train_acc(self):
        if self.manager.boosting_round.train_acc > self.best_train_acc:
            self.best_train_acc = self.manager.boosting_round.train_acc
            self.rounds_since_no_improvements = 0
        else:
            self.rounds_since_no_improvements += 1


class BreakOnPerfectTrainAccuracy(BreakCallback):
    def __init__(self, manager):
        super().__init__(manager)

    def on_boosting_round_end(self):
        if self.manager.boosting_round.train_acc_was_set_this_round:
            if np.isclose(self.manager.boosting_round.train_acc, 1.0):
                raise StopIteration
