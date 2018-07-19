import numpy as np
import sys, os
sys.path.append(os.getcwd())

from callbacks import Callback


class BreakCallback(Callback):
    pass


class BreakOnMaxStep(BreakCallback):
    def __init__(self, max_step_number=None, manager=None):
        super().__init__(manager)
        self.max_step_number = max_step_number

    def on_step_begin(self):
        if self.max_step_number is not None:
            if self.manager.step_number >= self.max_step_number-1:
                raise StopIteration


class BreakOnPlateau(BreakCallback):
    def __init__(self, patience=None, manager=None):
        super().__init__(manager)
        self.patience = patience
        self.rounds_since_no_improvements = 0
        self.best_train_acc = -1

    def on_step_end(self):
        if self.manager.step.train_acc is not None:
            self._update_best_train_acc()
            if self.patience is not None and self.rounds_since_no_improvements > self.patience:
                raise StopIteration

    def _update_best_train_acc(self):
        if self.manager.step.train_acc > self.best_train_acc:
            self.best_train_acc = self.manager.step.train_acc
            self.rounds_since_no_improvements = 0
        else:
            self.rounds_since_no_improvements += 1


class BreakOnPerfectTrainAccuracy(BreakCallback):
    def __init__(self, manager=None):
        super().__init__(manager)

    def on_step_end(self):
        if self.manager.step.train_acc is not None:
            if np.isclose(self.manager.step.train_acc, 1.0):
                raise StopIteration
