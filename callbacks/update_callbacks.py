import sys, os
sys.path.append(os.getcwd())

from callbacks import Callback


class UpdateCallback(Callback):
    pass


class UpdateTrainAcc(UpdateCallback):
    def on_boosting_round_end(self):
        if self.manager.boosting_round.train_acc_was_set_this_round:
            if self.manager.boosting_round.train_acc > self.manager.best_train_acc:
                self.manager.best_train_acc = self.manager.boosting_round.train_acc