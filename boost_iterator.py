import numpy as np
from warnings import warn
from time import time


class BoostingRound:
    """
    Class that stores information about the current boosting round, such as the number of the iteration 't' and the training and validation accuracies. The class contains a flag to indicate if the training accuracy have been updated for the current boosting round, and provides a formatting of the informations in the __str__ method.
    """
    def __init__(self):
        self._t = 0
        self._train_acc = 0
        self.train_acc_was_set_this_round = True
        self.valid_acc = None
        self.start_time = 0
        self.end_time = 0

    def __str__(self):

        boosting_round = 'Boosting round {t:03d}'.format(t=self.t)

        if self.train_acc_was_set_this_round:
            t_acc = 'train accuracy: {train_acc:.3f}'.format(train_acc=self.train_acc)
        else:
            self.warn_train_acc_was_not_updated()
            t_acc = 'train accuracy: ?.???'

        output = [boosting_round, t_acc]
        if self.valid_acc is not None:
            v_acc = 'val accuracy: {:.3f}'.format(self.valid_acc)
            output.append(v_acc)

        output.append(f'Round time: {self.end_time-self.start_time:.2f}s')

        return ' | '.join(output)

    def warn_train_acc_was_not_updated(self):
        if not self.train_acc_was_set_this_round:
            warn("The train_acc attribute should be set for the iterator to work properly. Otherwise, the 'patience' mechanism will not work, and the iteration will not stop if the training accuracy reaches 1.0.")

    @property
    def train_acc(self):
        return self._train_acc

    @train_acc.setter
    def train_acc(self, train_acc):
        self.train_acc_was_set_this_round = True
        self._train_acc = train_acc
        self.end_time = time()

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, t):
        self.train_acc_was_set_this_round = False # On a new round, train_acc is not yet updated.
        self._t = t
        self.start_time = time()


class BoostIterator:
    """
    Class that implements an iterator to boost in QuadBoost. The iterator yields a BoostingRound object that should be updated at each boosting round with the training accuracy. The iterator stops the iteration when:
        - the maximum number of boosting rounds 'T' has been reached (if T is not -1)
        - the training accuracy did not improve for 'patience' rounds (if patience is not None)
        - the training accuracy has reached 1.0
    """
    def __init__(self, T, patience):
        """
        T (int, optional, default=-1): Number of boosting rounds. If T=-1, the algorithm will boost indefinitely, until reaching a training accuracy of 1.0, or until the training accuracy does not improve for 'patience' consecutive boosting rounds.
        patience (int, optional, default=10): Number of boosting rounds before terminating the algorithm when the training accuracy shows no improvements. If patience=None, the boosting rounds will continue until T iterations.
        """
        self.T = T
        self.t = 0
        self.best_train_acc = -1
        self.rounds_since_no_improvements = 0
        self.patience = patience
        self.boosting_round = BoostingRound()

        if T == -1 and patience is None:
            warn("Beware that the values of 'T=-1' and 'patience=None' may result in an infinite loop if the algorithm does not converge to a training accuracy of 1.0.")

    def __iter__(self):
        return self

    def __next__(self):
        if self.t == 0: # On first round, no check needed.
            self.t = self.boosting_round.t = 1
            return self.boosting_round

        if self.T != -1 and self.t >= self.T:
            raise StopIteration

        if self.boosting_round.train_acc_was_set_this_round:
            self.update_best_train_acc()

            if np.isclose(self.best_train_acc, 1.0):
                raise StopIteration

            if self.patience is not None and self.rounds_since_no_improvements > self.patience:
                raise StopIteration
        else:
            self.boosting_round.warn_train_acc_was_not_updated()

        self.t += 1
        self.boosting_round.t = self.t
        return self.boosting_round

    def update_best_train_acc(self):
        if self.boosting_round.train_acc > self.best_train_acc:
            self.best_train_acc = self.boosting_round.train_acc
            self.rounds_since_no_improvements = 0
        else:
            self.rounds_since_no_improvements += 1


if __name__ == '__main__':
    a = 0
    safe = 0
    T = 10
    patience = None
    bi = BoostIterator(T, patience)
    for br in bi:
        # if safe == 0:
        a += .1
        # br.train_acc = a
        br.valid_acc = a
        print(br)

        safe += 1
        if safe >= 100:
            break
