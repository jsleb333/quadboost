import numpy as np
from sklearn.metrics import accuracy_score

import sys, os
sys.path.append(os.getcwd())

from weak_learner import Cloner
from weak_learner import decision_stump
from utils import *


class MulticlassDecisionTree(Cloner):
    def __init__(self, max_n_leafs, encoder):
        self.max_n_leafs = max_n_leafs
        self.encoder = encoder
        self.tree = None

    def fit(self, X, Y, W=None, n_jobs=1, sorted_X=None, sorted_X_idx=None):
        pass

    def predict(self, X):
        pass

    def evaluate(self, X, Y):
        pass


class Tree:
    def __init__(self, stump, parent=None):
        self.stump = stump
        self.parent = parent
        self.left_child = None
        self.right_child = None

    def __len__(self):
        if self.left_child is None and self.right_child is None:
            return 2
        elif self.left_child is not None and self.right_child is None:
            return len(self.left_child) + 1
        elif self.right_child is not None and self.left_child is None:
            return len(self.right_child) + 1
        else:
            return len(self.right_child) + len(self.left_child)


if __name__ == '__main__':
    pass
