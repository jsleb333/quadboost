import numpy as np
from sklearn.metrics import accuracy_score

import sys, os
sys.path.append(os.getcwd())

from weak_learner import Cloner
from weak_learner import MulticlassDecisionStump
from utils import *


class MulticlassDecisionTree(Cloner):
    def __init__(self, max_n_leafs=4, encoder=None):
        super().__init__()
        self.max_n_leafs = max_n_leafs
        self.encoder = encoder
        self.tree = None

    def fit(self, X, Y, W=None, n_jobs=1, sorted_X=None, sorted_X_idx=None):
        if self.encoder is not None:
            Y, W = self.encoder.encode_labels(Y)

        root = MulticlassDecisionStump().fit(X, Y, W, n_jobs, sorted_X, sorted_X_idx)
        self.tree = Tree(root)

        left_args, right_args = self.partition_examples(X, sorted_X_idx, root)

        left_leaf = Leaf(MulticlassDecisionStump().fit(X, Y, W, n_jobs, *left_args), root, 'left')
        right_leaf = Leaf(MulticlassDecisionStump().fit(X, Y, W, n_jobs, *right_args), root, 'right')
        potential_split = [left_leaf, right_leaf]

        n_leafs = 2
        while n_leafs < self.max_n_leafs:
            best_split = self._choose_best_split(potential_split)
            self._append_split(best_split)

    def _choose_best_split(self, potential_split):
        pass

    def _append_split(self, split):
        pass

    def partition_examples(self, X, sorted_X_idx, stump):
        n_examples, n_features = sorted_X_idx.shape
        n_examples_left, n_examples_right = stump.stump_idx, n_examples - stump.stump_idx

        sorted_X_idx_left = np.zeros((n_examples_left, n_features), dtype=int)
        sorted_X_idx_right = np.zeros((n_examples_right, n_features), dtype=int)

        X_partition = np.array([p for p in stump.partition_generator(X)], dtype=bool) # Partition of the examples X (X_partition[i] == 0 if examples i is left, else 1)
        range_n_features = np.arange(n_features)

        idx_left, idx_right = np.zeros(n_features, dtype=int), np.zeros(n_features, dtype=int)
        for xs_idx in sorted_X_idx: # For each row of indices, decide if the index should go left of right
            mask = X_partition[xs_idx]

            sorted_X_idx_left[idx_left[~mask], range_n_features[~mask]] = xs_idx[~mask]
            sorted_X_idx_right[idx_right[mask], range_n_features[mask]] = xs_idx[mask]

            idx_left += ~mask
            idx_right += mask

        sorted_X_left = X[sorted_X_idx_left, range(n_features)]
        sorted_X_right = X[sorted_X_idx_right, range(n_features)]

        return (sorted_X_left, sorted_X_idx_left), (sorted_X_right, sorted_X_idx_right)

    def predict(self, X):
        pass

    def evaluate(self, X, Y):
        pass


class Tree:
    def __init__(self, root_stump):
        self.stump = root_stump
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


class Leaf:
    def __init__(self, stump, parent, side):
        self.stump = stump
        self.parent = parent
        self.side = side


@timed
def main():
    mnist = MNISTDataset.load()
    (Xtr, Ytr), (Xts, Yts) = mnist.get_train_test(center=False, reduce=False)

    encoder = OneHotEncoder(Ytr)

    m = 10
    X = Xtr[:m,20:30].reshape((m,-1))
    Y = Ytr[:m]
    # X, Y = Xtr, Ytr
    wl = MulticlassDecisionTree(encoder=encoder)
    sorted_X, sorted_X_idx = MulticlassDecisionStump.sort_data(X)
    wl.fit(X, Y, n_jobs=4, sorted_X=sorted_X, sorted_X_idx=sorted_X_idx)
    # print('WL train acc:', wl.evaluate(X, Y))
    # print('WL test acc:', wl.evaluate(Xts, Yts))


if __name__ == '__main__':
    from mnist_dataset import MNISTDataset
    from label_encoder import OneHotEncoder
    main()
