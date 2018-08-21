import numpy as np
from sklearn.metrics import accuracy_score
import heapq as hq

import sys, os
sys.path.append(os.getcwd())

from weak_learner import Cloner
from weak_learner import MulticlassDecisionStump
from utils import *


class MulticlassDecisionTree(Cloner):
    def __init__(self, max_n_leafs=4, encoder=None):
        super().__init__()
        self.max_n_leafs = max_n_leafs
        self.n_leafs = 2
        self.encoder = encoder
        self.tree = None

    def fit(self, X, Y, W=None, n_jobs=1, sorted_X=None, sorted_X_idx=None):
        if self.encoder is not None:
            Y, W = self.encoder.encode_labels(Y)

        root = MulticlassDecisionStump().fit(X, Y, W, n_jobs, sorted_X, sorted_X_idx)
        self.tree = Tree(root)
        split = Leaf(root, self.tree, None, sorted_X_idx)
        parent = self.tree

        potential_split = []
        while self.n_leafs < self.max_n_leafs:
            self.n_leafs += 1

            left_args, right_args = self.partition_examples(X, split.sorted_X_idx, split.stump)

            left_leaf = Leaf(MulticlassDecisionStump().fit(X, Y, W, n_jobs, *left_args), parent, 'left', left_args[1])
            right_leaf = Leaf(MulticlassDecisionStump().fit(X, Y, W, n_jobs, *right_args), parent, 'right', right_args[1])

            potential_split.extend([left_leaf, right_leaf])

            split_idx, split = max(enumerate(potential_split), key=lambda pair: pair[1])
            del potential_split[split_idx]
            parent = self._append_split(split)

            for i in enumerate(self.tree):
                print(i)

    def _append_split(self, split):
        child = Tree(split.stump)
        if split.side == 'left':
            split.parent.left_child = child
        elif split.side =='right':
            split.parent.right_child = child

        return child

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

    def __len__(self):
        return len(self.tree)


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

    def __iter__(self):
        if self.left_child is not None:
            yield from self.left_child
        yield self
        if self.right_child is not None:
            yield from self.right_child

    def __str__(self):
        return str([i for i, n in enumerate(self)])


class Leaf(ComparableMixin, cmp_attr='risk_decrease'):
    def __init__(self, stump, parent, side, sorted_X_idx):
        """
        Args:
            stump (MulticlassDecisionStump object): Stump of the leaf.
            parent (MultclassDesicionStump object): Parent stump of the leaf. This information is needed to know where to append the leaf in the final tree.
            side (string, 'left' or 'right'): Side of the partition. Left corresponds to partition 0 and right to 1. This information is needed to know where to append the leaf in the final tree.
            sorted_X_idx (Array of shape (n_examples_side, n_features)): Array of indices of sorted X for the side of the leaf.
        """
        self.stump = stump
        self.parent = parent
        self.side = side
        self.sorted_X_idx = sorted_X_idx

    @property
    def risk_decrease(self):
        side = 0 if self.side == 'left' else 1
        return self.parent.stump.risks[side] - self.stump.risk


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
    print(len(wl))
    # print('WL train acc:', wl.evaluate(X, Y))
    # print('WL test acc:', wl.evaluate(Xts, Yts))


if __name__ == '__main__':
    from mnist_dataset import MNISTDataset
    from label_encoder import OneHotEncoder
    main()
