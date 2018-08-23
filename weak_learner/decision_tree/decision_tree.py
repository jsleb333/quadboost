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
        split = Split(root, None, None, sorted_X_idx)
        parent = self.tree

        potential_splits = []
        while self.n_leafs < self.max_n_leafs:
            self.n_leafs += 1

            left_args, right_args = self.partition_examples(X, split.sorted_X_idx, split.stump)

            left_split = Split(MulticlassDecisionStump().fit(X, Y, W, n_jobs, *left_args), parent, 'left', left_args[1])
            right_split = Split(MulticlassDecisionStump().fit(X, Y, W, n_jobs, *right_args), parent, 'right', right_args[1])

            potential_splits.extend([left_split, right_split])

            split_idx, split = max(enumerate(potential_splits), key=lambda pair: pair[1])
            del potential_splits[split_idx]

            parent = self._append_split(split)

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

        X_partition = stump.partition(X) # Partition of the examples X (X_partition[i] == 0 if examples i is left, else 1)
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
        n_examples = X.shape[0]
        n_partitions, n_classes = self.tree.stump.confidence_rates.shape
        Y_pred = np.zeros((n_examples, n_classes))

        node = self.tree
        partition = node.stump.partition(X)

        for i, x in enumerate(X):
            Y_pred[i] = self.percolate(x)
        return Y_pred

    def percolate(self, x):
        node = self.tree
        partition = node.stump.partition(x.reshape(1,-1), int)
        while True:
            if partition == 0:
                if node.left_child is None:
                    break
                else:
                    node = node.left_child
            if partition == 1:
                if node.right_child is None:
                    break
                else:
                    node = node.right_child

                partition = node.stump.partition(x.reshape(1,-1), int)

        return node.stump.confidence_rates[partition]

    def evaluate(self, X, Y):
        Y_pred = self.predict(X)
        if self.encoder != None:
            Y_pred = self.encoder.decode_labels(Y_pred)
        return accuracy_score(y_true=Y, y_pred=Y_pred)

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
        """
        Infix visit of nodes.
        """
        if self.left_child is not None:
            yield from self.left_child
        yield self
        if self.right_child is not None:
            yield from self.right_child

    def __str__(self):
        nodes_to_visit = [(0, self)]
        visited_nodes = ['0 (root)']
        i = 0
        while nodes_to_visit:
            node_no, node = nodes_to_visit.pop()

            if node.left_child:
                i += 1
                nodes_to_visit.append((i, node.left_child))
                visited_nodes.append(f'{i} (left of {node_no})')

            if node.right_child:
                i += 1
                nodes_to_visit.append((i, node.right_child))
                visited_nodes.append(f'{i} (right of {node_no})')

        return ' '.join(visited_nodes)


class Split(ComparableMixin, cmp_attr='risk_decrease'):
    def __init__(self, stump, parent, side, sorted_X_idx):
        """
        Args:
            stump (MulticlassDecisionStump object): Stump of the split.
            parent (MultclassDesicionStump object): Parent stump of the split. This information is needed to know where to append the split in the final tree.
            side (string, 'left' or 'right'): Side of the partition. Left corresponds to partition 0 and right to 1. This information is needed to know where to append the split in the final tree.
            sorted_X_idx (Array of shape (n_examples_side, n_features)): Array of indices of sorted X for the side of the split.
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

    m = 4
    X = Xtr[:m].reshape((m,-1))
    Y = Ytr[:m]
    # X, Y = Xtr, Ytr
    dt = MulticlassDecisionTree(max_n_leafs=4, encoder=encoder)
    sorted_X, sorted_X_idx = MulticlassDecisionStump.sort_data(X)
    dt.fit(X, Y, n_jobs=4, sorted_X=sorted_X, sorted_X_idx=sorted_X_idx)
    print('WL train acc:', dt.evaluate(X, Y))

    for n in dt.tree:
        print(n.stump.stump, n.stump.feature)
        print(n.stump.confidence_rates)
    print(dt.predict(X))
    # print('WL test acc:', dt.evaluate(Xts, Yts))

    ds = MulticlassDecisionStump(encoder=encoder)
    ds.fit(X, Y, n_jobs=4, sorted_X=sorted_X, sorted_X_idx=sorted_X_idx)
    print('WL train acc:', ds.evaluate(X, Y))


if __name__ == '__main__':
    from mnist_dataset import MNISTDataset
    from label_encoder import OneHotEncoder
    main()
