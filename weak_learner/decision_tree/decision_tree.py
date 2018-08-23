import numpy as np
from sklearn.metrics import accuracy_score
import heapq as hq

import sys, os
sys.path.append(os.getcwd())

from weak_learner import Cloner
from weak_learner import MulticlassDecisionStump
from utils import timed, ComparableMixin


class MulticlassDecisionTree(Cloner):
    """
    Decision tree classifier with innate multiclass algorithm. Each node is a MulticlassDecisionStump. The tree is grown by promoting the leaf with the best risk reduction to a node with two leaves.
    It assigns a confidence rates (scalar) for each class for each leaf.
    Parallelization is implemented for the MulticlassDecisionStump.
    """
    def __init__(self, max_n_leaves=4, encoder=None):
        """
        Args:
            max_n_leaves (int, optional): Maximum number of leaves the tree can have. The smallest tree have 2 leaves and is identical to a MulticlassDecisionStump.
            encoder (LabelEncoder object, optional, default=None): Encoder to encode labels. If None, no encoding will be made before fitting.
        """
        super().__init__()
        self.max_n_leaves = max_n_leaves
        self.n_leaves = 2
        self.encoder = encoder
        self.tree = None

    def fit(self, X, Y, W=None, n_jobs=1, sorted_X=None, sorted_X_idx=None):
        """
        Fits the tree to the data.

        The algorithm fits a first stump (the root), which splits the data in 2 leaves (partition). For each leaves of the tree, we compute the decrease in quadratic risk promoting a leaf into a stump would yield, then we append to the tree the one with the greatest reduction. The 2 hence created leaves are added to the pool of potential split and the process is repeated until the maximum number of leaves is reached.

        Args:
            X (Array of shape (n_examples, ...)): Examples
            Y (Array of shape (n_examples,) or (n_examples, n_classes)): Labels for the examples. If an encoder was provided at construction, Y should be a vector to be encoded.
            W (Array of shape (n_examples, n_classes)): Weights of each examples according to their class. Should be None if Y is not encoded.
            n_jobs (int, optional, default=1): Number of processes to execute in parallel to find the stumps.
            sorted_X (Array of shape (n_examples, ...), optional, default=None): Sorted examples along axis 0. If None, 'X' will be sorted, else it will not.
            sorted_X_idx (Array of shape (n_examples, ...), optional, default=None): Indices of the sorted examples along axis 0 (corresponds to argsort). If None, 'X' will be argsorted, else it will not.

        Returns self

        """
        if self.encoder is not None:
            Y, W = self.encoder.encode_labels(Y)

        X = X.reshape(X.shape[0], -1)

        root = MulticlassDecisionStump().fit(X, Y, W, n_jobs, sorted_X, sorted_X_idx)
        self.tree = Tree(root)
        split = Split(root, None, None, sorted_X_idx)
        parent = self.tree

        potential_splits = []
        while self.n_leaves < self.max_n_leaves:
            self.n_leaves += 1

            left_args, right_args = self._partition_examples(X, split.sorted_X_idx, split.stump)

            left_split = Split(MulticlassDecisionStump().fit(X, Y, W, n_jobs, *left_args), parent, 'left', left_args[1])
            right_split = Split(MulticlassDecisionStump().fit(X, Y, W, n_jobs, *right_args), parent, 'right', right_args[1])

            potential_splits.extend([left_split, right_split])

            split_idx, split = max(enumerate(potential_splits), key=lambda pair: pair[1])
            del potential_splits[split_idx]

            parent = self._append_split(split)

        return self

    def _append_split(self, split):
        child = Tree(split.stump)
        if split.side == 'left':
            split.parent.left_child = child
        elif split.side =='right':
            split.parent.right_child = child

        return child

    def _partition_examples(self, X, sorted_X_idx, stump):
        """
        Partition examples into 2 groups (left and right leaves of the node). This is done in O(n_examples * n_features).
        """
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
            node, partition = self.percolate(x)
            Y_pred[i] = node.stump.confidence_rates[partition]
        return Y_pred

    def percolate(self, x):
        """
        Percolate x along the tree, returning the final node and the partition/leaf of x.
        """
        node = self.tree
        x = x.reshape(1,-1)
        partition = node.stump.partition(x, int)
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

            partition = node.stump.partition(x, int)

        return node, partition

    def evaluate(self, X, Y):
        Y_pred = self.predict(X)
        if self.encoder != None:
            Y_pred = self.encoder.decode_labels(Y_pred)
        return accuracy_score(y_true=Y, y_pred=Y_pred)

    def __len__(self):
        return len(self.tree)

    @staticmethod
    def sort_data(X):
        return MulticlassDecisionStump.sort_data(X)


class Tree:
    """
    Simple binary tree data structure to hold the stumps making up the tree. Defines some utilitary functions to handle the data.
    """
    def __init__(self, stump):
        """
        The data is the stump, while the child (left or right) are themselves Tree instances. If they are None, it means they are leaves.
        """
        self.stump = stump
        self.left_child = None
        self.right_child = None

    def __len__(self):
        """
        Returns the number of leaves of the tree.
        """
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
        Prefix visit of nodes.
        """
        yield self
        if self.left_child is not None:
            yield from self.left_child
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


class Split(ComparableMixin, cmp_attr='risk_reduction'):
    """
    Basic class acting as a data holding and ordering structure. Maintains information necessary to append the stump to its correct place in the tree and to compute the risk for its leaves.

    The ComparableMixin parent of the class allows to quickly find the best split among many.
    """
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
    def risk_reduction(self):
        side = 0 if self.side == 'left' else 1
        return self.parent.stump.risks[side] - self.stump.risk


@timed
def main():
    mnist = MNISTDataset.load()
    (Xtr, Ytr), (Xts, Yts) = mnist.get_train_test(center=False, reduce=False)

    encoder = OneHotEncoder(Ytr)

    m = 1_000
    X = Xtr[:m].reshape((m,-1))
    Y = Ytr[:m]
    # X, Y = Xtr, Ytr
    dt = MulticlassDecisionTree(max_n_leaves=4, encoder=encoder)
    sorted_X, sorted_X_idx = dt.sort_data(X)
    dt.fit(X, Y, n_jobs=4, sorted_X=sorted_X, sorted_X_idx=sorted_X_idx)
    print('WL train acc:', dt.evaluate(X, Y))
    print('WL test acc:', dt.evaluate(Xts, Yts))


if __name__ == '__main__':
    from mnist_dataset import MNISTDataset
    from label_encoder import OneHotEncoder
    main()
