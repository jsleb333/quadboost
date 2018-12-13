import unittest as ut
import numpy as np

from quadboost.weak_learner.decision_stump import MulticlassDecisionStump
from quadboost.label_encoder import OneHotEncoder


class TestMulticlassDecisionStump(ut.TestCase):
    def setUp(self):
        self.X = np.array([[0,13,255],[0,52,127],[3,4,204]])
        self.Y = np.array([0,1,1])

    def test_sort_data_as_staticmethod(self):
        sorted_X, sorted_X_idx = MulticlassDecisionStump.sort_data(self.X)
        sorted_answer_X = np.array([[0,4,127], [0,13,204], [3,52,255]])
        sorted_X_idx_answer = np.array([[0,2,1], [1,0,2], [2,1,0]])
        self.assertTrue(np.all(sorted_X == sorted_answer_X))
        self.assertTrue(np.all(sorted_X_idx == sorted_X_idx_answer))

    def test_fit(self):
        ds = MulticlassDecisionStump(encoder=OneHotEncoder(self.Y))
        sorted_X, sorted_X_idx = ds.sort_data(self.X)
        ds.fit(self.X, self.Y, W=None, sorted_X=sorted_X, sorted_X_idx=sorted_X_idx)
        self.assertEqual(ds.feature, 2)
        self.assertEqual(ds.stump, 229.5)
        self.assertEqual(ds.stump_idx, 2)

    def test_predict(self):
        encoder = OneHotEncoder(self.Y)
        ds = MulticlassDecisionStump(encoder=encoder)
        sorted_X, sorted_X_idx = ds.sort_data(self.X)
        ds.fit(self.X, self.Y, W=None, sorted_X=sorted_X, sorted_X_idx=sorted_X_idx)

        Xts = np.array([[0,0,1], # Should be class 1
                        [0,0,255]]) # Should be class 0
        Y_pred = encoder.decode_labels(ds.predict(Xts))
        self.assertEqual(Y_pred[0], 1)
        self.assertEqual(Y_pred[1], 0)


if __name__ == '__main__':
    ut.main()
