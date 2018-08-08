import numpy as np
from utils import ComparableMixin

class Stump(ComparableMixin):
    """
    Stump is a simple class that stores the variables used by the MulticlassDecisionStump algorithm. It provides a method 'update' that changes the values only if the new stump is better than the previous one. It also provides a method 'compute_confidence_rates' for the stored stump.
    """
    def __init__(self, risk, moments):
        super().__init__(cmp_attr='risk')
        self.feature = risk.argmin()
        self.risk = risk[self.feature]
        self.stump_idx = 0
        self.moment_0 = moments[0,:,self.feature,:].copy()
        self.moment_1 = moments[1,:,self.feature,:].copy()

    def update(self, risk, moments, possible_stumps, stump_idx):
        """
        Updates the current stump with the new stumps only if the new risk is lower than the previous one.

        To optimize the algorithm, the risks are computed only for the acceptable stumps, which happen to be represented as the non zero entries of the variable 'possible_stumps'.
        """
        sparse_feature_idx = risk.argmin()
        if risk[sparse_feature_idx] < self.risk:
            self.feature = possible_stumps.nonzero()[0][sparse_feature_idx] # Retrieves the actual index of the feature
            self.risk = risk[sparse_feature_idx]
            self.moment_0 = moments[0,:,self.feature,:].copy()
            self.moment_1 = moments[1,:,self.feature,:].copy()
            self.stump_idx = stump_idx

    def compute_confidence_rates(self):
        return np.divide(self.moment_1, self.moment_0, where=self.moment_0!=0)

    def compute_stump_value(self, sorted_X):
        feat_val = lambda idx: sorted_X[idx, self.feature]
        if self.stump_idx != 0:
            self.stump = (feat_val(self.stump_idx) + feat_val(self.stump_idx-1))/2
        else:
            self.stump = feat_val(self.stump_idx) - 1
        return self.stump
