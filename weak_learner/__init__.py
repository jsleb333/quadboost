try:
    from weak_learner.weak_learner_base import _WeakLearnerBase, _Cloner
    from weak_learner.ridge import *
    from weak_learner.svr import *
    from weak_learner.decision_stump import MulticlassDecisionStump
    from weak_learner.decision_tree import MulticlassDecisionTree
    from weak_learner.random_convolution import RandomConvolution, Filters, LocalFilters, WeightFromBankGenerator, center_weight, normalize_weight, reduce_weight
except ModuleNotFoundError:
    from weak_learner_base import _WeakLearnerBase, _Cloner
    from ridge import *
    from svr import *
    from decision_stump import MulticlassDecisionStump
    from decision_tree import MulticlassDecisionTree
    from random_convolution import RandomConvolution, Filters, LocalFilters, WeightFromBankGenerator, center_weight, normalize_weight, reduce_weight
