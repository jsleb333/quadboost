try:
    from weak_learner_base import _WeakLearnerBase, _Cloner
    from ridge import *
    from svr import *
    from decision_stump import MulticlassDecisionStump
    from decision_tree import MulticlassDecisionTree
    from random_convolution import RandomConvolution, SparseRidgeRC, Filters, LocalFilters, WeightFromBankGenerator, center_weight, normalize_weight, reduce_weight
except ModuleNotFoundError:
    from .weak_learner_base import _WeakLearnerBase, _Cloner
    from .ridge import *
    from .svr import *
    from .decision_stump import MulticlassDecisionStump
    from .decision_tree import MulticlassDecisionTree
    from .random_convolution import RandomConvolution, SparseRidgeRC, Filters, LocalFilters, WeightFromBankGenerator, center_weight, normalize_weight, reduce_weight
