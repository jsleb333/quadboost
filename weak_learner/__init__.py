try:
    from weak_learner.weak_learner_base import _WeakLearnerBase
    from weak_learner.ridge import *
    from weak_learner.svr import *
    from weak_learner.decision_stump import MulticlassDecisionStump
    from weak_learner.decision_tree import MulticlassDecisionTree
    from weak_learner.random_convolution import RandomCompleteConvolution, RandomLocalConvolution
except ModuleNotFoundError:
    from weak_learner_base import _WeakLearnerBase
    from ridge import *
    from svr import *
    from decision_stump import MulticlassDecisionStump
    from decision_tree import MulticlassDecisionTree
    from random_convolution import RandomCompleteConvolution, RandomLocalConvolution
