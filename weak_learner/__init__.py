try:
    from weak_learner.weak_learner_base import WeakLearnerBase, Cloner
    from weak_learner.ridge import *
    from weak_learner.svr import *
    from weak_learner.random_filters import RandomFilters
    from weak_learner.decision_stump import MulticlassDecisionStump
    from weak_learner.decision_tree import MulticlassDecisionTree
except ModuleNotFoundError:
    from weak_learner_base import WeakLearnerBase, Cloner
    from ridge import *
    from svr import *
    from random_filters import RandomFilters
    from decision_stump import MulticlassDecisionStump
    from decision_tree import MulticlassDecisionTree
