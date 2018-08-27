try:
    from cloner import Cloner
    from ridge import *
    from svr import *
    from decision_stump import MulticlassDecisionStump
    from decision_tree import MulticlassDecisionTree
except ModuleNotFoundError:
    from weak_learner.cloner import Cloner
    from weak_learner.ridge import *
    from weak_learner.svr import *
    from weak_learner.decision_stump import MulticlassDecisionStump
    from weak_learner.decision_tree import MulticlassDecisionTree
