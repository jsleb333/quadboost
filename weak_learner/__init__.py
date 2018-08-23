try:
    from cloner import *
    from ridge import *
    from svr import *
    from decision_stump import *
    from decision_tree import *
except ModuleNotFoundError:
    from weak_learner.cloner import *
    from weak_learner.ridge import *
    from weak_learner.svr import *
    from weak_learner.decision_stump import *
    from weak_learner.decision_tree import *
