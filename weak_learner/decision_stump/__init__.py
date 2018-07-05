try:
    from stump import *
    from decision_stump import *
except ModuleNotFoundError:
    from weak_learner.decision_stump.stump import *
    from weak_learner.decision_stump.decision_stump import *
