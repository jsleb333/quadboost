try:
    from decision_stump import MulticlassDecisionStump
except ModuleNotFoundError:
    from weak_learner.decision_stump.decision_stump import MulticlassDecisionStump
