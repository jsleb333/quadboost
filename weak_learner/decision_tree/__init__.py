try:
    from decision_tree import MulticlassDecisionTree
except ModuleNotFoundError:
    from weak_learner.decision_tree.decision_tree import MulticlassDecisionTree
