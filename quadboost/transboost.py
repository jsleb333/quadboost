import numpy as np
import pickle as pkl
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import logging

import sys, os
sys.path.append(os.getcwd())

try:
    from weak_learner import *
    from label_encoder import LabelEncoder, OneHotEncoder, AllPairsEncoder
    from callbacks import CallbacksManagerIterator, Step
    from callbacks import ModelCheckpoint, CSVLogger, Progression, BestRoundTrackerCallback
    from callbacks import (BreakOnMaxStepCallback, BreakOnPerfectTrainAccuracyCallback,
                        BreakOnPlateauCallback, BreakOnZeroRiskCallback)
    from utils import *

except ModuleNotFoundError:
    from .weak_learner import *
    from .label_encoder import LabelEncoder, OneHotEncoder, AllPairsEncoder
    from .callbacks import CallbacksManagerIterator, Step
    from .callbacks import ModelCheckpoint, CSVLogger, Progression, BestRoundTrackerCallback
    from .callbacks import (BreakOnMaxStepCallback, BreakOnPerfectTrainAccuracyCallback,
                        BreakOnPlateauCallback, BreakOnZeroRiskCallback)
    from .utils import *

from quadboost import QuadBoostMH, QuadBoostMHCR


class TransBoost(QuadBoostMHCR):
    """
    QuadBoostMHCR, but with a twist: every τ steps, the previous weak learners must provide a (non linear) transformation to apply to X before resuming the training.
    """



