try:
    from misc import *
    from timed import timed
    from comparable_mixin import ComparableMixin
except ModuleNotFoundError:
    from utils.misc import *
    from utils.timed import timed
    from utils.comparable_mixin import ComparableMixin
