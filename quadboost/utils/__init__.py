try:
    from misc import *
    from timed import timed
    from comparable_mixin import ComparableMixin
except ModuleNotFoundError:
    from .misc import *
    from .timed import timed
    from .comparable_mixin import ComparableMixin
