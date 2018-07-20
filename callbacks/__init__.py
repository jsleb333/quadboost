try:
    from callback import *
    from break_callbacks import *
    from callback_list import *
    from progression import *
    from save_callbacks import *
    from iterator_manager import *
    from boost_manager import *
    from model_checkpoint import *
    from logger import *
except ModuleNotFoundError:
    from callbacks.callback import *
    from callbacks.break_callbacks import *
    from callbacks.callback_list import *
    from callbacks.progression import *
    from callbacks.save_callbacks import *
    from callbacks.iterator_manager import *
    from callbacks.boost_manager import *
    from callbacks.model_checkpoint import *
    from callbacks.logger import *
