try:
    from callback import *
    from break_callbacks import *
    from callback_list import *
    from update_callbacks import *
    from save_callbacks import *
except ModuleNotFoundError:
    from callbacks.callback import *
    from callbacks.break_callbacks import *
    from callbacks.callback_list import *
    from callbacks.update_callbacks import *
    from callbacks.save_callbacks import *
