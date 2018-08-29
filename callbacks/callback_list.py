import sys, os
sys.path.append(os.getcwd())

from callbacks import Callback, BreakCallback


class CallbackList:
    """
    Class that aggregates all Callback objects. The class aims to mimick the behaviour of a single callback.

    CallbackList will call all appropriate callbacks at the right time. BreakCallbacks are treated separately from ordinary callbacks. This is to ensure all callbacks are called before a StopIteration exception is called (this would prevent ordinary callbacks to be called).

    To facilitate the use of callbacks within a manager, when a manager is set to self.manager, all present callbacks' manager are also set to this manager. Moreover, all subsequently appended callbacks will be assigned this manager.
    """
    def __init__(self, manager=None, callbacks=list()):
        """
        Args:
            manager (Reference to the IteratorManager, optional): The manager can be set in the constructor or can be set later. All callbacks' manager will be set to this manager. The manager should not let to None, since most callbacks use this reference.
            callbacks (Iterable of Callback objects, optional): All callbacks methods will be called on time with this CallbackList. Callbacks can always be appended later and their manager will be set to the same as the CallbackList manager.
        """
        self.break_callbacks = []
        self.callbacks = []
        self.manager = manager

        for callback in callbacks: self.append(callback)

    @property
    def manager(self):
        return self._manager
    @manager.setter
    def manager(self, manager):
        self._manager = manager
        for callback in self: callbacks.manager = manager

    def append(self, callback):
        callback.manager = self.manager
        if issubclass(type(callback), BreakCallback):
            self.break_callbacks.append(callback)
        else:
            self.callbacks.append(callback)

    def __iter__(self):
        yield from self.callbacks
        yield from self.break_callbacks

    def on_iteration_begin(self):
        for callback in self: callback.on_iteration_begin()

    def on_iteration_end(self):
        for callback in self: callback.on_iteration_end()

    def on_step_begin(self):
        for callback in self: callback.on_step_begin()

    def on_step_end(self):
        for callback in self: callback.on_step_end()

    def on_exception_exit(self, exception_type=None, exception_message=None, traceback=None):
        for callback in self: callback.on_exception_exit(exception_type,
                                                         exception_message,
                                                         traceback)
