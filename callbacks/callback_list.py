import sys, os
sys.path.append(os.getcwd())

from callbacks import Callback, BreakCallback


class CallbackList:
    def __init__(self, manager=None, callbacks=list()):
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
