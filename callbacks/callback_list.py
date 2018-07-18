import sys, os
sys.path.append(os.getcwd())

from callbacks import Callback, BreakCallback


class CallbackList:
    def __init__(self, manager=None, callbacks=list()):
        self.break_callbacks = []
        self.callbacks = []

        for callback in callbacks: self.append(callback)

        self._manager = manager
        if manager is not None: self.manager = manager
    
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

    def on_fit_begin(self):
        for callback in self.callbacks: callback.on_fit_begin()

    def on_fit_end(self):
        for callback in self.callbacks: callback.on_fit_end()

    def on_boosting_round_begin(self):
        for callback in self.callbacks: callback.on_boosting_round_begin()
        for callback in self.break_callbacks: callback.on_boosting_round_begin()

    def on_boosting_round_end(self):
        for callback in self.callbacks: callback.on_boosting_round_end()
        for callback in self.break_callbacks: callback.on_boosting_round_end()
