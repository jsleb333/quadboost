import sys, os
sys.path.append(os.getcwd())

from callbacks import Callback, BreakCallback


class CallbackList:
    def __init__(self, callbacks=list()):
        self.break_callbacks = []
        self.callbacks = []
        
        for callback in callbacks:
            if issubclass(type(callback), BreakCallback):
                self.break_callbacks.append(callback)
            else:
                self.callbacks.append(callback)
    
    def append(self, callback):
        self.callbacks.append(callback)
    
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
