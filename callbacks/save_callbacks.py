import sys, os
sys.path.append(os.getcwd())

from callbacks import Callback


class PeriodicSaveCallback(Callback):
    def __init__(self, manager,
                 filename,
                 dirname='./',
                 period=1,
                 atomic_write=True,
                 open_mode='wb'):
        super().__init__(manager)
        self.filename = filename
        self.dirname = dirname
        self.period = period
        self.atomic_write = atomic_write
        self.open_mode = open_mode

    def _save_file(self, *args, **kwargs):
        raise NotImplementedError

    def save_file(self, *args, **kwargs):
        if self.atomic_write:
            with open(self.dirname + 'tmp_' + self.filename, self.open_mode):
                self._save_file(*args, **kwargs)
