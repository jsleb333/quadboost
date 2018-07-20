import sys, os
sys.path.append(os.getcwd())

import warnings
import pickle as pkl
import csv

from callbacks import Callback


class SaveCallback(Callback):
    def __init__(self, filename, dirname='.',
                 manager=None,
                 atomic_write=True, open_mode='wb'):
        super().__init__(manager)
        self.filename = filename
        self.dirname = dirname
        os.makedirs(dirname, exist_ok=True)

        self.atomic_write = atomic_write
        self.open_mode = open_mode

    @property
    def filedir(self):
        return self.dirname + '/' + self.format_filename(self.filename)

    @property
    def tmp_filedir(self):
        return self.dirname + '/tmp_' + self.format_filename(self.filename)

    def format_filename(self, filename):
        return filename

    def save(self, *args, **kwargs):
        atomic_save_successful = False
        if self.atomic_write:
            atomic_save_successful = self._atomic_save(*args, **kwargs)

        if not atomic_save_successful:
            self._save(self.filedir, *args, **kwargs)

    def _atomic_save(self, *args, **kwargs):
        self._save(self.tmp_filedir, *args, **kwargs)
        try:
            os.replace(self.tmp_filedir, self.filedir)
            atomic_save_successful = True
        except OSError:
            warnings.warn(f"Could not replace '{self.filedir}' with '{self.tmp_filedir}'. Saving non-atomically instead.")
            os.remove(self.tmp_filedir)
            atomic_save_successful = False

        return atomic_save_successful

    def _save(self, filedir, *args, **kwargs):
        raise NotImplementedError


class PeriodicSaveCallback(SaveCallback):
    def __init__(self, *args, period=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_calls = 0
        self.period = period

    def save(self, *args, override_period=False, **kwargs):
        self.n_calls += 1
        saved = False
        if self.period is not None:
            if self.n_calls % self.period == 0 or override_period:
                super().save(*args, **kwargs)
                saved = True
        return saved


class PickleSave(SaveCallback):
    def __init__(self, *args, protocol=pkl.HIGHEST_PROTOCOL, **kwargs):
        super().__init__(*args, **kwargs)
        self.protocol = protocol

    def _save(self, filedir, obj):
        with open(filedir, self.open_mode) as file:
            pkl.dump(obj, file, protocol=self.protocol)


class CSVSave(SaveCallback):
    def __init__(self, *args, open_mode='w', delimiter=',', newline='', **kwargs):
        super().__init__(*args, open_mode=open_mode, **kwargs)
        self.delimiter = delimiter
    
    def _save(self, filedir, doc):
        with open(filedir, self.open_mode, newline=self.newline)
            writer = csv.writer(file, delimiter=self.delimiter)
            writer.writerows(doc)
