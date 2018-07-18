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

    def _save_file(self, file, *args, **kwargs):
        raise NotImplementedError
    
    def _save(self, filedir, *args, **kwargs):
        with open(filedir, self.open_mode) as file:
            self._save_file(file, *args, **kwargs)

    def save(self, *args, **kwargs):
        atomic_save_successful = True
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
    
    @property
    def filedir(self):
        return self.dirname + '/' + self.format_filename(self.filename)
    
    @property
    def tmp_filedir(self):
        return self.dirname + '/tmp_' + self.format_filename(self.filename)
    
    def format_filename(self, filename):
        return filename


class PeriodicSaveCallback(SaveCallback):
    def __init__(self, *args, period=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_calls = 0
        self.period = period
    
    def save(self, *args, override_period=False, **kwargs):
        self.n_calls += 1
        saved = False
        if self.n_calls % self.period == 0 or override_period:
            super().save(*args, **kwargs)
            saved = True
        return saved


class PickleSave(SaveCallback):
    def __init__(self, *args, protocol=pkl.HIGHEST_PROTOCOL, **kwargs):
        super().__init__(*args, **kwargs)
        self.protocol = protocol

    def _save_file(self, file, obj):
        pkl.dump(obj, file, protocol=self.protocol)


class CSVSave:
    def __init__(self):
        pass
    
    def _save_file(self, file, doc):
        pass


class ModelCheckpoint(PeriodicSaveCallback, PickleSave):
    def __init__(self, *args,
                 save_best_only=False, monitor='train_acc',
                 save_last=True,
                 save_checkpoint_every_period=True,
                 **kwargs):
        """
        Args:
            save_best_only (Boolean, optional, default=False): If True, a checkpoint of the model will be made at every period only if it is better than the previous checkpoint according to 'monitor'.
            monitor (String, optional, either 'train_acc' or 'valid_acc', default='train_acc'): Value to monitor if 'save_best_only' is True.
            save_last (Boolean, optional, default=True): In the case 'period' is not 1, if 'save_last' is True, a checkpoint will be saved at the end of the iteration, regarless if the period.
            save_checkpoint_every_period (Boolean, optional, default=True): If True, a checkpoint will be saved every periods.

        See PeriodicSaveCallback and PickleSave documentation for other arguments.

        By default, all files will be overwritten at each save. However, one can insert a '{step}' substring in the specified 'filename' that will be formatted with the step number before being saved to differenciate the files.
        """
        super().__init__(*args, **kwargs)
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.current_best = 0

        self.save_last = save_last
        self.save_checkpoint_every_period = save_checkpoint_every_period

    def format_filename(self, filename):
        return filename.format(step=self.manager.boosting_round.round_number)

    def on_boosting_round_end(self):
        if self.save_checkpoint_every_period:
            if self.save_best_only:
                if self.manager.boosting_round.round_log[self.monitor] > self.current_best:
                    if self.save(self.manager.model):
                        self.current_best = self.manager.boosting_round.round_log[self.monitor]
            else:
                self.save(self.manager.model)
    
    def on_fit_end(self):
        if self.save_last:
            if self.save_best_only:
                if self.manager.boosting_round.round_log[self.monitor] > self.current_best:
                    self.current_best = self.manager.boosting_round.round_log[self.monitor]
                    self.save(self.manager.model, override_period=True)
            else:
                self.save(self.manager.model, override_period=True)
            