import sys, os
sys.path.append(os.getcwd())


try:
    from quadboost.callbacks import PeriodicSaveCallback, PickleSaveMixin
except ModuleNotFoundError:
    from callbacks import PeriodicSaveCallback, PickleSaveMixin


class ModelCheckpoint(PeriodicSaveCallback, PickleSaveMixin):
    """
    This class will make a checkpoint of the whole QuadBoost object in a Pickle, which can be loaded later.
    """
    def __init__(self, *args,
                 save_best_only=False, monitor='train_acc',
                 save_last=True,
                 save_checkpoint_every_period=True,
                 overwrite_old_save=True,
                 **kwargs):
        """
        Args:
            save_best_only (Boolean, optional): If True, a checkpoint of the model will be made at every period only if it is better than the previous checkpoint according to 'monitor'.
            monitor (String, optional, either 'train_acc' or 'valid_acc'): Value to monitor if 'save_best_only' is True.
            save_last (Boolean, optional): In the case 'period' is not 1, if 'save_last' is True, a checkpoint will be saved at the end of the iteration, regarless if the period.
            save_checkpoint_every_period (Boolean, optional): If True, a checkpoint will be saved every periods.
            overwrite_old_save (Boolean, optional): If True, each time a checkpoint is made, the old checkpoint will be erased, even if the filename is different. To keep all models, set this parameter to False.

        See PeriodicSaveCallback and PickleSaveMixin documentation for other arguments.

        By default, all files will be overwritten at each save. However, one can insert a '{round}' substring in the specified 'filename' that will be formatted with the round number before being saved to differenciate the files.
        """
        super().__init__(*args, **kwargs)
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.current_best = 0

        self.save_last = save_last
        self.save_checkpoint_every_period = save_checkpoint_every_period
        self.overwrite_old_save = overwrite_old_save

    def format_filename(self, filename):
        return filename.format(round=self.manager.step.step_number+1)

    def on_iteration_begin(self):
        self.old_filedir = self.filedir

    def on_step_end(self):
        if self.save_checkpoint_every_period:
            if self.save_best_only:
                if getattr(self.manager.step, self.monitor) > self.current_best:
                    if self.save(self.manager.caller):
                        self.current_best = getattr(self.manager.step, self.monitor)
            else:
                self.save(self.manager.caller)

            self.erase_old_save()

    def on_iteration_end(self):
        if self.save_last:
            if self.save_best_only:
                if getattr(self.manager.step, self.monitor) > self.current_best:
                    self.current_best = getattr(self.manager.step, self.monitor)
                    self.save(self.manager.caller, override_period=True)
            else:
                self.save(self.manager.caller, override_period=True)

            self.erase_old_save()

    def erase_old_save(self):
        if self.overwrite_old_save and (self.filedir != self.old_filedir):
            try:
                os.remove(self.old_filedir)
            except FileNotFoundError: pass

        self.old_filedir = self.filedir
