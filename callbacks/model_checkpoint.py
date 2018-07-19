import sys, os
sys.path.append(os.getcwd())

from callbacks import SaveCallback, PickleSave


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
        return filename.format(step=self.manager.step.round_number)

    def on_step_end(self):
        if self.save_checkpoint_every_period:
            if self.save_best_only:
                if self.manager.step.round_log[self.monitor] > self.current_best:
                    if self.save(self.manager.caller):
                        self.current_best = self.manager.step.round_log[self.monitor]
            else:
                self.save(self.manager.caller)

    def on_iteration_end(self):
        if self.save_last:
            if self.save_best_only:
                if self.manager.step.round_log[self.monitor] > self.current_best:
                    self.current_best = self.manager.step.round_log[self.monitor]
                    self.save(self.manager.caller, override_period=True)
            else:
                self.save(self.manager.caller, override_period=True)
