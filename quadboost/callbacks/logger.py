import sys, os
sys.path.append(os.getcwd())


try:
    from quadboost.callbacks import PeriodicSaveCallback, CSVSaveMixin
except ModuleNotFoundError:
    from callbacks import PeriodicSaveCallback, CSVSaveMixin


class CSVLogger(PeriodicSaveCallback, CSVSaveMixin):
    """
    This class will save a complete log of the 'step' object of the CallbacksManagerIterator into a CSV. For more flexibility, define your own 'Step' class.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log = []

    def on_iteration_begin(self):
        if not self.log:
            self.log.append([field for field in self.manager.step.__dict__])
            self.save(self.log)

    def on_step_end(self):
        self.log.append([value for value in self.manager.step.__dict__.values()])
        self.save(self.log)
