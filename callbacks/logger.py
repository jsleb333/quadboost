import sys, os
sys.path.append(os.getcwd())

from callbacks import PeriodicSaveCallback, CSVSave


class CSVLogger(PeriodicSaveCallback, CSVSave):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log = []

    def on_iteration_begin(self):
        self.log.append([field for field in self.manager.step.__dict__])
        self.save(self.log)
        
    def on_step_end(self):
        self.log.append([value for value in self.manager.step.__dict__.values()])
        self.save(self.log)
        