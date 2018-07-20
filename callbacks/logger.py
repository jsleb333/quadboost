import sys, os
sys.path.append(os.getcwd())

from callbacks import PeriodicSaveCallback, CSVSave


class CSVLogger(PeriodicSaveCallback, CSVSave):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def on_iteration_begin(self):
        row = [field for field in self.manager.step.__dict__]
        self.save(row)
        self.open_mode = 'a'
        
    def on_step_end(self):
        row = [value for value in self.manager.step.__dict__.values()]
        self.save(row)
        