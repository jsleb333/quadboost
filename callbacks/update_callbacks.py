import time as t
import sys, os
sys.path.append(os.getcwd())

from callbacks import Callback


class UpdateCallback(Callback):
    pass


class Progression(UpdateCallback):
    def on_iteration_begin(self):
        self.start_time = t.time()

    def on_step_end(self):
        # Round number
        output = [f'Boosting round {self.manager.step_number+1:03d}']

        # Train accuracy
        if self.manager.step.train_acc is not None:
            output.append(f'Train acc: {self.manager.step.train_acc:.2f}')
        else:
            output.append('Train acc: ?.???')

        # Valid accuracy
        if self.manager.step.valid_acc is not None:
            output.append(f'Valid acc: {self.manager.step.valid_acc:.2f}')

        # Time
        self.end_time = t.time()
        output.append(f'Time {self.end_time-self.start_time:.2f}s')
        self.start_time = self.end_time

        sys.stdout.write(' | '.join(output) + '\n')
        sys.stdout.flush()
