import time
import sys, os
sys.path.append(os.getcwd())

from callbacks import Callback


class Progression(Callback):
    """
    This class format a readable output to the console to follow the progression of the training of the QuadBoost algorithm. It outputs a string in the format:
        Boosting round ### | Train acc: #.### | Valid acc: #.### | Time: #.##s

    It omits 'Valid acc' if none was used in the algorithm.
    """
    def on_step_begin(self):
        self.start_time = time.time()

    def on_step_end(self):
        # Round number
        output = [f'Boosting round {self.manager.step.step_number+1:03d}']

        # Train accuracy
        if self.manager.step.train_acc is not None:
            output.append(f'Train acc: {self.manager.step.train_acc:.3f}')
        else:
            output.append('Train acc: ?.???')

        # Valid accuracy
        if self.manager.step.valid_acc is not None:
            output.append(f'Valid acc: {self.manager.step.valid_acc:.3f}')

        # Risk
        if self.manager.step.risk is not None:
            output.append(f'Risk: {self.manager.step.risk:.3f}')

        # Time
        self.end_time = time.time()
        output.append(f'Time {self.end_time-self.start_time:.2f}s')

        sys.stdout.write(' | '.join(output) + '\n')
        sys.stdout.flush()
