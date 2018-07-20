import sys, os
sys.path.append(os.getcwd())

from callbacks import CallbackList, BreakOnMaxStep


class Step:
    """
    Simple Step class. Step subclasses should define the __next__ method, which is called by the __next__ method of the IteratorManager.

    This class only returns the step number of the iteration.
    """
    def __init__(self, manager=None):
        self.manager = manager

    def __next__(self):
        return self.manager.step_number    


class IteratorManager:
    """
    Class that manages an iterator using callbacks at 4 different moments in the iteration: on_iteration_begin, on_iteration_end, on_step_begin, on_step_end.

    The iteration is managed with BreakCallbacks which should raise a StopIteration exception on_step_begin or on_step_end when a condition is not satisfied.
    An iteration procedure can be launched by calling the 'iterate' method.

    At each step, __next__ will return an object. By default, it returns the step number.
    """
    def __init__(self, caller=None, callbacks=None, step=None):
        """
        Args:
            caller (Object which creates an IteratorManager, optional): Reference to the caller object. If the callbacks do not use the attributes of the caller, it can be omitted.
            callbacks (Iterable of Callback objects, optional): Callbacks handles functions to call at specific time in the program. Usage examples: stop the iteration or save the caller or the logs.
            step (Object with __next__ method defined, optional): Each __next__ call of IteratorManager will return the object returned by next(step). If None, the step number will be returned.
        """
        self.caller = caller
        self.callbacks = CallbackList(manager=self, callbacks=callbacks or [])
        self.step = step or Step(self)

    def __iter__(self):
        """
        The callback 'on_iteration_begin' is called here.
        """
        if self.callbacks.break_callbacks == []:
            raise RuntimeError('Callbacks should include at least one BreakCallback, else it would result in an infinite loop.')

        self.step_number = -1
        self.first_step = True
        self.callbacks.on_iteration_begin()
        
        return self

    def iterate(self, max_step_number=None):
        """
        Initialize an iteration procedure. The iterator is itself and yields the step number if no 'step' callable was given in the constructor. The iterator stops the iteration when a BreakCallback raise a StopIteration exception.

        This method is useful to initialize an iteration with arguments since iter does not allow it.

        Args:
            max_step_number (int, optional): If max_step_number is not None, the IteratorManager will act like the standard 'range' function with callbacks.
        """
        if max_step_number:
            self.callbacks.append(BreakOnMaxStep(max_step_number))

        return self

    def __next__(self):
        """
        Steps to next iteration. Callbacks 'on_step_begin', 'on_step_end' and 'on_iteration_end' are called here.

        Returns the object returned by next(self.step).
        """
        try:
            if not self.first_step:
                self.callbacks.on_step_end()
            else:
                self.first_step = False

            self.callbacks.on_step_begin()

        except StopIteration:
            self.callbacks.on_iteration_end()
            raise

        self.step_number += 1
        return next(self.step)


if __name__ == '__main__':
    from callbacks import BreakOnMaxStep
    a = 0
    safe = 0
    max_step_number = 10

    bi = IteratorManager()
    for br in bi.iterate(max_step_number):
        print(br)

        safe += 1
        if safe >= 100:
            break
