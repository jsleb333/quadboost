import sys, os
sys.path.append(os.getcwd())

try:
    from quadboost.callbacks import CallbackList
except ModuleNotFoundError:
    from callbacks import CallbackList


class Step:
    """
    Simple Step class which contains the current state of the step. The minimum state is the step number, which is used in many callbacks. Step subclasses can redefine the __next__ method to update the next step.

    The __next__ method of this class returns itself.
    """
    def __init__(self, step_number=0):
        self.step_number = step_number - 1

    def __next__(self):
        self.step_number += 1
        return self

    def __str__(self):
        return str(self.step_number)


class CallbacksManagerIterator:
    """
    Context manager to manage an iterator by using callbacks at 4 different moments in the iteration: 'on_iteration_begin', 'on_iteration_end', 'on_step_begin', 'on_step_end'. Additionally, it will call 'on_exception_exit' callbacks if an exception should occur during the iteration.

    The iteration is managed with BreakCallbacks which should raise a StopIteration exception 'on_step_begin' or 'on_step_end' when a condition is not satisfied.

    The manager should be called in a 'with' statement so that the callbacks are called. The __exit__ method also handles 'on_exception_exit' callbacks if the iteration was terminated by an exception, and will always call 'on_iteration_end'.

    On each iteration step, a Step object is returned.
    """
    def __init__(self, caller=None, callbacks=None, step=None, starting_step_number=0):
        """
        Args:
            caller (Object which creates an CallbacksManagerIterator, optional): Reference to the caller object. If the callbacks do not use the attributes of the caller, it can be omitted.
            callbacks (Iterable of Callback objects, optional): Callbacks handles functions to call at specific time in the program. Usage examples: stop the iteration or save the caller or the logs.
            step (Object with __next__ method defined, optional): Each __next__ call of CallbacksManagerIterator will return the object returned by next(step). If None, a Step object counting the number of iterations will be returned.
            starting_step_number (int, optional): Number of the step to start from. Only used if 'step' is None. Useful to resume an interrupted iteration.
        """
        self.caller = caller
        self.callbacks = CallbackList(manager=self, callbacks=callbacks or [])
        self.step = step or Step(starting_step_number)

    def __enter__(self):
        """
        The callback 'on_iteration_begin' is called here.
        """
        self.callbacks.on_iteration_begin()
        return self

    def __exit__(self, exception_type, exception_message, traceback):
        """
        The callback 'on_iteration_end' is called here.
        """
        if exception_type is not None:
            self.callbacks.on_exception_exit(exception_type, exception_message, traceback)
        self.callbacks.on_iteration_end()

    def __iter__(self):
        """
        Yields an iterator where the callbacks 'on_step_begin' and 'on_step_end' are called at the right time.
        """
        if not self.callbacks.break_callbacks:
            raise RuntimeError('Callbacks should include at least one BreakCallback, else it would result in an infinite loop.')

        while True:
            self.callbacks.on_step_begin()
            yield next(self.step)
            self.callbacks.on_step_end()


if __name__ == '__main__':
    from quadboost.callbacks import BreakOnMaxStepCallback
    a = 0
    safe = 0
    cb = [BreakOnMaxStepCallback(10)]

    with CallbacksManagerIterator(caller=None, callbacks=cb, starting_step_number=0) as bi:
        for br in bi:
            print(br)

            safe += 1
            if safe >= 100:
                break
