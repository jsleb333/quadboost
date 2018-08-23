import multiprocessing as mp
from multiprocessing.queues import Queue
import warnings


class PicklableExceptionWrapper:
    """
    Wraps an Exception object to make it picklable so that the traceback follows. Useful for multiprocessing when an exception is raised in a subprocess.
    """
    def __init__(self, exception_type=None, exception_value=None, traceback=None):
        full_exception = (exception_type, exception_value, traceback)
        self.exception_value = exception_value
        self.traceback = traceback
        try:
            import tblib.pickling_support
            tblib.pickling_support.install()
        except ModuleNotFoundError:
            warnings.warn('Traceback of original error could not be carried from subprocess. If you want the full traceback, you should consider install the tblib module. A print of it follows.')
            import traceback
            traceback.print_exception(*full_exception)
            self.traceback = None

    def raise_exception(self):
        if self.traceback:
            raise self.exception_value.with_traceback(self.traceback)
        else:
            raise self.exception_value


class SafeQueue(Queue):
    def __init__(self):
        ctx = mp.context._default_context.get_context()
        super().__init__(maxsize=0, ctx=ctx)

    def __enter__(self):
        return self

    def __exit__(self, *full_exception):
        if full_exception[0] is not None:
            self.put(PicklableExceptionWrapper(*full_exception))

    def __iter__(self):
        for _ in range(len(self)):
            yield self.get()

    def get(self, *args, **kwargs):
        item = super().get(*args, **kwargs)
        if issubclass(type(item), PicklableExceptionWrapper):
            item.raise_exception()
        return item

    def __len__(self):
        return self.qsize()


def parallelize(func, func_args, n_jobs):
    with mp.Pool(n_jobs) as pool:
        parallel_return = pool.map(func, func_args)
    return parallel_return
