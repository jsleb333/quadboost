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
    """
    multiprocessing Queue wrapper that mimicks a Python list and that is safe if an exception occurs within a subprocess.
    """
    def __init__(self):
        ctx = mp.context._default_context.get_context()
        super().__init__(maxsize=0, ctx=ctx)

    def __enter__(self):
        return self

    def __exit__(self, *full_exception):
        if full_exception[0] is not None:
            self.append(PicklableExceptionWrapper(*full_exception))

    def __iter__(self):
        for _ in range(len(self)):
            yield self.pop()
    
    def pop(self, block=True, timeout=None):
        """
        Returns last item from queue safely.
        """
        if len(self) > 0:
            item = self.get(block, timeout)
            if issubclass(type(item), PicklableExceptionWrapper):
                item.raise_exception()
            return item
        else:
            raise IndexError('pop from empty queue.')

    def __len__(self):
        return self.qsize()
    
    def append(self, item):
        self.put(item)


def parallelize(func, func_args, n_jobs):
    with mp.Pool(n_jobs) as pool:
        parallel_return = pool.map(func, func_args)
    return parallel_return


def parallel_processes(func_target, func_args_iter):
    processes = []
    for args in func_args_iter:
        process = mp.Process(target=func_target, args=args)
        processes.append(process)

    for process in processes: process.start()
    for process in processes: process.join()
    
    return processes


def dummy_parallel(queue, i):
    queue.append(i)


if __name__ == '__main__':
    import time

    n_processes = 2
    queue = SafeQueue()
    processes = []
    for i in range(n_processes):
        process = mp.Process(target=dummy_parallel, args=(queue, i))
        processes.append(process)

    for process in processes: process.start()
    for process in processes: process.join()
    
    for i in range(n_processes+1):
        print(queue.pop())