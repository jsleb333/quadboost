import os, sys
import numpy as np
import matplotlib.pyplot as plt
import functools
import multiprocessing as mp
from time import time
from datetime import datetime as dt
import argparse
import inspect
import warnings


class PicklableExceptionWrapper:
    """
    Wraps an Exception object to make it picklable so that the traceback follows. Useful for multiprocessing when an exception is raised in a subprocess.
    """
    def __init__(self, exception):
        self.exception = exception
        full_exception = sys.exc_info()
        try:
            import tblib.pickling_support
            tblib.pickling_support.install()
            self.traceback = full_exception[2]
        except ModuleNotFoundError:
            warnings.warn('Traceback of original error could not be carried from subprocess. If you want the full traceback, you should consider install the tblib module. A print of it follows.')
            import traceback
            traceback.print_exception(*full_exception)
            self.traceback = None

    def raise_exception(self):
        if self.traceback:
            raise self.exception.with_traceback(self.traceback)
        else:
            raise self.exception


def safe_queue_to_list(queue):
    items = []
    for _ in range(queue.qsize()):
        item = queue.get()
        if issubclass(type(item), PicklableExceptionWrapper):
            item.raise_exception()
        items.append(item)

    return items


def parse(func):
    """
    Quick and dirty way to make any main with optional keyword arguments parsable from the command line.
    """
    @functools.wraps(func)
    def wrapper(**kwargs):
        # Get default kwargs
        signature_kwargs = {k:v.default for k, v in inspect.signature(func).parameters.items()}
        # Update default values with values of caller
        signature_kwargs.update(kwargs)
        # Parse kwargs
        parser = argparse.ArgumentParser()
        for key, value in signature_kwargs.items():
            parser.add_argument(f'--{key}', dest=key, default=value, type=type(value))
        kwargs = vars(parser.parse_args())
        # Returns the original func with new kwargs
        return func(**kwargs)
    return wrapper


def to_one_hot(Y):
    labels = set(Y)
    n_classes = len(labels)
    Y_one_hot = np.zeros((len(Y), n_classes))
    for i, label in enumerate(Y):
        Y_one_hot[i,label] = 1

    return Y_one_hot


def compute_subplots_shape(N, aspect_ratio=9/16):
    """
    Returns the shape (n, m) of the subplots that will fit N images with respect to the given aspect_ratio.
    """
    if aspect_ratio == 0:
        return N, 1

    n = int(np.sqrt(aspect_ratio*N))
    m = int(np.sqrt(1/aspect_ratio*N))

    while m*n < N:
        if n/m <= aspect_ratio:
            n += 1
        else:
            m += 1

    return n, m


def make_fig_axes(N, aspect_ratio=9/16):
    n, m = compute_subplots_shape(N)
    fig, axes = plt.subplots(n, m)

    # Reshaping axes
    if n == 1 and m == 1:
        axes = [[axes]]
    elif n == 1 or m == 1:
        axes = [axes]
    axes = [ax for line_axes in axes for ax in line_axes]
    for ax in axes[N:]:
        ax.axis('off')

    return fig, axes[:N]


def split_int(n, k):
    """
    Equivalent of numpy 'array_split' function, but for integers instead of arrays.
    Returns n%k tuples of integers with difference equal to (n//k) + 1 and k - n%k tuples of integers with difference equal to n//k.
    """
    idx0, idx1 = 0, 0
    for i in range(k):
        idx0 = idx1
        idx1 = idx1 + n//k
        if i < n%k:
            idx1 += 1
        yield (idx0, idx1)


def parallelize(func, func_args, n_jobs):
    with mp.Pool(n_jobs) as pool:
        parallel_return = pool.map(func, func_args)
    return parallel_return


def timed(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t = time()
        time_format = '%Y-%m-%d %Hh%Mm%Ss'
        func_name = "of '" + func.__name__ + "' " if func.__name__ != 'main' else ''
        print(f'Execution {func_name}started on {dt.now().strftime(time_format)}.\n')
        try:
            func_return = func(*args, **kwargs)
        except:
            print(f'\nExecution terminated after {time()-t:.2f} seconds on {dt.now().strftime(time_format)}.\n')
            raise
        print(f'\nExecution {func_name}completed in {time()-t:.2f} seconds on {dt.now().strftime(time_format)}.\n')
        return func_return
    return wrapper


class ComparableMixin:
    """
    Mixin class that delegates the rich comparison operators to the specified attribute.

    Note: Uses __init_subclass__ as a work around for a bug with the 'Queue' class of 'multiprocessing' when parallelizing.
    """
    def __init_subclass__(cls, *, cmp_attr):
        def get_cmp_attr(self): return getattr(self, cmp_attr)
        cls.cmp_attr = property(get_cmp_attr)

        for operator_name in ['__eq__', '__ne__', '__lt__', '__le__', '__gt__', '__ge__']:
            def operator_func(self, other, operator_name=operator_name):
                other_attr = other.cmp_attr if hasattr(other, 'cmp_attr') else other
                try:
                    return getattr(self.cmp_attr, operator_name)(other_attr)
                except TypeError:
                    return NotImplemented

            setattr(cls, operator_name, operator_func)


if __name__ == '__main__':
    timed(split_int)(10,2)
