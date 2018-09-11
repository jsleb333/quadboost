import os, sys
import numpy as np
import matplotlib.pyplot as plt
import functools
<<<<<<< HEAD
from time import time, sleep
from datetime import datetime as dt
=======
>>>>>>> dev
import argparse
import inspect
from collections import defaultdict


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


<<<<<<< HEAD
# def timed(func):
#     @functools.wraps(func)
#     def wrapper(*args, **kwargs):
#         t = time()
#         datetime_format = '%Y-%m-%d %Hh%Mm%Ss'
#         func_name = "of '" + func.__name__ + "' " if func.__name__ != 'main' else ''
#         print(f'Execution {func_name}started on {dt.now().strftime(datetime_format)}.\n')
#         try:
#             func_return = func(*args, **kwargs)
#         except:
#             print(f'\nExecution terminated after {time()-t:.2f} seconds on {dt.now().strftime(datetime_format)}.\n')
#             raise
#         print(f'\nExecution {func_name}completed in {time()-t:.2f} seconds on {dt.now().strftime(datetime_format)}.\n')
#         return func_return
#     return wrapper


try:
    from colorama import Fore, Style, init
    init()
except ModuleNotFoundError:
    # Emulate the Fore class of colorama with a class that as an empty string for every attributes.
    class EmptyStringAttrClass:
        def __getattribute__(self, attr): return ''
    Fore = EmptyStringAttrClass()
    Style = EmptyStringAttrClass()


class timed:
    """
    Wrapper class to time a function. The wrapper takes optional keyword arguments to customize the display.
    """
    def __init__(self, func=None, *, datetime_format='%Y-%m-%d %Hh%Mm%Ss',
                 main_color='LIGHTYELLOW_EX',
                 exception_exit_color='LIGHTRED_EX',
                 func_name_color='LIGHTBLUE_EX',
                 time_color='LIGHTCYAN_EX',
                 datetime_color='LIGHTMAGENTA_EX'):
        """
        Args:
            func (callable or None): Function or method to time.
            datetime_format (str or None, optional): Datetime format used to display the date and time. The format follows the template of the 'datetime' package. If None, no date or time will be displayed.
            main_color (str): Color in which the main text will be displayed. Choices are those from the package colorama.
            exception_exit_color (str): Color in which the exception text will be displayed. Choices are those from the package colorama.
            func_name_color (str): Color in which the function name will be displayed. Choices are those from the package colorama.
            time_color (str): Color in which the time taken by the function will be displayed. Choices are those from the package colorama.
            datetime_color (str): Color in which the date and time of day will be displayed. Choices are those from the package colorama.

        Supported colors:
            BLACK, WHITE, RED, BLUE, GREEN, CYAN, MAGENTA, YELLOW, LIGHTRED_EX, BLIGHTLUE_EX, GRLIGHTEEN_EX, CLIGHTYAN_EX, MAGELIGHTNTA_EX, YELLIGHTLOW_EX,
        """
        self.func = func
        self.start_time = None
        self.datetime_format = datetime_format

        self.main_color = getattr(Fore, main_color)
        self.exception_exit_color = getattr(Fore, exception_exit_color)
        self.func_name_color = getattr(Fore, func_name_color)
        self.time_color = getattr(Fore, time_color)
        self.datetime_color = getattr(Fore, datetime_color)

    def __call__(self, *args, **kwargs):
        # If timed was called with keyword arguments, the first call is then to initialize the function to wrap.
        if self.func is None:
            self.func, = args
            return self
        else:
            self.wrapper(*args, **kwargs)

    def wrapper(self, *args, **kwargs):
        self._start_timer()
        try:
            return_value = self.func(*args, **kwargs)
        except:
            self._exception_exit_end_timer()
            raise
        self._normal_exit_end_timer()
        return return_value

    @property
    def func_name(self):
        if self.func.__name__ != 'main':
            return f"of '{self.func_name_color}{self.func.__name__}{self.main_color}' "
        else:
            return ''

    @property
    def datetime(self):
        if self.datetime_format is None:
            return ''
        else:
            return ' on ' + self.datetime_color + dt.now().strftime(self.datetime_format) + self.main_color

    @property
    def time(self):
        return self.time_color + f'{time()-self.start_time:.2f}'

    def _start_timer(self):
        self.start_time = time()
        print(self.main_color
            + f'Execution {self.func_name}started{self.datetime}.\n'
            + Style.RESET_ALL)

    def _exception_exit_end_timer(self):
        print(self.exception_exit_color
            + f'\nExecution terminated after {self.time}{self.exception_exit_color} seconds{self.datetime}{self.exception_exit_color}.\n'
            + Style.RESET_ALL)

    def _normal_exit_end_timer(self):
        print(self.main_color
            + f'\nExecution {self.func_name}completed in {self.time}{self.main_color} seconds{self.datetime}.\n'
            + Style.RESET_ALL)


=======
>>>>>>> dev
if __name__ == '__main__':
    @timed()
    def test(t=.1):
        print('hehe')
        sleep(t)

    test()
    test('a')
