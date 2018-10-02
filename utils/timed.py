from datetime import datetime as dt
from time import time
import functools
try:
    from colorama import Fore, Style, init
    init()
except ModuleNotFoundError:
    # Emulate the Fore and Style class of colorama with a class that as an empty string for every attributes.
    class EmptyStringAttrClass:
        def __getattr__(self, attr): return ''
    Fore = EmptyStringAttrClass()
    Style = EmptyStringAttrClass()


class timed:
    """
    Wrapper class to time a function and display it in a colorful fashion. The wrapper takes optional keyword arguments to customize the display.
    """
    def __init__(self, func=None, *, datetime_format='%Y-%m-%d %Hh%Mm%Ss',
                 display_func_name=True,
                 main_color='LIGHTYELLOW_EX',
                 exception_exit_color='LIGHTRED_EX',
                 func_name_color='LIGHTBLUE_EX',
                 time_color='LIGHTCYAN_EX',
                 datetime_color='LIGHTMAGENTA_EX'):
        """
        Args:
            func (callable or None): Function or method to time.
            datetime_format (str or None, optional): Datetime format used to display the date and time. The format follows the template of the 'datetime' package. If None, no date or time will be displayed.
            display_func_name (bool): Whether or not the function name should be displayed.
            main_color (str): Color in which the main text will be displayed. Choices are those from the package colorama.
            exception_exit_color (str): Color in which the exception text will be displayed. Choices are those from the package colorama.
            func_name_color (str): Color in which the function name will be displayed. Choices are those from the package colorama.
            time_color (str): Color in which the time taken by the function will be displayed. Choices are those from the package colorama.
            datetime_color (str): Color in which the date and time of day will be displayed. Choices are those from the package colorama.

        Supported colors:
            BLACK, WHITE, RED, BLUE, GREEN, CYAN, MAGENTA, YELLOW, LIGHTRED_EX, BLIGHTLUE_EX, GRLIGHTEEN_EX, CLIGHTYAN_EX, MAGELIGHTNTA_EX, YELLIGHTLOW_EX,

        Example 1:
            >>> from utils import timed
            >>> @timed
            ... def foo():
            ...     print('foo!')
            ...
            >>> foo()
            Execution of 'foo' started on 2018-09-10 20h25m06s.

            foo!

            Execution of 'foo' completed in 0.00 seconds on 2018-09-10 20h25m06s.

        Example 2:
            >>> @timed(datetime_format='%Hh%Mm%Ss', display_func_name=False, main_color='WHITE')
            ... def bar():
            ...     print('bar!')
            ...     raise RuntimeError
            ...
            >>> bar()
            Execution started on 20h25m06s.

            bar!

            Execution terminated after 0.00 seconds on 20h25m06s.

            Traceback (most recent call last):
            File "...", line ..., in module bar()
            ...
                raise RuntimeError
            RuntimeError
        """
        self.func = func
        if func is not None:
            functools.update_wrapper(self, self.func)
        self.start_time = None
        self.datetime_format = datetime_format
        self.display_func_name = display_func_name

        self.main_color = getattr(Fore, main_color)
        self.exception_exit_color = getattr(Fore, exception_exit_color)
        self.func_name_color = getattr(Fore, func_name_color)
        self.time_color = getattr(Fore, time_color)
        self.datetime_color = getattr(Fore, datetime_color)

    def __call__(self, *args, **kwargs):
        if self.func is None:
            # self.func is None if timed was initialized with keyword arguments. In that case, the first call to __call__ is to initialize the function to wrap.
            self.func, = args
            functools.update_wrapper(self, self.func)
            return self
        else:
            return self.wrapper(*args, **kwargs)

    def wrapper(self, *args, **kwargs):
        """
        Actual method that wraps.
        """
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
        if self.display_func_name: #self.func.__name__ != 'main':
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
    def elapsed_time(self):
        return self.time_color + f'{time()-self.start_time:.2f}'

    def _start_timer(self):
        self.start_time = time()
        print(self.main_color
            + f'Execution {self.func_name}started{self.datetime}.\n'
            + Style.RESET_ALL)

    def _exception_exit_end_timer(self):
        print(self.exception_exit_color
            + f'\nExecution terminated after {self.elapsed_time}{self.exception_exit_color} seconds{self.datetime}{self.exception_exit_color}.\n'
            + Style.RESET_ALL)

    def _normal_exit_end_timer(self):
        print(self.main_color
            + f'\nExecution {self.func_name}completed in {self.elapsed_time}{self.main_color} seconds{self.datetime}.\n'
            + Style.RESET_ALL)


def old_timed(func):
    # For reference, this is an old simpler version of timed.
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t = time()
        datetime_format = '%Y-%m-%d %Hh%Mm%Ss'
        func_name = "of '" + func.__name__ + "' " if func.__name__ != 'main' else ''
        print(f'Execution {func_name}started on {dt.now().strftime(datetime_format)}.\n')
        try:
            func_return = func(*args, **kwargs)
        except:
            print(f'\nExecution terminated after {time()-t:.2f} seconds on {dt.now().strftime(datetime_format)}.\n')
            raise
        print(f'\nExecution {func_name}completed in {time()-t:.2f} seconds on {dt.now().strftime(datetime_format)}.\n')
        return func_return
    return wrapper


if __name__ == '__main__':
    @timed
    def foo():
        print('foo!')
    foo()

    @timed(datetime_format='%Hh%Mm%Ss', display_func_name=False, main_color='WHITE')
    def bar():
        print('bar!')
        raise RuntimeError
    bar()
