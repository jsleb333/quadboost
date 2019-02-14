import os, sys
import numpy as np
import matplotlib.pyplot as plt
import functools
import argparse
import inspect


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
            value_type = type(value)
            if isinstance(value, bool):
                value_type = bool_parse
            if isinstance(value, list):
                if len(value) > 0:
                    list_type = type(value[0])
                else:
                    list_type = str
                value_type = list_parse(list_type)
            parser.add_argument(f'--{key}', dest=key, default=value, type=value_type)
        kwargs = vars(parser.parse_args())
        # Returns the original func with new kwargs
        return func(**kwargs)
    return wrapper


def bool_parse(arg):
    if arg.lower() in ('true', 't', 'yes', 'y', '1'):
        return True
    elif arg.lower() in ('false', 'f', 'no', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def list_parse(list_type):
    print(list_type)
    def _list_parse(arg):
        arg = arg.replace('[', '').replace(']','')
        arg = arg.split(',')
        return [list_type(value) for value in arg]
    return _list_parse


def identity_func(arg):
    return arg


def identity_method(self, arg):
    return arg


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


if __name__ == '__main__':
    print([i for i in split_int(10,3)])
