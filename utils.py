import os
import numpy as np
import matplotlib.pyplot as plt
from functools import wraps


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


def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        from time import time
        t0 = time()
        try:
            func_return = func(*args, **kwargs)
            name = "of '" + func.__name__ + "' " if func.__name__ != 'main' else ''
            t1 = time() - t0
        except:
            t1 = time() - t0
            print(f'\nExecution terminated after {t1:.2f} seconds.\n')
            raise
        print(f'\nExecution {name}completed in {t1:.2f} seconds.\n')
        return func_return
    return wrapper


if __name__ == '__main__':
    timed(compute_subplots_shape)(4)