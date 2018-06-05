import os
import numpy as np


def to_one_hot(Y):
    labels = set(Y)
    n_classes = len(labels)
    Y_one_hot = np.zeros((len(Y), n_classes))
    for i, label in enumerate(Y):
        Y_one_hot[i,label] = 1
    
    return Y_one_hot


def subplots_shape(N, aspect_ratio=3/4):
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


def plot_spectrograms(spectrograms, tlim=[], flim=[], titles='', show=True):
    """
    Plots a single spectrogram or a list of spectrograms. If 'spectrograms' is a list, 'tlim', 'flim' and 'titles' must all be lists too.
    """
    
    # Management of one vs multiple spectrograms
    if type(spectrograms) is not list:
        spectrograms = [spectrograms]
    N = len(spectrograms)
    if type(titles) is str:
        titles = [titles]*N
    if type(titles) is not list:
        titles = [titles]
    if tlim == []:
        tlim = [[0,1]]*N
    if flim == []:
        flim = [[100,6000]]*N

    # Size of the subplots grid
    n, m = subplots_shape(N)

    fig, axes = plt.subplots(n, m)
    k = 0
    for i in range(n):
        for j in range(m):
            if n == 1 and m == 1:
                ax = axes
            elif n == 1:
                ax = axes[j]
            else:
                ax = axes[i,j]

            if k < N:
                spectrogram = spectrograms[k]
                xmin, xmax = tlim[i*m+j]
                ymin, ymax = flim[i*m+j]
                ax.imshow(spectrogram,
                          origin='lower',
                          cmap='PRGn',
                          extent=[xmin,xmax,ymin,ymax],
                          aspect=xmax/ymax,
                          )
                ax.set_title(titles[k])
                k += 1
            else:
                ax.axis('off')

    if show:
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        plt.show()
    return fig, axes


if __name__ == '__main__':
    for N in range(1,100):
        n, m = subplots_shape(N, 9/16)
        print(N, (n, m), n/m)