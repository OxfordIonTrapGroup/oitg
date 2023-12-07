import numpy as np
import numpy.fft
from . import FitBase


def parameter_initialiser(x, y, p):

    p['y0'] = np.mean(y)
    p['x0'] = 0
    p['a'] = (np.max(y) - np.min(y)) / 2

    if y[np.argmax(x)] > y[np.argmin(x)]:
        p['m'] = 1
    else:
        p['m'] = -1


def fitting_function(x, p):

    y = p['a'] * np.tanh((x - p['x0']) / p['m'])
    y += p['y0']

    return y


# Hyperbolic tan
tanh = FitBase.FitBase(['x0', 'y0', 'a', 'm'],
                       fitting_function,
                       parameter_initialiser=parameter_initialiser)
