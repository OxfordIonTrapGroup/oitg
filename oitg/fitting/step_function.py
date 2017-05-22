
import numpy as np
from . import FitBase


def parameter_initialiser(x, y, p):

    p['y_start'] = y[np.argmin(x)]
    p['y_end'] = y[np.argmax(x)]
    p['x0'] = x[int((len(x) - 1) / 2 -1)]


def fitting_function(x, p):

    y =  np.where(x <= p['x0'], p['y_start'], p['y_end'])
    return y


step_function = FitBase.FitBase(
    ['y_start', 'y_end', 'x0'],
    fitting_function,
    parameter_initialiser=parameter_initialiser
)
