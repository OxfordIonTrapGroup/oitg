import numpy as np
from . import FitBase


def parameter_initialiser(x, y, p):

    p['x0'] = np.mean(x)
    p['d'] = np.mean(y)
    p['k1'] = 0
    p['k2'] = 0


def fitting_function(x, p):
    y = np.where(x <= p['x0'], p['k1'] * (x - p['x0']) + p['d'],
                 p['k2'] * (x - p['x0']) + p['d'])
    return y


v_function = FitBase.FitBase(['k1', 'k2', 'd', 'x0'],
                             fitting_function,
                             parameter_initialiser=parameter_initialiser)
