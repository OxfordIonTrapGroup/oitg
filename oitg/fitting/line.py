
import numpy as np
import numpy.fft
from . import FitBase


def parameter_initialiser(x, y, p):

    p['a'] = np.mean(y)
    p['b'] = 0


def fitting_function(x, p):

    y = p['a']
    y += p['b']*x

    return y

def derived_parameter_function(p, p_error):

    return p, p_error

parabola = FitBase.FitBase(['a', 'b', 'c'], fitting_function,
        parameter_initialiser=parameter_initialiser,
        derived_parameter_function=derived_parameter_function)
