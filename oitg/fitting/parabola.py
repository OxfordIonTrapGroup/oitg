
import numpy as np
import numpy.fft
from . import FitBase


def parameter_initialiser(x, y, p):

    p['a'] = np.mean(y)
    p['b'] = 0
    p['c'] = 0


def fitting_function(x, p):

    y = p['a']
    y += p['b']*x
    y += p['c']*x**2

    return y

def derived_parameter_function(p, p_error):

    # Calculate position of the extremum
    x_ext = -p['b']/(2*p['c'])

    # x_ext_error = x_ext*np.sqrt((p_error['b']/p['b'])**2
    #                            + (p_error['c']/p['c'])**2)

    y_ext = p['a']
    y_ext += p['b']*x_ext
    y_ext += p['c']*x_ext**2

    #y_ext_error = 

    return p, p_error

parabola = FitBase.FitBase(['a', 'b', 'c'], fitting_function,
        parameter_initialiser=parameter_initialiser,
        derived_parameter_function=derived_parameter_function)
