import numpy as np
import numpy.fft
from . import FitBase


def parameter_initialiser(x, y, p):

    p['a'] = np.mean(y)
    p['b'] = 0
    p['c'] = 0


def fitting_function(x, p):

    y = p['a']
    y += p['b'] * x
    y += p['c'] * x**2

    return y


def derived_parameter_function(p, p_error):

    # Calculate position of the extremum
    x_ext = -p['b'] / (2 * p['c'])

    # x_ext_error = x_ext*np.sqrt((p_error['b']/p['b'])**2
    #                            + (p_error['c']/p['c'])**2)

    y_ext = p['a']
    y_ext += p['b'] * x_ext
    y_ext += p['c'] * x_ext**2

    # y_ext_error =

    p['x_ext'] = x_ext
    p['y_ext'] = y_ext

    return p, p_error


parabola = FitBase.FitBase(['a', 'b', 'c'],
                           fitting_function,
                           parameter_initialiser=parameter_initialiser,
                           derived_parameter_function=derived_parameter_function)


def parameter_initialiser_shifted(x, y, p):
    order = np.argsort(x)

    x_first = x[order[0]]
    y_first = y[order[0]]

    center_idx = order[len(order) // 2]
    y_center = y[center_idx]

    x_last = x[order[-1]]
    y_last = y[order[-1]]

    if y_first > y_center < y_last:
        min_idx = np.argmin(y)
        p['position'] = x[min_idx]
        p['offset'] = y[min_idx]
        p['scale'] = ((y_first + y_last) / 2 - p['offset']) / (
            (x_last - x_first) / 2)**2
    elif y_first < y_center > y_last:
        max_idx = np.argmax(y)
        p['position'] = x[max_idx]
        p['offset'] = y[max_idx]
        p['scale'] = ((y_first + y_last) / 2 - p['offset']) / (
            (x_last - x_first) / 2)**2
    else:
        # Give up.
        p['position'] = x[order[len(order) // 2]]
        p['offset'] = 0
        p['scale'] = 0


def fitting_function_shifted(x, p):
    return (x - p['position'])**2 * p['scale'] + p['offset']


shifted_parabola = FitBase.FitBase(['position', 'scale', 'offset'],
                                   fitting_function_shifted,
                                   parameter_initialiser=parameter_initialiser_shifted)
