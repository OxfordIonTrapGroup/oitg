import numpy as np
from . import FitBase


def parameter_initialiser(x, y, p):

    p['y0'] = np.mean(y)

    # Estimate the sign of the gaussian
    dy_min = p['y0'] - np.min(y)
    dy_max = np.max(y) - p['y0']

    if dy_max >= dy_min:
        # The peak should be positive
        p['a'] = dy_max
        p['x0'] = x[np.argmax(y)]
    else:
        # The peak should be negative
        p['a'] = dy_min
        p['x0'] = x[np.argmin(y)]

    # Estimate the sigma
    # In most cases the this initial parameter is a good guess
    # since most data-sets are sampled so that this is the case
    p['sigma'] = (1 / 5) * (np.max(x) - np.min(x))


def fitting_function(x, p):

    y = p['a'] * np.exp(-0.5 * ((x - p['x0']) / p['sigma'])**2)
    y += p['y0']

    return y


def derived_parameter_function(p, p_error):

    # Calculate the FWHM from the sigma
    p['fwhm'] = 2.35482 * p['sigma']
    p_error['fwhm'] = 2.35482 * p_error['sigma']

    return p, p_error


gaussian = FitBase.FitBase(['x0', 'y0', 'a', 'sigma'],
                           fitting_function,
                           parameter_initialiser=parameter_initialiser,
                           derived_parameter_function=derived_parameter_function)
