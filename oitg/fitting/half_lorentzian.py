import numpy as np
from . import FitBase


def parameter_initialiser(x, y, p):

    p['y0'] = np.mean(y)

    # Estimate the sign of the curve
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

    # Estimate the FWHM
    # In most cases the this initial parameter is a good guess
    # since most data-sets are sampled so that this is the case
    p['fwhm'] = (1 / 5) * (np.max(x) - np.min(x))


def fitting_function_left(x, p):

    y = p['a'] * (0.5 * p['fwhm'])**2
    y /= (x - p['x0'])**2 + (0.5 * p['fwhm'])**2
    y += p['y0']

    y = np.where(x < p['x0'], y, p['y0'])

    return y


def fitting_function_right(x, p):

    y = p['a'] * (0.5 * p['fwhm'])**2
    y /= (x - p['x0'])**2 + (0.5 * p['fwhm'])**2
    y += p['y0']

    y = np.where(x > p['x0'], y, p['y0'])

    return y


half_lorentzian_left = FitBase.FitBase(['x0', 'y0', 'a', 'fwhm'],
                                       fitting_function_left,
                                       parameter_initialiser=parameter_initialiser)

half_lorentzian_right = FitBase.FitBase(['x0', 'y0', 'a', 'fwhm'],
                                        fitting_function_right,
                                        parameter_initialiser=parameter_initialiser)
