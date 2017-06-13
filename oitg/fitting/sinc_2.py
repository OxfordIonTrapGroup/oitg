
import numpy as np
import numpy.fft
from . import FitBase


def parameter_initialiser(x, y, p):

    p['y0'] = np.mean(y)
    p['x0'] = 0
    p['a'] = (np.max(y) - np.min(y))/2
    #This sets the initial FWHM to be approx half the x range
    p['width'] = (np.max(x) - np.min(x))/6


def fitting_function(x, p):

    #sinc^2 fitting function. Here the width is approx FWHM/2.78
    y = p['a']*(np.sinc((x-p['x0'])/p['width']))**2
    y += p['y0']

    return y


# Sinc^2 with 'dumb' initialiser
sinc_2 = FitBase.FitBase(['x0', 'y0', 'a', 'width'], fitting_function,
                        parameter_initialiser=parameter_initialiser)
