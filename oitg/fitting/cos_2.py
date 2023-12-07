import numpy as np
import numpy.fft
from . import FitBase


def parameter_initialiser(x, y, p):

    p['y0'] = np.mean(y)
    p['x0'] = 0
    p['a'] = (np.max(y) - np.min(y)) / 2
    p['period'] = np.max(x) - np.min(x)


def fitting_function(x, p):

    y = p['a'] * (np.cos(2 * np.pi * (x - p['x0']) / p['period'])**2)
    y += p['y0']

    return y


# Cosine^2 with 'dumb' initialiser
cos_2 = FitBase.FitBase(['x0', 'y0', 'a', 'period'],
                        fitting_function,
                        parameter_initialiser=parameter_initialiser)


def parameter_initialiser_fft(x, y, p):

    # Take the Fourier transform of the y axis
    y_ft = numpy.fft.rfft(y)

    # Find the maximum frequency component, apart from DC
    i_max = np.argmax(np.abs(y_ft[1:]))

    # Calculate which period this corresponds to
    period_sample = np.max(x) - np.min(x)
    p['period'] = 2 * period_sample / (i_max + 1.0)

    p['y0'] = np.mean(y)
    p['x0'] = 0
    p['a'] = (np.max(y) - np.min(y)) / 2


# Cosine^2 with initialiser which extracts the initial period with
# an fft, only works when the x-axis is regularly spaced
cos_2_fft = FitBase.FitBase(['x0', 'y0', 'a', 'period'],
                            fitting_function,
                            parameter_initialiser=parameter_initialiser_fft)
