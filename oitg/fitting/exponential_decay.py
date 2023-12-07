import numpy as np
from . import FitBase


def parameter_initialiser(x, y, p):

    p['y0'] = y[np.argmin(x)]
    p['x0'] = 0
    p['y_inf'] = y[np.argmax(x)]
    p['tau'] = np.max(x) - np.min(x)


def fitting_function(x, p):

    y = p['y0'] + (p['y_inf'] - p['y0']) * (1 - np.exp(-(x - p['x0']) / p['tau']))

    # Function returns y0 for x <= x0
    y = np.where(x <= p['x0'], p['y0'], y)

    return y


def derived_parameter_function(p, p_err):
    p["t_1_e"] = p["x0"] + p["tau"]

    # This is just a Gaussian error propagation guess.
    p_err["t_1_e"] = np.sqrt(p_err["x0"]**2 + (p_err["tau"] / 2)**2)

    return p, p_err


# Exponential decay function parameterised such that:
# for x <= x0:
#   y = y0
# for x > x0:
#   y = y0 + (y_inf-y0)*(1-exp(-(x-x0)/tau))
#   so that for (x-x0) >> tau, y = y_inf
exponential_decay = FitBase.FitBase(
    ['x0', 'y0', 'y_inf', 'tau'],
    fitting_function,
    parameter_initialiser=parameter_initialiser,
    derived_parameter_function=derived_parameter_function)
