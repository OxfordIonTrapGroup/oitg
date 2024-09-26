from . import FitBase


def parameter_initialiser(x, y, p):
    k = (y[-1] - y[0]) / (x[-1] - x[0])
    p['a'] = y[0] - x[0] * k
    p['b'] = k


def fitting_function(x, p):

    y = p['a']
    y += p['b'] * x

    return y


line = FitBase.FitBase(['a', 'b'],
                       fitting_function,
                       parameter_initialiser=parameter_initialiser)
