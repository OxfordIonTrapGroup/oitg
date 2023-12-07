import numpy as np
from scipy.signal import lombscargle
from . import FitBase


def parameter_initialiser(x, y, p):
    # Sort data by time (x-axis)
    mask = np.argsort(x)
    x = x[mask]
    y = y[mask]

    min_step = np.min(x[1:] - x[:-1])
    duration = x[-1] - x[0]
    # Nyquist limit does not apply to irregularly spaced data
    # We'll use it as a starting point anyway...
    f_max = 0.5 / min_step
    # relaxed Fourier limit
    f_min = 0.25 / duration

    omega_list = 2 * np.pi * np.linspace(f_min, f_max, int(f_max / f_min))
    # the periodogram should give the correct width up-to a factor of 2
    pgram = lombscargle(x, y, omega_list, precenter=True)
    p["width"] = omega_list[np.argmax(pgram)] / np.pi

    p['y0'] = np.mean(y)
    y_diff = y - p['y0']
    peak_ind = np.argmax(np.abs(y_diff))
    p['x0'] = x[peak_ind]
    p['a'] = y[peak_ind] - p['y0']
    return p


def fitting_function(x, p):
    """returns values of the function

    .. math::
        a*sinc($\\pi \frac{(x-x0)}{width}$)+y0
    """
    y = p['a'] * (np.sinc((x - p['x0']) / p['width']))**2
    y += p['y0']

    return y


def derived_params(p_dict, p_error_dict):
    # calculate width*pi used in conventional sinc definition
    p_dict['omega'] = p_dict['width'] * np.pi
    p_error_dict['omega'] = p_error_dict['width'] * np.pi
    return (p_dict, p_error_dict)


# Sinc^2 fitter
sinc_2 = FitBase.FitBase(['x0', 'y0', 'a', 'width'],
                         fitting_function,
                         parameter_initialiser=parameter_initialiser,
                         derived_parameter_function=derived_params)

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    error = 0.1
    range = 10
    offset = 3

    x = np.linspace(-range / 2, range / 2, 30)
    y = np.sinc(x - offset)**2
    y += np.random.normal(size=len(y), scale=error)

    p, p_err, x_fit, y_fit = sinc_2.fit(x,
                                        y,
                                        y_err=np.full(y.shape, error),
                                        evaluate_function=True)
    print(p)
    print(p_err)
    plt.figure()
    plt.plot(x, y)
    plt.plot(x_fit, y_fit)
    plt.show()
