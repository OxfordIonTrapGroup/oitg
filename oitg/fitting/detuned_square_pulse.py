import numpy as np
from scipy.signal import lombscargle
from . import FitBase


def parameter_initialiser(x, y, p):
    """assumes close to pi-pulse"""
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

    omega_list = 2 * np.pi * np.linspace(f_min, f_max, int(10 * f_max / f_min))
    # the periodogram should give the correct width up-to a factor of 2
    pgram = lombscargle(x, y, omega_list, precenter=True)
    # sqrt(3) factor derived from assuming pi pulse
    p["t_pulse"] = omega_list[np.argmax(pgram)] / np.sqrt(3)

    p['omega'] = np.pi / p["t_pulse"]
    p['y0'] = np.mean(y)
    y_diff = y - p['y0']
    peak_ind = np.argmax(np.abs(y_diff))
    p['offset'] = x[peak_ind]
    p['a'] = y[peak_ind] - p['y0']
    return p


def fitting_function(detuning, p):
    """returns values of the function

    f(detuning) = a*omega^2 * t_pulse^2 /4 * sinc^2(W*t_pulse/2) + y0
    where W = sqrt(omega^2 + detuning^2)
    """
    w = np.sqrt(p['omega']**2 + (detuning - p['offset'])**2)

    # beware! np.sinc(x) = sin(pi*x)/(pi*x)
    y = p['a'] * (p['omega'] * p['t_pulse'] / 2 * np.sinc(w * p['t_pulse'] /
                                                          (2 * np.pi)))**2
    y += p['y0']
    return y


def derived_params(p_dict, p_error_dict):
    # pulse area arror
    p_dict['t_error'] = p_dict['t_pulse'] - np.pi / p_dict['omega']
    # normalised to pi pulse
    p_dict['area_error'] = p_dict['t_pulse'] * p_dict['omega'] / np.pi - 1.0

    p_error_dict['t_error'] = np.sqrt(p_error_dict['t_pulse']**2 +
                                      (np.pi / p_dict['omega'] *
                                       (p_error_dict['omega'] / p_dict['omega']))**2)
    p_error_dict['area_error'] = p_dict['t_pulse'] * p_dict['omega'] / np.pi \
        * np.sqrt(
            (p_error_dict['t_pulse'] / p_dict['t_pulse']) ** 2 +
            (p_error_dict['omega'] / p_dict['omega']) ** 2
    )
    return (p_dict, p_error_dict)


# fitter
detuned_square_pulse = FitBase.FitBase(
    ['omega', 't_pulse', 'offset', 'a', 'y0'],
    fitting_function,
    parameter_initialiser=parameter_initialiser,
    derived_parameter_function=derived_params,
    parameter_bounds={
        "omega": (0, np.inf),
        "t_pulse": (0, np.inf),
        "a": (-np.inf, np.inf),
        "offset": (-np.inf, np.inf),
        "y0": (-np.inf, np.inf),
    })

if __name__ == '__main__':

    omega = 1e6
    t_pulse, offset, a, y0 = np.pi / omega + 1e-7, 2e3, 1.0, 0.0

    error = 0.01
    range = 10 * omega

    x = np.linspace(-range / 2, range / 2, 35)

    temp = np.sqrt(omega**2 + (x - offset)**2)
    y = np.sinc(temp * t_pulse / (2 * np.pi))**2
    y *= a * omega**2 * t_pulse**2 / 4
    y += y0
    y += np.random.normal(size=len(y), scale=error)

    p, p_err, x_fit, y_fit = detuned_square_pulse.fit(x,
                                                      y,
                                                      y_err=np.full(y.shape, error),
                                                      evaluate_function=True,
                                                      initialise={},
                                                      constants={})
    print(p)
    print(p_err)

    from matplotlib import pyplot as plt
    plt.figure()
    plt.plot(x, y)
    plt.plot(x_fit, y_fit)
    plt.show()

    if False:
        min_step = np.min(x[1:] - x[:-1])
        duration = x[-1] - x[0]
        # Nyquist limit does not apply to irregularly spaced data
        # We'll use it as a starting point anyway...
        f_max = 0.5 / min_step
        # relaxed Fourier limit
        f_min = 0.25 / duration

        omega_list = 2 * np.pi * np.linspace(f_min, f_max, int(10 * f_max / f_min))
        # the periodogram should give the correct width up-to a factor of 2
        pgram = lombscargle(x, y, omega_list, precenter=True)

        plt.figure()
        plt.plot(omega_list, pgram)
        plt.show()
