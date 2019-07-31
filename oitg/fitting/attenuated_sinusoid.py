"""Fit an attenuated sinusoid with optional dead-time to data"""
import numpy as np
from scipy.signal import lombscargle
from .FitBase import FitBase


def fitting_function(t, p_dict):
    """attenuated sinusoid with dead time"""
    y = np.exp(-p_dict['rate'] * t) * \
        (p_dict['a'] * np.sin((t - p_dict['t_dead']) * p_dict['omega']
                              + p_dict['phi'])
         + p_dict['c_offset']) + p_dict['c_equ']
    # hold constant during dead time
    return np.where(t > np.min(t) + p_dict['t_dead'], y,
                    p_dict['c_offset'] + p_dict['c_equ'] +
                    p_dict['a'] * np.sin(p_dict['phi'])
                    )


def init_all(t, y, p_dict):
    """Initialise parameters under the general assumptions that:
    t & y are 1D arrays of real numbers
    associated data coordinates share the same index in t & y

    Note: Data may be in any order and arbitrarily spaced"""
    # p_dict is an instance of FitParameters
    # user-inintialised values are not overwritten
    # non user-initialised parameters are pre-initialised to 0

    # sort data by time (x-axis)
    mask = np.argsort(t)
    t = t[mask]
    y = y[mask]

    p_dict['t_dead'] = 0.0
    p_dict['rate'] = 0.0

    min_step = np.min(t[1:] - t[:-1])
    duration = t[-1] - t[0]
    f_max = 0.5 / min_step
    f_min = 0.5 / duration

    omega_list = 2 * np.pi * np.linspace(f_min, f_max, int(f_max / f_min))
    if ('c_equ' in p_dict.constant_parameter_names or
        'c_equ' in p_dict.initialised_parameter_names):
        temp = (y - p_dict['c_equ']) * np.exp(p_dict['rate'] * t)
        pgram = lombscargle(t, temp, omega_list, precenter=True)
    else:
        pgram = lombscargle(t, y, omega_list, precenter=True)
    p_dict['omega'] = omega_list[np.argmax(pgram)]

    temp = pgram[np.searchsorted(omega_list, p_dict['omega'])]
    p_dict['a'] = np.sqrt(temp * 4 / len(y))

    # having estimated 'omega', average over a period
    t_period = 2 * np.pi / p_dict['omega']
    p_dict['c_equ'] = np.mean(y[np.argwhere(t > t[-1] - 2 * t_period)])
    p_dict['c_offset'] = np.mean(y[np.argwhere(t < t[0] + t_period)]) - \
                         p_dict['c_equ']

    y0_norm = (y[0] - p_dict['c_equ'] - p_dict['c_offset']) / p_dict['a']
    if y0_norm <= -1.0:
        p_dict['phi'] = -np.pi / 2
    elif y0_norm >= 1.0:
        p_dict['phi'] = np.pi / 2
    else:
        p_dict['phi'] = np.arcsin(y0_norm)
    return p_dict  # input is changed due to mutability of dict!


def derived_params(p_dict, p_error_dict):
    """calculate commonly used derived parameters and their error

    This method neglects parameter covariances!"""
    p_dict['t_pi'] = p_dict['t_dead'] + np.pi / p_dict['omega']
    p_dict['t_pi/2'] = p_dict['t_dead'] + np.pi / 2 / p_dict['omega']
    p_dict['period'] = 2 * np.pi / p_dict['omega']
    p_dict['tau_decay'] = 1 / p_dict['rate']

    # this error calculation neglects parameter covariance!
    # may want to upgrade FitBase to make covariance matrix available.
    p_error_dict['t_pi'] = np.sqrt(p_error_dict['t_dead']**2 +
                                   (np.pi / p_dict['omega'] *
                                    (p_error_dict['omega'] / p_dict['omega'])
                                    )**2
                                   )
    p_error_dict['t_pi/2'] = np.sqrt(p_error_dict['t_dead']**2 +
                                     (np.pi / 2 / p_dict['omega'] *
                                      (p_error_dict['omega'] / p_dict['omega'])
                                      )**2
                                     )
    p_error_dict['period'] =  p_dict['period'] * \
                             (p_error_dict['omega'] / p_dict['omega'])
    p_error_dict['tau_decay'] = p_dict['tau_decay'] * \
                             (p_error_dict['rate'] / p_dict['rate'])

    return (p_dict, p_error_dict)



attenuated_sinusoid = FitBase(
    ["omega", "t_dead", "a", "c_offset", "c_equ", "rate", "phi"],
    fitting_function=fitting_function, parameter_initialiser=init_all,
    derived_parameter_function=derived_params,
    parameter_bounds={"omega": (0, np.inf),
                      "t_dead": (0, np.inf),
                      "a": (0, np.inf),
                      "c_offset": (-np.inf, np.inf),
                      "c_equ": (-np.inf, np.inf),
                      "rate": (0, np.inf),
                      "phi": (-np.inf, np.inf),  # allows fit to wrap phase
                      })


if __name__ == "__main__":
    # example and debugging
    omega = 8.3e3
    amp = 0.5
    rel_noise = 2e-1
    # fit does not work with significant offset as initial frequency guess breaks
    offset = -2
    equ = 1.0
    rate = 1e2
    phi = np.pi / 2

    n_init = 25
    t_init = 1.3 * 2 * np.pi / omega

    n_end = 35
    t_delay = 1 / rate
    t_end = 2.8 * 2 * np.pi / omega

    temp0 = t_init * np.random.rand(n_init)
    temp1 = t_delay + t_end * np.random.rand(n_end)
    t = np.concatenate((temp0, temp1))
    y = np.sin(t * omega + phi) + rel_noise * np.random.rand(len(t))
    y *= amp
    y += offset
    y *= np.exp(-rate * t)
    y += equ

    # fix these fit parameters to a specific value
    const_dict = {
        't_dead': 0.0,
        'phi': phi,
        'c_equ': equ,
        # 'c_equ': equ
    }
    init_dict = {
        'c_offset': offset,
        'rate': rate,
    }
    p, p_err, x_fit, y_fit = attenuated_sinusoid.fit(
        t, y,
        y_err=np.ones(y.shape) * np.sqrt(1/3 - 1/4) * amp * rel_noise,
        initialise=init_dict,
        evaluate_function=True, evaluate_x_limit=[0, t_end + t_delay],
        constants=const_dict)

    print("done")
    if True:
        print("p", p)
        print("p_err", p_err)
    if True:
        from matplotlib import pyplot as plt
        mask = np.argsort(t)
        t = t[mask]
        y = y[mask]

        plt.figure()
        plt.errorbar(t, y,
                     yerr=np.ones(y.shape) * np.sqrt(1/3 - 1/4) * \
                          amp * rel_noise,
                     ecolor='k',
                     label="input")
        plt.plot(x_fit, y_fit, color="y", label="fit")
        plt.legend()
        plt.show()
    if False:  # frequency initial guess introspection (fit debugging)
        mask = np.argsort(t)
        t = t[mask]
        y = y[mask]

        min_step = np.min(t[1:] - t[:-1])
        duration = t[-1] - t[0]
        f_max = 0.5 / min_step
        f_min = 0.5 / duration

        omega_list = 2 * np.pi * np.linspace(f_min, f_max, int(f_max / f_min))
        periodigram = lombscargle(t, y, omega_list, precenter=True)
        plt.figure()
        plt.plot(omega_list, periodigram)
        plt.xlim(0, omega * 5)
        plt.show()
