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

    # sort data by time (x-axis)
    mask = np.argsort(t)
    t = t[mask]
    y = y[mask]

    p_dict['t_dead'] = 0.0
    p_dict['rate'] = 0.0
    # is there a reliable way to make lombscargle
    # non-senisitve to a known exponential decay?
    # ex: if c_equ and rate are known
    #     (y - c_equ) * exp(r*t) would only leave sin and dc offset

    min_step = np.min(t[1:] - t[:-1])
    duration = t[-1] - t[0]
    f_max = 0.5 / min_step
    f_min = 0.5 / duration

    omega_list = 2 * np.pi * np.linspace(f_min, f_max, int(f_max / f_min))
    pgram = lombscargle(t, y, omega_list, precenter=True)
    p_dict['omega'] = omega_list[np.argmax(pgram)]

    # improved amplitude guess if frequency is provided
    if 'omega' in p_dict.constant_parameter_names or \
        'omega' in p_dict.initialised_parameter_names:
        p_dict['a'] = np.sqrt(np.max(pgram) * 4 / len(y))
    else:
        temp = pgram[np.searchsorted(omega_list, p_dict['omega'])]
        p_dict['a'] = np.sqrt(temp * 4 / len(y))


    # having estimated 'omega' average over one period
    t_period = 2 * np.pi / p_dict['omega']

    p_dict['c_equ'] = np.mean(y[np.argwhere(t > t[-1] - 2*t_period)])
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


attenuated_sinusoid = FitBase(
    ["omega", "t_dead", "a", "c_offset", "c_equ", "rate", "phi"],
    fitting_function=fitting_function, parameter_initialiser=init_all,
    # derived_parameter_function=derived_params,
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
    offset = -0.5
    equ = 1.0
    rate = 1e2
    phi = np.pi/2

    n_init = 25
    t_init = 0.8*2*np.pi/omega

    n_end = 35
    t_delay = 1 / rate
    t_end = 2.8 * 2 * np.pi / omega


    temp0 = t_init * np.random.rand(n_init)
    temp1 = t_delay + t_end * np.random.rand(n_end)
    t = np.concatenate((temp0, temp1))
    y = np.sin(t * omega + phi) + rel_noise * np.random.rand(len(t))
    y *= amp
    y += offset
    y *= np.exp(-rate*t)
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
    if True:  # frequency initial guess introspection (fit debugging)
        mask = np.argsort(t)
        t = t[mask]
        y = y[mask]

        min_step = np.min(t[1:] - t[:-1])
        duration = t[-1] - t[0]
        f_max = 0.5 / min_step
        f_min = 0.5 / duration

        omega_list = 2 * np.pi * np.linspace(f_min, f_max, int(f_max / f_min))
        pgram = lombscargle(t, y, omega_list, precenter=True)
        plt.figure()
        plt.plot(omega_list, pgram)
        plt.xlim(0, omega*5)
        plt.show()

