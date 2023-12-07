"""Fit a decaying sinusoid with optional dead-time to data"""
import numpy as np
from scipy.signal import lombscargle
from .FitBase import FitBase


def fitting_function(t, p_dict):
    """decaying sinusoid with dead time
    p_dict fields:
        rate: decay rate of sinusoid (1/tau)
        a: initial sinusoid amplitude
        t_dead: represents the time before which the signal is held constant
        omega: angular oscillation frequency
        phi: initial phase at first data-point
        c_offset: initial oscillation offset from equilibrium constant
        c_equ: equilibrium value the oscillation decays to after a long time
    """
    y = np.exp(-p_dict['rate'] * (t - p_dict['t_dead'])) * \
        (p_dict['a'] * np.sin((t - p_dict['t_dead']) * p_dict['omega']
                              + p_dict['phi'])
         + p_dict['c_offset']) + p_dict['c_equ']
    # hold constant during dead time
    return np.where(
        t > p_dict['t_dead'], y,
        p_dict['c_offset'] + p_dict['c_equ'] + p_dict['a'] * np.sin(p_dict['phi']))


def init_all(t, y, p_dict):
    """Initialise parameters under the general assumptions that:
    t & y are 1D arrays of real numbers
    associated data coordinates share the same index in t & y

    Note: Data may be in any order and arbitrarily spaced"""
    # p_dict is an instance of FitParameters
    # user-initialised values are not overwritten
    # non user-initialised parameters are pre-initialised to 0

    # sort data by time (x-axis)
    mask = np.argsort(t)
    t = t[mask]
    y = y[mask]

    # Estimate frequency. Starting with a Lomb-Scargle periodogram (which
    # supports irregularly-spaced samples), we pick the strongest frequency
    # component which leads to a pi time larger than the smallest sample spacing
    # and smaller than the data time range. (Nyquist and Fourier limit)
    min_step = np.min(t[1:] - t[:-1])
    duration = t[-1] - t[0]
    # Nyquist limit does not apply to irregularly spaced data
    # We'll use it as a starting point anyway...
    f_max = 0.5 / min_step
    # relaxed Fourier limit
    f_min = 0.25 / duration
    omega_list = 2 * np.pi * np.linspace(f_min, f_max, int(f_max / f_min))

    # Lomb-Scargle does not deal well with exponential decays where c_offset > a
    # if an initial c_equ and decay rate are provided, we can remove the
    # exponential decay for estimating the frequency.
    if ('c_equ' in p_dict.constant_parameter_names
            or 'c_equ' in p_dict.initialised_parameter_names):
        temp = (y - p_dict['c_equ']) * np.exp(p_dict['rate'] * t)
        pgram = lombscargle(t, temp, omega_list, precenter=True)
    else:
        pgram = lombscargle(t, y, omega_list, precenter=True)
    p_dict['omega'] = omega_list[np.argmax(pgram)]

    # this amplitude guess is aware of user initialised omega
    temp = pgram[np.searchsorted(omega_list, p_dict['omega'])]
    p_dict['a'] = np.sqrt(temp * 4 / len(y))

    # having estimated 'omega', average over a period
    t_period = 2 * np.pi / p_dict['omega']
    p_dict['c_equ'] = np.mean(y[np.argwhere(t > t[-1] - 2 * t_period)])
    p_dict['c_offset'] = np.mean(y[np.argwhere(t < t[0] + t_period)]) - \
        p_dict['c_equ']

    p_dict['t_dead'] = 0.0
    p_dict['rate'] = 0.0  # TODO: be smarter about the decay rate guess

    # estimate phase from initial value
    # this tends to converge even is inaccurately initialised
    y0_norm = (y[0] - p_dict['c_equ'] - p_dict['c_offset']) / p_dict['a']
    if y0_norm <= -1.0:
        p_dict['phi'] = -np.pi / 2
    elif y0_norm >= 1.0:
        p_dict['phi'] = np.pi / 2
    else:
        p_dict['phi'] = np.arcsin(y0_norm)
    return p_dict  # input is changed due to mutability!


def derived_params(p_dict, p_error_dict):
    """calculate commonly used derived parameters and their error

    This method neglects parameter covariances!"""
    # the decaying maximum transfer time is slightly shifted from the sinusoid
    # pi-time! Additionally, starting with a phase offset effects the minimum

    # time of maximum population transfer
    # t = t_dead + 1/omega*(n*pi-phi-arcsin(1/sqrt(1+(rate/omega)**2))
    #                       + arcsin(c_offset * rate /
    #                                (a*omega * sqrt(1+(rate/omega)**2)))
    #                       )
    # NB: this representation avoids div 0 errors for rate=0
    #
    # replace n*pi with pi/2 + (corrections + pi/2)%pi
    # This gives the pi pulse equivalent time
    p_dict['t_max_transfer'] = (
        p_dict['t_dead'] + 1 / p_dict['omega'] *
        (np.pi / 2 +
         (-p_dict['phi'] - np.arcsin(1 / np.sqrt(1 +
                                                 (p_dict['rate'] / p_dict['omega'])**2))
          + np.arcsin(p_dict['c_offset'] * p_dict['rate'] /
                      (p_dict['a'] * p_dict['omega'] * np.sqrt(1 + (
                          p_dict['rate'] / p_dict['omega'])**2))) + np.pi / 2) % np.pi))
    p_dict['period'] = 2 * np.pi / p_dict['omega']
    p_dict['tau_decay'] = 1 / p_dict['rate']

    # this error calculation neglects parameter covariance!
    # may want to upgrade FitBase to make covariance matrix available.

    # calculate error for t_max_transfer
    # df = sqrt(sum_i( df/dx_i * dx_i))
    dtdt_dead = 1
    dtdphi = -1 / p_dict['omega']

    # derivative of arcsin wrt it's argument
    def darcsindx(x):
        return 1 / np.sqrt(1 - x * x)

    # common term
    temp0 = 1 / np.sqrt(1 + (p_dict['rate'] / p_dict['omega'])**2)
    # derivative of second arcsin argument wrt c_offset
    temp1 = p_dict['rate'] / (p_dict['a'] * p_dict['omega']) * temp0
    # second arcsin argument
    temp2 = temp1 * p_dict['c_offset']
    dtdc_offset = 1 / p_dict['omega'] * temp1 * darcsindx(temp2)

    # derivative of second arcsin argument wrt a
    temp3 = -temp2 / p_dict['a']
    dtda = 1 / p_dict['omega'] * temp3 * darcsindx(temp2)

    # 1/omega * derivative of second arcsin argument wrt rate
    temp4 = p_dict['c_offset'] * temp0 / (p_dict['a'] *
                                          (p_dict['omega']**2 + p_dict['rate']**2))
    dtdrate = (1 / p_dict['omega'] * p_dict['rate'] / p_dict['omega']**2 * temp0**3 *
               darcsindx(temp0) + temp4 * darcsindx(temp2))

    # derivative of first arcsin argument wrt omega
    temp5 = p_dict['rate']**2 * (temp0 / p_dict['omega'])**3
    # derivative of second arcsin argument wrt omega
    temp6 = (p_dict['c_offset'] * p_dict['rate'] * temp0 /
             (p_dict['a'] * p_dict['omega']**2) *
             ((p_dict['rate'] * temp0 / p_dict['omega'])**2 - 1))
    dtdomega = 1 / p_dict['omega'] * (-temp5 * darcsindx(temp0) +
                                      temp6 * darcsindx(temp2))

    dtdc_equ = 0.0

    covar_mat = np.diag([
        p_error_dict['t_dead']**2,
        p_error_dict['phi']**2,
        p_error_dict['c_offset']**2,
        p_error_dict['a']**2,
        p_error_dict['rate']**2,
        p_error_dict['omega']**2,
        p_error_dict['c_equ']**2,
    ])
    deriv_vect = np.array(
        [dtdt_dead, dtdphi, dtdc_offset, dtda, dtdrate, dtdomega, dtdc_equ])

    p_error_dict['t_max_transfer'] = np.sqrt(
        np.einsum('i,ij,j', deriv_vect, covar_mat, deriv_vect))
    p_error_dict['period'] = p_dict['period'] * \
        (p_error_dict['omega'] / p_dict['omega'])
    p_error_dict['tau_decay'] = p_dict['tau_decay'] * \
        (p_error_dict['rate'] / p_dict['rate'])

    return (p_dict, p_error_dict)


decaying_sinusoid = FitBase(
    ["omega", "t_dead", "a", "c_offset", "c_equ", "rate", "phi"],
    fitting_function=fitting_function,
    parameter_initialiser=init_all,
    derived_parameter_function=derived_params,
    parameter_bounds={
        "omega": (0, np.inf),
        "t_dead": (0, np.inf),
        "a": (0, np.inf),
        "c_offset": (-np.inf, np.inf),
        "c_equ": (-np.inf, np.inf),
        "rate": (0, np.inf),
        "phi": (-np.inf, np.inf),  # allows fit to wrap phase
    })

if __name__ == "__main__":
    # example and debugging
    omega = 3.05e4
    amp = 0.5
    rel_noise = 1e-1
    # fit does not work with significant offset as initial frequency guess breaks
    offset = 0.0
    equ = 0.5
    rate = 1e2
    phi = np.pi / 2

    n_init = 65
    t_init = 10.4 * 2 * np.pi / omega

    n_end = 65
    t_delay = 1.9 / rate
    t_end = 10.2 * 2 * np.pi / omega

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
        'c_offset': offset,
        'c_equ': equ,
        # 'phi': phi,
    }
    init_dict = {
        # 't_dead': 0.0,
        'phi': phi,
        # 'rate': rate,
    }
    p, p_err, x_fit, y_fit = decaying_sinusoid.fit(
        t,
        y,
        y_err=np.ones(y.shape) * np.sqrt(1 / 3 - 1 / 4) * amp * rel_noise,
        initialise=init_dict,
        evaluate_function=True,
        evaluate_x_limit=[0, t_end + t_delay],
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
        plt.errorbar(t,
                     y,
                     yerr=np.ones(y.shape) * np.sqrt(1 / 3 - 1 / 4) * amp * rel_noise,
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
