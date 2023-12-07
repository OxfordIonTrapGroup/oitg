"""Fit a sinusoid with optional dead-time to data"""
import numpy as np
from scipy.signal import lombscargle
from .FitBase import FitBase


def fitting_function(t, p_dict):
    """
    Sinusoid with dead time

    This allows fitting of a function that is constant for t <= t_dead and
    sinusoidal for t > t_dead, with a phase such that the transition at
    t=t_dead is continuous.
    A discontinuous transition from a constant level to a sinusoid is not
    supported at the moment.

    If the dead time is non-zero, the parameter "phi" refers to the phase
    offset using a reference time of t=t_dead. This means that, for example,
    if t_dead is non-zero and phi=0.0, the result is a sine curve that goes
    through zero at t=t_dead, but would not necessarily do so at t=0
    (extrapolating back through dead time).

    If t_dead is known to be zero, best fit results are obtained if this
    parameter is constrained manually when fitting.
    """
    y = p_dict["a"] * np.sin((t - p_dict["t_dead"]) * p_dict["omega"]
                             + p_dict["phi"]) \
        + p_dict["c"]
    # Hold constant during dead time
    return np.where(t > p_dict["t_dead"], y,
                    p_dict["a"] * np.sin(p_dict["phi"]) + p_dict["c"])


def parameter_initialiser(t, y, p_dict):
    """
    Initialise parameters under the following assumptions:
    - t, y are 1D arrays of real numbers
    - associated data coordinates share the same index in t and y

    Note: Data may be in any order and arbitrarily spaced
    """
    # Sort data by time (x-axis)
    mask = np.argsort(t)
    t = t[mask]
    y = y[mask]

    min_step = np.min(t[1:] - t[:-1])
    duration = t[-1] - t[0]
    # Nyquist limit does not apply to irregularly spaced data
    # We'll use it as a starting point anyway...
    f_max = 0.5 / min_step
    # relaxed Fourier limit
    f_min = 0.25 / duration

    omega_list = 2 * np.pi * np.linspace(f_min, f_max, int(f_max / f_min))
    pgram = lombscargle(t, y, omega_list, precenter=True)

    p_dict["omega"] = omega_list[np.argmax(pgram)]
    p_dict["c"] = np.mean(y)
    p_dict["a"] = np.sqrt(np.max(pgram) * 4 / len(y))
    p_dict["t_dead"] = np.min(t)

    y0_norm = (y[0] - p_dict["c"]) / p_dict["a"]
    if y0_norm <= -1.0:
        p_dict["phi"] = -np.pi / 2
    elif y0_norm >= 1.0:
        p_dict["phi"] = np.pi / 2
    else:
        p_dict["phi"] = np.arcsin(y0_norm)
    return p_dict


def derived_params(p_dict, p_error_dict):
    """
    Calculate commonly used derived parameters and their error

    This method neglects parameter covariances!
    """
    p_dict["max"] = p_dict["c"] + np.abs(p_dict["a"])
    p_dict["min"] = p_dict["c"] - np.abs(p_dict["a"])
    p_dict["t_pi"] = p_dict["t_dead"] + np.pi / p_dict["omega"]
    p_dict["t_pi/2"] = p_dict["t_dead"] + np.pi / 2 / p_dict["omega"]
    p_dict["period"] = 2 * np.pi / p_dict["omega"]

    # this error calculation neglects parameter covariance!
    # may want to upgrade FitBase to make covariance matrix available.

    p_error_dict["max"] = \
        np.sqrt(p_error_dict["c"]**2
                + p_error_dict["a"]**2)

    p_error_dict["min"] = \
        np.sqrt(p_error_dict["c"]**2
                + p_error_dict["a"]**2)

    p_error_dict["t_pi"] = \
        np.sqrt(p_error_dict["t_dead"]**2
                + (np.pi / p_dict["omega"]
                    * (p_error_dict["omega"] / p_dict["omega"]))**2)

    p_error_dict["t_pi/2"] = \
        np.sqrt(p_error_dict["t_dead"]**2
                + (np.pi / 2 / p_dict["omega"]
                    * (p_error_dict["omega"] / p_dict["omega"]))**2)

    p_error_dict["period"] = \
        2 * np.pi / p_dict["omega"] * (p_error_dict["omega"] / p_dict["omega"])

    return (p_dict, p_error_dict)


sinusoid = FitBase(
    ["omega", "t_dead", "a", "c", "phi"],
    fitting_function=fitting_function,
    parameter_initialiser=parameter_initialiser,
    derived_parameter_function=derived_params,
    parameter_bounds={
        "omega": (0, np.inf),
        "t_dead": (0, np.inf),
        "a": (0, np.inf),
        "c": (-np.inf, np.inf),
        "phi": (-np.inf, np.inf),  # allows fit to wrap phase
    })

if __name__ == "__main__":
    # example and debugging
    n_sample = 30
    t_max = 1e-4
    amp = 0.1
    rel_noise = 0.5
    offset = 40
    phi = np.pi

    t = t_max * np.random.rand(n_sample)
    y = np.sin(t * 2.3e5 + phi) + rel_noise * np.random.rand(n_sample)
    y *= amp
    y += offset

    # fix these fit parameters to a specific value
    const_dict = {
        't_dead': 0.0,
        # 'phi': phi,
        # 'c': offset
    }
    p, p_err, x_fit, y_fit = sinusoid.fit(t,
                                          y,
                                          y_err=np.ones(y.shape) *
                                          np.sqrt(1 / 3 - 1 / 4) * amp * rel_noise,
                                          evaluate_function=True,
                                          evaluate_x_limit=[0, t_max],
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
