import numpy as np
from scipy.signal import lombscargle
from . import FitBase

"""Fit a typical Rabi flop time scan with a decaying cosine curve, including
initial dead time to account for AOM/... switching effects.

Currently only supports positive-sign scans (starting at y=1, not y=0).
"""

def parameter_initialiser(x, y, p):
    x_range = np.amax(x) - np.amin(x)
    if x_range == 0.0:
        x_range = 1.0

    # Estimate frequency (using Long-Scargle periodogram, to support irregularly
    # spaced x points).
    # TODO: Could use better heuristics for frequency range based on minimum
    # distance between points -> aliasing.
    freq = 2 * np.pi / x_range
    freqs = np.linspace(0.1 * freq, 10 * freq, 2 * len(x))
    pgram = lombscargle(x, y, freqs, precenter=True)
    p["t_period"] = 2 * np.pi / freqs[np.argmax(pgram)]

    p["t_dead"] = 0.0

    p["y_lower"] = 2 * np.mean(y) - 1

    # TODO: Estimate decay time constant using RMS amplitude from global mean
    # in first and last chunk.
    p["tau_decay"] = 10e-6

def fitting_function(x, p):
    y_upper = 1.0
    shifted_t = (x - p["t_dead"])
    y = p["y_lower"] + (y_upper - p["y_lower"]) / 2 * (np.exp(-shifted_t / p["tau_decay"]) * np.cos(2 * np.pi / p["t_period"] * shifted_t) + 1)
    return np.where(x < p["t_dead"], y_upper, y)

def derived_parameter_function(p, p_err):
    p["t_pi"] = p["t_dead"] + p["t_period"] / 2
    p_err["t_pi"] = np.sqrt(p_err["t_dead"]**2 + (p_err["t_period"] / 2)**2)
    return p, p_err

rabi_flop = FitBase.FitBase(
    ["t_period", "t_dead", "y_lower", "tau_decay"],
    fitting_function, parameter_initialiser=parameter_initialiser,
    derived_parameter_function=derived_parameter_function)
