import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import lombscargle
from . import FitBase
"""Fit a typical Rabi flop time scan with a decaying cosine curve, including
initial dead time to account for AOM/... switching effects.

Currently only supports positive-sign scans (starting at y=1, not y=0).

For guessing the initial parameters, it is assumed that the scan covers at
least about a quarter of an oscillation period, and that the sampling is dense
enough to resolve the oscillation (more than two points per period).
"""


def parameter_initialiser(x, y, p):
    # Sort by time so sample spacings and early/late chunks are meaningful.
    order = np.argsort(x)
    x = np.asarray(x)[order]
    y = np.asarray(y)[order]

    t_min = x[0]
    t_range = x[-1] - x[0]
    steps = np.diff(x)
    steps = steps[steps > 0]
    if t_range <= 0.0 or len(steps) == 0:
        # Degenerate scan; avoid divisions by zero below (the fit cannot work
        # on such data anyway).
        t_range = 1.0
        steps = np.array([1.0])
    y_mean = np.mean(y)

    # Estimate the decay time constant by comparing the RMS deviations from
    # the global mean (~oscillation amplitudes) in the first and last thirds
    # of the scan. Clamp the result to a sane range to keep the least-squares
    # problem well-conditioned even for scans without visible decay.
    k = max(len(x) // 3, 1)
    a_early = np.sqrt(np.mean((y[:k] - y_mean)**2))
    a_late = np.sqrt(np.mean((y[-k:] - y_mean)**2))
    t_centre_diff = np.mean(x[-k:]) - np.mean(x[:k])
    if a_early > a_late > 0.0 and t_centre_diff > 0.0:
        tau_decay = t_centre_diff / np.log(a_early / a_late)
    else:
        tau_decay = np.inf
    p["tau_decay"] = np.clip(tau_decay, t_range / 10, 10 * t_range)

    # Estimate frequency using a Lomb-Scargle periodogram (which supports
    # irregularly-spaced samples). Search between a quarter of an oscillation
    # over the whole scan (relaxed Fourier limit, to also handle scans that
    # do not quite reach the first minimum) and the Nyquist frequency
    # corresponding to the median sample spacing, with the grid chosen fine
    # enough to resolve the position of periodogram peaks (of width
    # ~2 pi / t_range) well.
    omega_min = 0.5 * np.pi / t_range
    omega_max = max(np.pi / np.median(steps), 4 * omega_min)
    num_omegas = int(np.ceil((omega_max - omega_min) / (np.pi / (4 * t_range))))
    omegas = np.linspace(omega_min, omega_max, max(num_omegas, 32))
    # Subtract the mean ourselves, as lombscargle(precenter=True) modifies y
    # in place on SciPy 1.15+, corrupting the data used for the actual fit.
    # To keep the strong low-frequency components introduced by the decaying
    # envelope from overshadowing the oscillation, also divide out the decay
    # estimated above (clamping the envelope to avoid blowing up the noise in
    # the tail of strongly damped scans).
    envelope = np.maximum(np.exp(-(x - t_min) / p["tau_decay"]), np.exp(-2.0))
    z = (y - y_mean) / envelope
    pgram = lombscargle(x, z - np.mean(z), omegas)

    p["t_dead"] = 0.0

    p["y_lower"] = np.clip(2 * y_mean - 1, 0, 1)

    # Consider the strongest few periodogram peaks and their first harmonics
    # (as noise and irregular sampling can cause a subharmonic to end up
    # stronger than the true frequency) as candidates, preferring those which
    # lead to a pi time larger than t_min (i.e. the first minimum not before
    # the scanned range). Among those, pick the one that actually matches the
    # data best when combined with the other initial parameter estimates.
    peak_idxs = np.nonzero((pgram[1:-1] >= pgram[:-2])
                           & (pgram[1:-1] >= pgram[2:]))[0] + 1
    if len(peak_idxs) == 0:
        peak_idxs = np.array([np.argmax(pgram)])
    peaks = peak_idxs[np.argsort(-pgram[peak_idxs])][:3]
    candidates = [c for omega in omegas[peaks] for c in (omega, 2 * omega)]
    allowed = [omega for omega in candidates if np.pi / omega > t_min]
    if allowed:
        candidates = allowed

    trial = {"t_dead": 0.0, "y_lower": p["y_lower"], "tau_decay": p["tau_decay"]}

    def sum_squares(omega):
        trial["t_period"] = 2 * np.pi / omega
        return np.sum((y - fitting_function(x, trial))**2)

    p["t_period"] = 2 * np.pi / min(candidates, key=sum_squares)


def fitting_function(x, p):
    y_upper = 1.0
    shifted_t = (x - p["t_dead"])
    y = p["y_lower"] + (y_upper - p["y_lower"]) / 2 * (
        np.exp(-shifted_t / p["tau_decay"]) *
        np.cos(2 * np.pi / p["t_period"] * shifted_t) + 1)
    return np.where(x < p["t_dead"], y_upper, y)


def derived_parameter_function(p, p_err):
    non_decaying_pi_time = p["t_dead"] + p["t_period"] / 2

    # Compute the point of maximum population transfer (minimum in y) which
    # will be slightly shifted towards zero in the face of non-zero tau_decay.
    fit = minimize_scalar(lambda t: fitting_function(t, p),
                          method="brent",
                          bracket=[0.9 * non_decaying_pi_time, non_decaying_pi_time])
    if fit.success:
        p["t_pi"] = fit.x
    else:
        p["t_pi"] = non_decaying_pi_time

    # This is just a Gaussian error propagation guess.
    p_err["t_pi"] = np.sqrt(p_err["t_dead"]**2 + (p_err["t_period"] / 2)**2)
    return p, p_err


rabi_flop = FitBase.FitBase(
    ["t_period", "t_dead", "y_lower", "tau_decay"],
    fitting_function,
    parameter_initialiser=parameter_initialiser,
    derived_parameter_function=derived_parameter_function,
    parameter_bounds={
        "t_period": (0, np.inf),
        "t_dead": (0, np.inf),
        "y_lower": (0, 1),
        "tau_decay": (0, np.inf)
    },
    derived_parameter_names=["t_pi"],)
