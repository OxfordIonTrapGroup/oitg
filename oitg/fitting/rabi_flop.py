"""Fit a typical Rabi flop time scan with a decaying cosine curve, including
initial dead time to account for AOM/... switching effects.

The flop starts at y_start and oscillates towards y_lower, with the contrast decaying
exponentially in time (i.e. y approaches the mean of the two levels for t >> tau_decay).
Both scans starting at y = 1 and at y = 0 are supported; unless explicitly specified by
the user (either as a constant or initial value), y_start is inferred from the data and
held constant at 0 or 1 during the fit. To fix the direction ahead of time, pass e.g.
constants={"y_start": 0.0}; to instead let the fit also refine the initial level (e.g.
to absorb state preparation errors), pass it as an initial value.

Note: For backwards compatibility, the level the flop oscillates towards is always
called y_lower, even though it is the *upper* level for flops starting at y = 0.
Parameter sets without a y_start entry (e.g. from before flops starting at y = 0 were
supported) are interpreted as starting at y = 1.

Scans which do not start (close to) one of the extrema are not covered by this model;
see :mod:`.sinusoid`/:mod:`.decaying_sinusoid`.

For guessing the initial parameters, it is assumed that the scan covers at least about a
quarter of an oscillation period, and that the sampling is dense enough to resolve the
oscillation (more than two points per period).
"""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import lombscargle

from . import FitBase


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
    a_early = np.sqrt(np.mean((y[:k] - y_mean) ** 2))
    a_late = np.sqrt(np.mean((y[-k:] - y_mean) ** 2))
    t_centre_diff = np.mean(x[-k:]) - np.mean(x[:k])
    if a_early > a_late > 0.0 and t_centre_diff > 0.0:
        tau_decay = t_centre_diff / np.log(a_early / a_late)
    else:
        tau_decay = np.inf
    p["tau_decay"] = np.clip(tau_decay, t_range / 10, 10 * t_range)

    # Estimate frequency using a Lomb-Scargle periodogram (which supports irregularly-
    # spaced samples). Search between a quarter of an oscillation over the whole scan
    # (to also handle scans that do not quite reach the first minimum) and the Nyquist
    # frequency corresponding to a typical small sample spacing (see below), with the
    # grid chosen fine enough to resolve the position of periodogram peaks (of width
    # ~2 pi / t_range) well.
    #
    # We use the first decile of the spacings rather than e.g. the median, as the
    # spacings can be very non-uniform e.g. during acquisition of randomly ordered
    # scans, or for geometrically spaced scans. The first decile rather the minimum, as
    # well as the clamp to a fraction of the mean, is a heuristic to avoid making the
    # evaluation too expensive, and can be tuned further or dropped.
    omega_min = 0.5 * np.pi / t_range
    step_ref = max(np.quantile(steps, 0.1), np.mean(steps) / 4)
    omega_max = max(np.pi / step_ref, 4 * omega_min)
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

    # Consider the strongest few periodogram peaks and their first harmonics (as noise
    # and irregular sampling can cause a subharmonic to end up stronger than the true
    # frequency) as candidates, preferring those which lead to a pi time larger than
    # t_min (i.e. the first extremum not before the scanned range). Among those, pick
    # the one that actually matches the data best when combined with the other initial
    # parameter estimates.
    #
    # We don't consider frequencies outside the search range and prefer lower
    # frequencies to avoid aliasing of harmonics.
    # Pad with -inf so that maxima at the edges of the search range are also picked up.
    # In particular, for flops slower than the scan, the periodogram just increases
    # monotonically towards omega_min, so there would otherwise be no candidate
    # anywhere near the true frequency.
    padded = np.concatenate(([-np.inf], pgram, [-np.inf]))
    peak_idxs = np.nonzero(
        (padded[1:-1] >= padded[:-2]) & (padded[1:-1] >= padded[2:])
    )[0]
    if len(peak_idxs) == 0:
        peak_idxs = np.array([np.argmax(pgram)])
    peaks = peak_idxs[np.argsort(-pgram[peak_idxs])][:3]
    candidates = list(omegas[peaks])
    candidates += [2 * omega for omega in omegas[peaks] if 2 * omega <= omega_max]
    allowed = [omega for omega in candidates if np.pi / omega > t_min]
    if allowed:
        candidates = allowed
    candidates.sort()

    # Unless the user has fixed the starting level (and thus the direction of the flop),
    # also let both a flop starting at y = 0 and one starting at y = 1 compete based on
    # the rms error for the parameter estimates (approximating y_lower as the opposite
    # excursion from the data mean relative to y_start).
    if p.is_initialised("y_start"):
        y_starts = [p["y_start"]]
    else:
        y_starts = [0.0, 1.0]

    def trial_parameters(y_start, omega):
        if p.is_initialised("y_lower"):
            y_lower = p["y_lower"]
        else:
            y_lower = np.clip(2 * y_mean - y_start, 0, 1)
        return {
            "t_period": 2 * np.pi / omega,
            "t_dead": 0.0,
            "y_start": y_start,
            "y_lower": y_lower,
            "tau_decay": p["tau_decay"],
        }

    def sum_squares(trial):
        return np.sum((y - fitting_function(x, trial)) ** 2)

    best = min(
        (trial_parameters(s, omega) for s in y_starts for omega in candidates),
        key=sum_squares,
    )
    p.hold_constant("y_start", best["y_start"])
    p["y_lower"] = best["y_lower"]
    p["t_period"] = best["t_period"]


def transferred_fraction(t, p):
    """Just the exponentially damped population transfer probability (starting at 0),
    without dead time, starting sign, or contrast reduction.
    """
    return (1 - np.exp(-t / p["tau_decay"]) * np.cos(2 * np.pi / p["t_period"] * t)) / 2


def fitting_function(x, p):
    # Parameter sets from before flops starting at y = 0 were supported.
    try:
        y_start = p["y_start"]
    except KeyError:
        y_start = 1.0

    shifted_t = x - p["t_dead"]
    y = y_start + (p["y_lower"] - y_start) * transferred_fraction(shifted_t, p)
    return np.where(x < p["t_dead"], y_start, y)


def derived_parameter_function(p, p_err):
    half_period = p["t_period"] / 2

    # Compute the point of maximum population transfer (the first extremum in
    # y), which will be slightly shifted towards zero in the face of non-zero
    # tau_decay.
    fit = minimize_scalar(
        lambda t: -transferred_fraction(t, p),
        method="brent",
        bracket=[0.9 * half_period, half_period],
    )
    if fit.success:
        p["t_pi"] = p["t_dead"] + fit.x
    else:
        p["t_pi"] = p["t_dead"] + half_period

    # This is just a Gaussian error propagation guess.
    p_err["t_pi"] = np.sqrt(p_err["t_dead"] ** 2 + (p_err["t_period"] / 2) ** 2)
    return p, p_err


rabi_flop = FitBase.FitBase(
    ["t_period", "t_dead", "y_start", "y_lower", "tau_decay"],
    fitting_function,
    parameter_initialiser=parameter_initialiser,
    derived_parameter_function=derived_parameter_function,
    parameter_bounds={
        "t_period": (0, np.inf),
        "t_dead": (0, np.inf),
        "y_start": (0, 1),
        "y_lower": (0, 1),
        "tau_decay": (0, np.inf),
    },
    derived_parameter_names=["t_pi"],
)
