import numpy as np
from statsmodels.stats.proportion import proportion_confint


def binom_twosided(k, N):
    """Returns the estimated source probability and confidence interval from
    a sample of k Trues out of N samples.
    """
    if k > N or k < 0:
        raise ValueError("k must be between 0 and N (k={}, N={})".format(k, N))

    # 'beta' is Clopper–Pearson method; chosen alpha corresponds to 1σ of a normal
    # distribution (68%).
    confint = proportion_confint(k, N, alpha=0.3173, method='beta')

    # Strip out NaNs for confidence intervals at the boundary
    if np.isnan(confint[0]):
        confint = (0, confint[1])
    elif np.isnan(confint[1]):
        confint = (confint[0], 1)

    p = k / N
    return p, confint


def binom_onesided(k, N):
    """Returns the estimated source probability and uncertainty from a sample
    of k Trues out of N samples.
    """
    p, confint = binom_twosided(k, N)
    # This is not great at the boundary (k=0 or k=N).
    # TODO: numerically test this, perhaps add a correction term
    # (Laplace law of succession?)
    uncertainty = (confint[1] - confint[0]) / 2

    return p, uncertainty
