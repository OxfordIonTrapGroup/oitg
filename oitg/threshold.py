import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson
from scipy.special import gammainc


def optimise_readout(bright_rate, dark_rate, dark_to_bright_rate=(1 / 1.168),
                     p_bright=0.5):
    """Calculate optimal threshold bin time & threshold count

    The calculation assumes both bright and dark counts are
    Poisson distributed and gives a threshold minimising the error probability

    The calculation accounts for de-shelving during the readout bin time.
    See thesis: Alice Burrell, 2010

    :param bright_rate: expected bright count rate in $s^-1$
    :param dark_rate: expected dark count ratein $s^-1$
    :param dark_to_bright_rate: dark state decay to become bright in $s^-1$.
        As this function seeks the global minimum error, the rate must not be
        zero. Default of 1/1.168 is the Calcium D5/2 shelf decay rate.
    :param p_bright: probability of encountering a bright state (default=0.5)

    :returns: (target_t_bin [s], threshold_count, p_error)"""
    thresh = 1
    error_last = 1.
    t_bin, error = optimise_t_bin(
        bright_rate, dark_rate, thresh, dark_to_bright_rate, p_bright)
    while error_last > error:
        error_last = error
        t_bin_last = t_bin
        thresh += 1
        t_bin, error = optimise_t_bin(
            bright_rate, dark_rate, thresh, dark_to_bright_rate, p_bright)

    return (t_bin_last, thresh - 1, error_last)


def optimise_treshold(bright_rate, dark_rate, t_bin, dark_to_bright_rate=0.,
                      p_bright=0.5):
    """Calculate optimal threshold threshold count for a given bin time

    The calculation accounts for de-shelving during the readout bin time.
    See thesis: Alice Burrell, 2010

    :param bright_rate: expected bright count rate in $s^-1$
    :param dark_rate: expected dark count ratein $s^-1$
    :param dark_to_bright_rate: dark state decay to become bright in $s^-1$
    :param p_bright: probability of encountering a bright state (default=0.5)

    :returns: (target_t_bin [s], threshold_count, p_error)"""
    # de-shelving increases the count rate of dark states (non-poissonian)
    # The no-de-shelving threshold is therefore a lower bound
    thresh_min = poisson_optimal_thresh_count(
        bright_rate * t_bin, dark_rate * t_bin, p_bright)
    thresh_max = bright_rate * t_bin

    # thresholds are discrete - could implement discreet optimisation, but the
    # range of values is small in practice. Therefore we can try all options
    thresh_vec = np.arange(thresh_min, thresh_max + 1, dtype=np.int_)
    err_vec = np.array([calc_p_error(bright_rate, dark_rate, t_bin,
                                     i, dark_to_bright_rate,
                                     p_bright=p_bright)
                        for i in thresh_vec])

    min_idx = np.argmin(err_vec)
    return thresh_vec[min_idx], err_vec[min_idx]


def optimise_t_bin(bright_rate, dark_rate, thresh_count, dark_to_bright_rate=0.,
                   p_bright=0.5):
    """Calculate optimal threshold bin time for a given threshold count

    The calculation accounts for de-shelving during the readout bin time.
    See thesis: Alice Burrell, 2010

    :param bright_rate: expected bright count rate in $s^-1$
    :param dark_rate: expected dark count ratein $s^-1$
    :param dark_to_bright_rate: dark state decay to become bright in $s^-1$
    :param p_bright: probability of encountering a bright state (default=0.5)

    :returns: (target_t_bin [s], threshold_count, p_error)"""
    t_bin_init = thresh_count * (0.5 / dark_rate + 0.5 / bright_rate)
    t_scale = 1e-4
    err_scale = 1e-6

    def p_error(x):
        return calc_p_error(bright_rate, dark_rate, x[0] * t_scale,
                            thresh_count, dark_to_bright_rate,
                            p_bright=p_bright) / err_scale

    result = minimize(p_error,
                      np.array([t_bin_init / t_scale]),
                      bounds=((1 / bright_rate / t_scale, np.inf),))

    return result.x[0] * t_scale, result.fun * err_scale


def calc_p_error(bright_rate, dark_rate, t_bin, thresh_count,
                 dark_to_bright_rate=0., p_bright=0.5):
    """Calculate error probability for Poisson statistics with de-shelving

    See thesis: Alice Burrell, 2010

    :param bright_rate: expected bright count rate in $s^-1$
    :param dark_rate: expected dark count rate in $s^-1$
    :param t_bin: integration time in s.
    :param thresh_count: threshold count for discriminating bright/dark state
        (Assumes the exact threshold count is evaluated as dark)
    :param dark_to_bright_rate: dark state decay to become bright in $s^-1$
    :param p_bright: probability of encountering a bright state (default=0.5)
    """
    def p_n_given_dark(n):
        "Burrell Eqns 3.2 & 3.6"
        if dark_to_bright_rate == 0.:
            return poisson.pmf(n, mu=dark_rate * t_bin)

        rb_tau = (bright_rate - dark_rate) / dark_to_bright_rate
        eps = bright_rate * t_bin / rb_tau

        x_n = np.exp(-eps) * np.power(rb_tau / (rb_tau - 1), n) / (rb_tau - 1)
        x_n *= (gammainc(n + 1, eps * (rb_tau - 1))
                - gammainc(n + 1, dark_rate * t_bin / rb_tau * (rb_tau - 1)))

        d_n = poisson.pmf(n, mu=dark_rate * t_bin) * np.exp(
            -t_bin * dark_to_bright_rate) + x_n
        return d_n

    # up-to & including thresh_count
    n_vec = np.arange(np.round(thresh_count) + 1, dtype=np.int_)

    p_error = (1 - p_bright) * (1 - np.sum(p_n_given_dark(n_vec[:-1])))
    p_error += p_bright * np.sum(poisson.pmf(n_vec, mu=bright_rate * t_bin))
    return p_error


def poisson_optimal_thresh_count(mean_bright, mean_dark, p_bright=0.5):
    """Optimal threshold rate in the absence of de-shelving

    The calculation assumes both bright and dark counts are Poisson distributed
    and gives a threshold minimising the error probability.

    The calculation neglects de-shelving and accidental shelving during the
    readout bin time. It is therefore not suitable for P(error) < 2e-4.
    See thesis: Alice Burrell, 2010

    :param mean_bright: expected counts if the ion started in a bright state
    :param mean_dark: expected counts if the ion started in a dark state
    :param p_bright: probability of encountering a bright state
        (default=0.5)

    :returns: threshold_count
    """
    thresh = np.log(p_bright / (1 - p_bright)) + mean_bright - mean_dark
    thresh /= np.log(mean_bright / mean_dark)
    return np.round(thresh)
