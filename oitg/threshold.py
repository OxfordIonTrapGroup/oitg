import numpy as np
from scipy.optimize import least_squares
from oitg.fitting.poisson import poisson_iter


def calc_target_bin_time(bright_rate, dark_rate, p_error_target, p_bright=0.5):
    """calculate optimal treshold bin time for target error chance

    The calculation assumes both bright and dark counts are
    Poisson distributed and gives a threshold minimising the error probability

    :param bright_rate: expected bright count rate in $s^-1$
    :param dark_rate: expected dark count ratein $s^-1$
    :param p_error_target: target error probability
    :param p_bright: probability of encountering a bright state (default=0.5)

    :returns: (target_t_bin [s], threshold_rate [$s^-1$])"""
    def residuals(t_bin):
        return (
            calc_p_error(bright_rate, dark_rate, t_bin[0], p_bright=p_bright)
            - p_error_target
        )

    t_bin_init = 1e-3
    result = least_squares(residuals, t_bin_init,
                           bounds=(0, np.inf),
                           x_scale=(1e-2,), f_scale=(0.1*p_error_target)**2)
    # assert result['cost'] < (0.1*p_error_target)**2, \
    #     "result ing error outside of tolerance"

    thresh_rate = calc_thresh_rate(bright_rate, dark_rate,
                                   p_bright=p_bright, t_bin=result['x'][0])

    return result['x'][0], thresh_rate


def calc_p_error(bright_rate, dark_rate, t_bin, p_bright=0.5):
    """assumes exact threshold count is evaluated as dark

    :param bright_rate: expected bright count rate in $s^-1$
    :param dark_rate: expected dark count ratein $s^-1$
    :param t_bin: integration time in s.
    :param p_bright: probability of encountering a bright state (default=0.5)
    """
    thresh_rate = calc_thresh_rate(bright_rate, dark_rate,
                                   p_bright=p_bright, t_bin=t_bin)
    thresh_count = np.ceil(thresh_rate*t_bin).astype(np.int_)
    # if use_sum:  # this breaks for
    n_vec = np.arange(thresh_count + 1, dtype=np.int_)

    p_error = (1 - p_bright) * (1 - np.sum(
        poisson_iter(n_vec[:-1], dark_rate * t_bin)))
    p_error += p_bright * np.sum(
        poisson_iter(n_vec, bright_rate * t_bin))
    return p_error


def calc_thresh_rate(bright_rate, dark_rate, t_bin=1e-3, p_bright=0.5):
    """Optimal threshold rate for distinguishing bright and dark states

    The calculation assumes both bright and dark counts are
    Poisson distributed and gives a threshold minimising the error probability

    :param bright_rate: expected bright count rate in $s^-1$
    :param dark_rate: expected dark count ratein $s^-1$
    :param t_bin: integration time in s. Only relevant for p_bright!=0.5
        (default=1e-3)
    :param p_bright: probability of encountering a bright state
        (default=0.5)

    :returns: threshold_rate /$s^-1$
    """
    thresh = np.log(p_bright/(1-p_bright))/t_bin + bright_rate - dark_rate
    thresh /= np.log(bright_rate/dark_rate)
    return thresh


if __name__=="__main__":
    test_threshhold = True
    test_calc_p_error = True
    test_bin_time = True


    bright = 4e4
    dark = 2e4
    p_bright = 0.5
    error_target = 1e-3
    t_bin = 2e-3

    if test_threshhold:
        print(calc_thresh_rate(bright, dark, t_bin=t_bin, p_bright=p_bright))

    if test_calc_p_error:
        print("p_error_calc", calc_p_error(bright, dark, t_bin, p_bright))

    if test_bin_time:
        t_bin, thresh_rate = calc_target_bin_time(
            bright, dark, error_target, p_bright=p_bright)
        print("t_bin, thresh_rate", t_bin, thresh_rate)
        print("p_error for this bin",
              calc_p_error(bright, dark, t_bin, p_bright))

