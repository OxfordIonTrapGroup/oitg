import numpy as np
from scipy.integrate import quad
from scipy.special import gamma


def poisson_iter(n, mean):
    """probability of n given Poisson distribution with mean

    iterative calculation avoids the breakdown of
    np.exp(-mean) * mean**n / factorial(n)
    for n>10

    This does not require n to be an integer, allowing the distribution to be
    integrated.

    can accept vector n"""
    max_iter = int(np.max(n))
    ind_offset = n%1
    value = np.exp(-mean) * mean**ind_offset /gamma(1+ind_offset)
    if hasattr(n, "__len__"):
        is_arr = True
    else:
        is_arr = False
    for ind in range(max_iter):
        if is_arr:
            mask = np.argwhere(n>ind)
            value[mask] *= mean / (ind+ind_offset[mask]+1)
        else:
            value *= mean / (ind+ind_offset+1)
    return value


def poisson_large_n(n, mean):
    """
    probability of n given Poisson distribution with mean

    iterative calculation is slow for very large n.

    We need to approximate the Poisson distribution for large n
    critically the term mean^n/n! becomes small at large n
    we therefore want to approximate this term directly, avoiding
    large numbers.

    mean^n/n! = exp(ln(mean)*n)/gamma(n+1)
    = 1/integral{x=0->inf}( exp(-x + n ln(x/mean)))

    Conveniently this form does not require n to be an integer, allowing
    integration of this distribution

    only accepts scalar n"""
    def integrand(x, n, mean):
        return np.exp(-x + n * np.log(x/mean))
    return np.exp(-mean) / quad(integrand, 0, np.inf,
                                args=(n, mean), epsabs=0,
                                )[0]

# TODO: add a fitting routine for this


if __name__=='__main__':
    test_poission = True

    if test_poission:
        from matplotlib import pyplot as plt

        from scipy.stats import poisson

        mean = 200
        n = np.arange(300)
        fig, ax = plt.subplots(2,1, sharex=True)

        ax[0].plot(n, poisson.pmf(n, mu=mean), label="scipy")
        ax[0].plot(n, poisson_iter(n, mean), label="iter")
        p_large_n = np.zeros(n.shape)
        for idx in n:
            p_large_n[idx] = poisson_large_n(idx, mean)
        ax[0].plot(n, p_large_n, label="integrate")
        ax[0].legend()
        ax[0].set_ylabel("pmf")
        ax[1].plot(n, poisson_iter(n, mean)-poisson.pmf(n, mu=mean),
                   label="iter")
        ax[1].plot(n, p_large_n-poisson.pmf(n, mu=mean), label="integrate")
        ax[1].legend()
        ax[1].set_ylabel("deviation from scipy")
        ax[1].set_xlabel("n")
        plt.show()