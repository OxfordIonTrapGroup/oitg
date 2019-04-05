import numpy as np
from scipy.special import eval_genlaguerre, factorial
from scipy.signal import lombscargle
from FitBase import FitBase


def gen_rab_freq(n0=0, n1=1, omega0=1.0, eta=0.1):
    """calculate rabi frequency associated with single photon, n0->n1 transition

    all inputs may be vectors
    n0, n1: SHM excitation states of the transition

    omega: strength of coupling term
           (would be rabi frequency if not for lamb-dicke effect)

    eta: lamb-dicke parameter

    returns rabi frequency of transition (vector if inputs are vectors)
   """
    n_s, n_l = np.sort(np.array([n0, n1]), axis=0)
    return omega0 * np.exp(-eta*eta/2) \
        * np.sqrt(factorial(n_s)/factorial(n_l)) \
        * eta**(n_l-n_s) * eval_genlaguerre(n_s, n_l-n_s, eta*eta)


def mk_omega_vec(n_max=171, delta_n=0, omega0=1, eta=0.1):
    """Rabi frequency for single photon transition

    wraps gen_rab_freq to return omega for lowest n_max SHM modes
    transitions change SHM excitation by delta_n

    returns vector of rabi frequencies
    """
    n0 = np.arange(n_max)
    n1 = n0 + delta_n  # negative n1 is un-physical!
    omega_vec = gen_rab_freq(n0, np.abs(n1), omega0, eta)
    omega_vec = np.where(n1 < 0, 0.0, omega_vec)
    return omega_vec


def sim_rabi(t_vec, omega_vec, delta, n_bar):
    """simulate transition probability

    t_vec: times to be evaluated

    omega_vec: vector of all involved rabi frequencies
               (as provided by mk_omega_vec)

    delta: detuning (rad s^-1)

    n_bar: mean occupation number

    returns vector of state 1 populations"""
    t = t_vec[:, np.newaxis]
    # oscilation freq is effected by detuning!
    omega_eff = np.sqrt(delta*delta + omega_vec*omega_vec)[np.newaxis, :]
    # transition fraction of each transition
    pop_vec = np.where(omega_eff != 0,
                       (omega_vec/omega_eff * np.sin(omega_eff/2*t))**2,
                       0)
    # weight according to thermal dist (derive through partition function)
    weights = (n_bar/(n_bar+1))**(np.arange(len(omega_vec))) / (n_bar+1)
    weights = weights/sum(weights)  # re-normalise due to truncation

    pop_vec = pop_vec * weights[np.newaxis, :]
    return np.sum(pop_vec, axis=1)


def fitting_function(t, p):
    # omega0, omega1 are of approximatly unit amplitude
    omega_eff = p['omega'] * p['omega_factor_vec']
    if not isinstance(t, np.ndarray):
        t = np.array([t])
    return sim_rabi(t, omega_eff, p['net_det'], p['n_bar'])


def parameter_initialiser(t, y, p):
    """initialises suggested fit parameters"""
    # lombscargle accepts irregularly sampled data (use sane bounds != 0)
    omega = np.linspace(1e2, 1e7, 10000)  # rad s^-1
    p['omega'] = omega[np.argmax(lombscargle(t, y, omega, precenter=True))]
    p['n_bar'] = 1  # small enough to fit n_bar=0.1


def mk_const_param(dn=0, eta=0.248309):
    """dn: 0 (+-1) for carrier (sideband)
    eta: lamb-dicke parameter
    :returns: constant param dict"""
    p = {}
    p['net_det'] = 0.0
    p['omega_factor_vec'] = mk_omega_vec(171, dn, 1.0, eta)
    return p


# pre-configured fitting object (specify constant params in fit call)
dipole_bsb_car_rsb_fit = FitBase(
    ['omega_factor_vec', 'net_det', 'omega', 'n_bar'],
    fitting_function, parameter_initialiser,
    parameter_bounds={'eta': (0, np.inf), 'n_bar': (0, np.inf)})

if __name__ == "__main__":  # example & debug code
    from oitg.uncertainty_to_string import uncertainty_to_string
    from matplotlib import pyplot as plt
    # input parameters
    nmax, eta, delta_n, net_det, n_bar, omega_eff = \
        171, 2.48309e-1, 0, 0.0, 15.3, 2.5e5
    # sample times - these don't need to be equally spaced
    t = 2 * np.pi/omega_eff * np.linspace(0, 2.5, 500)

    # simulate data for fitting
    omega_eff_vec = mk_omega_vec(nmax, delta_n=delta_n,
                                 omega0=omega_eff, eta=eta)
    y = sim_rabi(t, omega_eff_vec, net_det, n_bar) \
        + np.random.random(t.shape)*0.05

    plt.figure()
    plt.plot(t*omega_eff/(2*np.pi), y)
    plt.ylim([0, 1])
    plt.xlim([min(t*omega_eff/(2*np.pi)), max(t*omega_eff/(2*np.pi))])
    plt.ylabel("population")
    plt.xlabel("time*omega_eff/2pi")

    # fit to data
    p_fit, p_err, fit_x, fit_y = dipole_bsb_car_rsb_fit.fit(
        t, y, constants=mk_const_param(delta_n, eta), evaluate_function=True)

    plt.plot(fit_x*omega_eff/(2*np.pi), fit_y)
    plt.show()
    print("n_bar", uncertainty_to_string(p_fit['n_bar'], p_err['n_bar']))
    print("omega", uncertainty_to_string(p_fit['omega'], p_err['omega']))

    if False:  # frequency initial guess introspection (fit debugging)
        plt.figure()
        f = np.linspace(1e-1, 5, 10000)
        plt.plot(f, lombscargle(t, y, f, precenter=True))
        from numpy.fft import rfftfreq, rfft
        plt.plot(rfftfreq(len(t), t[1]-t[0])*2*np.pi,
                 np.abs(rfft(y-np.mean(y), norm="ortho")))
        plt.show()
