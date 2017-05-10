"""
Hyperfine Transition.

Module to assist in calculation of hyperfine transition frequencies
of Ca43 using the Breit-Rabi formula.

All values are handled in SI units, so:
    Magnetic fields are in Tesla
    Frequencies are in Hz
"""
from oitg import units
import math
import scipy.optimize

# Hyperfine splitting for Ca43
hyperfine_splitting = -3.2256082864E9

nuclear_spin = 7 / 2

# g-factors taken from Tom's thesis
g_j = 2.00225664
g_i = -1.315348 * 2 / 7

# Magnetic field strength of 146G clock qubit
b0 = 146.0942e-4

# Magnetron
mu = (g_j * units.mu_N + g_j * units.mu_B)
mu /= units.h * hyperfine_splitting


def _breit_rabi(self, b, m_f, f_sign):
    """
    Breit-Rabi formula.

    From Woodgate, p.193
    sign = +1 for F=4, sign = -1 for F=3
    """
    frequency_shift = (
        - units.h * hyperfine_splitting / (2 * (2 * nuclear_spin + 1)) -
        b * g_i * units.mu_N * m_f +
        (f_sign * units.h * hyperfine_splitting /
            2 * math.sqrt(
                1 + 2 * m_f * (b * mu) /
                (nuclear_spin + (1 / 2)) +
                (b * mu)**2)))

    frequency_shift /= units.h

    return frequency_shift


def transition_frequency(m_f4, m_f3, b=None,
                         relative=False):
    """
    Calculate the transition frequency between two hyperfine states.

    This used to be called df_trans/df_trans_det.
    F=4,m_f4 -> F=3,m_f3 for given magnetic field, b. If field is None
    then the ~146G clock field is used.
    If relative is True, then the frequency is given has the zero-field
    hyperfine splitting subtracted.
    """
    if b is None:
        b = b0

    f = _breit_rabi(b, m_f3, -1) - _breit_rabi(b, m_f4, 1)

    if relative:
        f -= abs(hyperfine_splitting)
    return f


def calculate_b_from_frequency(m_f4, m_f3, frequency):
    """
    Calculate the field from a transition frequency.

    Calculates the B-field necessary for the transition
    F=4,mF4 -> F=3,mF3 to have the given frequency
    """
    def function_to_minimise(b, m_f4, m_f3, frequency):
        value = transition_frequency(m_f4=m_f4, m_f3=m_f3, b=b)
        value -= frequency
        return value

    b = scipy.optimize.fsolve(
        function_to_minimise,
        b0,
        args=(m_f4, m_f3, frequency))

    return b[0]
