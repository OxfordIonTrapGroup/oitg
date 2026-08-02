import numpy as np
import unittest

from oitg.fitting.rabi_flop import fitting_function, rabi_flop


def simulate(t, p_true, num_shots=50, seed=0):
    """Simulate a thresholded readout time scan with binomial projection
    noise, returning (y, y_err) as they would arrive from the experiment."""
    rng = np.random.default_rng(seed)
    counts = rng.binomial(num_shots, fitting_function(t, p_true))
    y = counts / num_shots
    p_hat = (counts + 1) / (num_shots + 2)
    y_err = np.sqrt(p_hat * (1 - p_hat) / (num_shots + 2))
    return y, y_err


class RabiFlopTest(unittest.TestCase):
    def _check_period(self, t_period, tau_decay, y_lower, t_dead=0.0, t=None):
        if t is None:
            t = np.linspace(0.0, 4e-3, 51)
        p_true = {
            "t_period": t_period,
            "t_dead": t_dead,
            "y_lower": y_lower,
            "tau_decay": tau_decay
        }
        for seed in range(3):
            y, y_err = simulate(t, p_true, seed=seed)
            p, p_err = rabi_flop.fit(t, y, y_err)
            self.assertAlmostEqual(p["t_period"],
                                   t_period,
                                   delta=max(0.05 * t_period,
                                             4 * p_err["t_period"]))

    def test_many_periods(self):
        # ~13 oscillations per scan, no visible decay.
        self._check_period(0.3e-3, np.inf, 0.0)

    def test_many_periods_decaying(self):
        self._check_period(0.4e-3, 2e-3, 0.0)

    def test_moderate(self):
        self._check_period(1e-3, 20e-3, 0.25)

    def test_slow(self):
        # Only just above one period per scan.
        self._check_period(3.5e-3, np.inf, 0.0)

    def test_reduced_contrast(self):
        self._check_period(0.5e-3, np.inf, 0.4)

    def test_irregular_sampling(self):
        rng = np.random.default_rng(42)
        t = np.sort(rng.uniform(0.0, 4e-3, 51))
        self._check_period(0.45e-3, 20e-3, 0.2, t=t)

    def test_overdamped(self):
        # Decays away before completing a full oscillation; the period is
        # only loosely defined, so just require the fit to match the data.
        t = np.linspace(0.0, 4e-3, 51)
        p_true = {
            "t_period": 2.5e-3,
            "t_dead": 0.0,
            "y_lower": 0.0,
            "tau_decay": 1e-3
        }
        y, y_err = simulate(t, p_true)
        p, p_err, residuals = rabi_flop.fit(t, y, y_err, calculate_residuals=True)
        self.assertLess(np.sqrt(np.mean(residuals**2)), 0.1)

    def test_t_dead(self):
        t = np.linspace(0.0, 4e-3, 51)
        p_true = {
            "t_period": 1e-3,
            "t_dead": 40e-6,
            "y_lower": 0.0,
            "tau_decay": np.inf
        }
        y, y_err = simulate(t, p_true)
        p, p_err = rabi_flop.fit(t, y, y_err)
        self.assertAlmostEqual(p["t_dead"],
                               40e-6,
                               delta=max(20e-6, 4 * p_err["t_dead"]))
        self.assertAlmostEqual(p["t_pi"], 540e-6, delta=30e-6)


if __name__ == '__main__':
    unittest.main()
