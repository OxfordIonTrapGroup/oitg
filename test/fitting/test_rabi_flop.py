import unittest

import numpy as np
from scipy.optimize import minimize_scalar

from oitg.fitting.rabi_flop import (
    derived_parameter_function,
    fitting_function,
    rabi_flop,
)


def simulate(t, p_true, num_shots=50, seed=0):
    """Simulate a thresholded readout time scan with binomial projection
    noise, returning (y, y_err) as they would arrive from the experiment."""
    rng = np.random.default_rng(seed)
    counts = rng.binomial(num_shots, fitting_function(t, p_true))
    y = counts / num_shots
    p_hat = (counts + 1) / (num_shots + 2)
    y_err = np.sqrt(p_hat * (1 - p_hat) / (num_shots + 2))
    return y, y_err


def make_params(t_period, tau_decay, y_start, y_lower, t_dead=0.0):
    return {
        "t_period": t_period,
        "t_dead": t_dead,
        "y_start": y_start,
        "y_lower": y_lower,
        "tau_decay": tau_decay,
    }


def first_max_transfer_time(p):
    """Reference for t_pi independent of the model internals (just numerically optimise
    the maximum distance from y_start within the first 3/4 of a period plus dead time).
    """
    result = minimize_scalar(
        lambda t: -abs(float(fitting_function(t, p)) - p["y_start"]),
        method="bounded",
        bounds=(p["t_dead"], p["t_dead"] + 0.75 * p["t_period"]),
        options={"xatol": 1e-11},
    )
    assert result.success
    return result.x


class RabiFlopTest(unittest.TestCase):
    def _check_period(self, t_period, tau_decay, contrast_loss, t_dead=0.0, t=None):
        """Check the fitted period for both a flop starting at y = 1 (ending at
        y = contrast_loss) and one starting at y = 0 (ending at y = 1 - contrast_loss).
        """
        if t is None:
            t = np.linspace(0.0, 4e-3, 51)
        for y_start in (1.0, 0.0):
            y_lower = abs(y_start - (1.0 - contrast_loss))
            p_true = make_params(t_period, tau_decay, y_start, y_lower, t_dead)
            for seed in range(3):
                with self.subTest(y_start=y_start, seed=seed):
                    y, y_err = simulate(t, p_true, seed=seed)
                    p, p_err = rabi_flop.fit(t, y, y_err)
                    self.assertAlmostEqual(
                        p["t_period"],
                        t_period,
                        delta=max(0.05 * t_period, 4 * p_err["t_period"]),
                    )
                    # The starting level should have been inferred from the
                    # data and held constant.
                    self.assertEqual(p["y_start"], y_start)
                    self.assertEqual(p_err["y_start"], 0.0)

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

    def test_geometric_sampling(self):
        # Geometrically spaced scan covering ~25 periods with only a few
        # samples per period towards the end, as e.g. used for scans where
        # the frequency is not known in advance.
        t = np.geomspace(3e-6, 350e-6, 51)
        self._check_period(13e-6, 300e-6, 0.05, t=t)

    def test_overdamped(self):
        # Decays away before completing a full oscillation; the period is
        # only loosely defined, so just require the fit to match the data.
        t = np.linspace(0.0, 4e-3, 51)
        for y_start, y_lower in ((1.0, 0.0), (0.0, 1.0)):
            p_true = make_params(2.5e-3, 1e-3, y_start, y_lower)
            y, y_err = simulate(t, p_true)
            p, p_err, residuals = rabi_flop.fit(t, y, y_err, calculate_residuals=True)
            self.assertLess(np.sqrt(np.mean(residuals**2)), 0.1)

    def test_t_dead(self):
        t = np.linspace(0.0, 4e-3, 51)
        for y_start, y_lower in ((1.0, 0.0), (0.0, 1.0)):
            p_true = make_params(1e-3, np.inf, y_start, y_lower, t_dead=40e-6)
            y, y_err = simulate(t, p_true)
            p, p_err = rabi_flop.fit(t, y, y_err)
            self.assertAlmostEqual(
                p["t_dead"], 40e-6, delta=max(20e-6, 4 * p_err["t_dead"])
            )
            self.assertAlmostEqual(p["t_pi"], 540e-6, delta=30e-6)

    def test_t_pi(self):
        # t_pi should be the first point of maximum population transfer regardless
        # of the direction of the flop, shifted towards zero for finite decay times
        # and offset by the dead time.
        t = np.linspace(0.0, 4e-3, 51)
        t_period = 1e-3
        for y_start in (1.0, 0.0):
            for tau_decay in (np.inf, 2e-3, 0.5e-3):
                for t_dead in (0.0, 40e-6):
                    with self.subTest(
                        y_start=y_start, tau_decay=tau_decay, t_dead=t_dead
                    ):
                        p_true = make_params(
                            t_period, tau_decay, y_start, abs(y_start - 0.9), t_dead
                        )
                        t_pi_true = first_max_transfer_time(p_true)

                        # Check derived parameter calculation to high precision.
                        p, _ = derived_parameter_function(
                            dict(p_true), {k: 0.0 for k in p_true}
                        )
                        self.assertAlmostEqual(p["t_pi"], t_pi_true, delta=1e-9)
                        if np.isinf(tau_decay):
                            self.assertAlmostEqual(
                                p["t_pi"], t_dead + t_period / 2, delta=1e-9
                            )
                        else:
                            self.assertLess(p["t_pi"], t_dead + t_period / 2)
                            self.assertGreater(p["t_pi"], t_dead + t_period / 4)

                        # Simulate readout noise and check with lower precision.
                        y, y_err = simulate(t, p_true)
                        p, p_err = rabi_flop.fit(t, y, y_err)
                        self.assertEqual(p["y_start"], y_start)
                        self.assertAlmostEqual(
                            p["t_pi"], t_pi_true, delta=max(30e-6, 4 * p_err["t_pi"])
                        )
                        self.assertGreater(p_err["t_pi"], 0.0)

    def test_user_specified_y_start(self):
        t = np.linspace(0.0, 4e-3, 51)
        y, y_err = simulate(t, make_params(1e-3, np.inf, 0.0, 0.9))

        # Explicitly specifying the starting level as a constant overrides the
        # inference from the data.
        p, p_err = rabi_flop.fit(t, y, y_err, constants={"y_start": 1.0})
        self.assertEqual(p["y_start"], 1.0)
        self.assertEqual(p_err["y_start"], 0.0)

        # Given an initial value, the starting level is a free fit parameter.
        p, p_err = rabi_flop.fit(t, y, y_err, initialise={"y_start": 0.1})
        self.assertAlmostEqual(p["y_start"], 0.0, delta=max(0.05, 4 * p_err["y_start"]))
        self.assertGreater(p_err["y_start"], 0.0)
        self.assertAlmostEqual(p["y_lower"], 0.9, delta=max(0.05, 4 * p_err["y_lower"]))
        self.assertAlmostEqual(
            p["t_period"], 1e-3, delta=max(0.05e-3, 4 * p_err["t_period"])
        )

    def test_user_specified_y_lower(self):
        # A user-specified y_lower should not keep the direction of the flop from being
        # inferred correctly.
        t = np.linspace(0.0, 4e-3, 51)
        for y_start in (1.0, 0.0):
            y_lower = abs(y_start - 0.9)
            y, y_err = simulate(t, make_params(1e-3, np.inf, y_start, y_lower))
            p, p_err = rabi_flop.fit(t, y, y_err, constants={"y_lower": y_lower})
            self.assertEqual(p["y_start"], y_start)
            self.assertEqual(p["y_lower"], y_lower)
            self.assertAlmostEqual(
                p["t_period"], 1e-3, delta=max(0.05e-3, 4 * p_err["t_period"])
            )

    def test_legacy_parameters(self):
        # Parameter sets from before y_start was introduced (as e.g. stored in old fit
        # annotations) should still evaluate as flops starting at y = 1.
        t = np.linspace(0.0, 4e-3, 51)
        legacy = {"t_period": 1e-3, "t_dead": 0.0, "y_lower": 0.2, "tau_decay": 2e-3}
        np.testing.assert_allclose(
            fitting_function(t, legacy),
            fitting_function(t, make_params(1e-3, 2e-3, 1.0, 0.2)),
        )


if __name__ == "__main__":
    unittest.main()
