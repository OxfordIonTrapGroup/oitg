import numpy as np
import unittest

from oitg.fitting.detuned_square_pulse import detuned_square_pulse


class SinusoidTest(unittest.TestCase):
    def test_simple_data(self):
        omega = 1e6
        t_pulse, offset, a, y0 = np.pi / omega + 1e-7, 2e3, 1.0, 0.0

        error = 0.01
        range = 10 * omega

        x = np.linspace(-range / 2, range / 2, 35)

        temp = np.sqrt(omega**2 + (x - offset)**2)
        y = np.sinc(temp * t_pulse / (2 * np.pi))**2
        y *= a * omega**2 * t_pulse**2 / 4
        y += y0
        y += np.random.normal(size=len(y), scale=error)

        p, p_err, x_fit, y_fit = detuned_square_pulse.fit(x,
                                                          y,
                                                          y_err=np.full(y.shape, error),
                                                          evaluate_function=True)

        # test precision
        self.assertAlmostEqual(omega, p['omega'], delta=3 * p_err['omega'])
        self.assertAlmostEqual(a, p['a'], delta=3 * p_err['a'])
        self.assertAlmostEqual(y0, p['y0'], delta=3 * p_err['y0'])
        self.assertAlmostEqual(t_pulse, p['t_pulse'], delta=3 * p_err['t_pulse'])
        self.assertAlmostEqual(offset, p['offset'], delta=3 * p_err['offset'])

        # derived data precision
        self.assertAlmostEqual(t_pulse - np.pi / omega,
                               p['t_error'],
                               delta=3 * p_err['t_error'])
        self.assertAlmostEqual(t_pulse * omega / np.pi - 1.0,
                               p['area_error'],
                               delta=3 * p_err['area_error'])

    def test_area_error_data(self):
        omega = 1e6
        t_pulse, offset, a, y0 = np.pi / omega + 1e-6, 2e3, 1.0, 0.0

        error = 0.02
        range = 10 * omega

        x = np.linspace(-range / 2, range / 2, 35)

        temp = np.sqrt(omega**2 + (x - offset)**2)
        y = np.sinc(temp * t_pulse / (2 * np.pi))**2
        y *= a * omega**2 * t_pulse**2 / 4
        y += y0
        y += np.random.normal(size=len(y), scale=error)

        p, p_err, x_fit, y_fit = detuned_square_pulse.fit(
            x,
            y,
            y_err=np.full(y.shape, error),
            evaluate_function=True,
            constants={'t_pulse': t_pulse})

        # test precision
        self.assertAlmostEqual(omega, p['omega'], delta=3 * p_err['omega'])
        self.assertAlmostEqual(a, p['a'], delta=3 * p_err['a'])
        self.assertAlmostEqual(y0, p['y0'], delta=3 * p_err['y0'])
        # self.assertAlmostEqual(t_pulse, p['t_pulse'],
        #                        delta=3*p_err['t_pulse'])
        self.assertAlmostEqual(offset, p['offset'], delta=3 * p_err['offset'])

        # derived data precision
        self.assertAlmostEqual(t_pulse - np.pi / omega,
                               p['t_error'],
                               delta=3 * p_err['t_error'])
        self.assertAlmostEqual(t_pulse * omega / np.pi - 1.0,
                               p['area_error'],
                               delta=3 * p_err['area_error'])

    def test_derived_params(self):
        omega = 1e6
        t_pulse, offset, a, y0 = np.pi / omega + 1e-7, 2e3, 1.0, 0.0

        error = 0.02
        range = 10 * omega

        x = np.linspace(-range / 2, range / 2, 35)

        temp = np.sqrt(omega**2 + (x - offset)**2)
        y = np.sinc(temp * t_pulse / (2 * np.pi))**2
        y *= a * omega**2 * t_pulse**2 / 4
        y += y0
        y += np.random.normal(size=len(y), scale=error)

        p, p_err, x_fit, y_fit = detuned_square_pulse.fit(
            x,
            y,
            y_err=np.full(y.shape, error),
            evaluate_function=True,
        )
        self.assertIn('t_error', p)
        self.assertIn('t_error', p_err)
        self.assertIn('area_error', p)
        self.assertIn('area_error', p_err)


if __name__ == '__main__':
    unittest.main()
