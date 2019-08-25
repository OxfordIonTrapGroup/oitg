import numpy as np
import unittest

from oitg.fitting.sinusoid import sinusoid


class SinusoidTest(unittest.TestCase):
    def test_random_data(self):

        n_sample = 30
        t_max = 1e-1
        amp = 1.5
        rel_noise = 0.1
        offset = 40
        phi = np.pi / 2
        t_dead = 0.0
        omega = np.pi / t_max * 8.2

        t = np.zeros(n_sample)
        t[1:] = t_max * np.random.rand(n_sample - 1)

        y = np.sin((t - t_dead) * omega + phi)
        y += rel_noise * np.random.normal(size=n_sample)
        y *= amp
        y += offset

        # hold constant during dead time
        y = np.where(t > t_dead, y, offset + amp * np.sin(phi))

        # fix these fit parameters to a specific value
        const_dict = {
            't_dead': t_dead,
            # 'phi': phi,
            # 'c': offset
        }
        p, p_err = sinusoid.fit(t,
                                y,
                                y_err=np.ones(y.shape) * amp * rel_noise,
                                evaluate_function=False,
                                evaluate_x_limit=[0, t_max],
                                constants=const_dict)

        self.assertAlmostEqual(omega, p['omega'], delta=4 * p_err['omega'])
        self.assertAlmostEqual(amp, p['a'], delta=4 * p_err['a'])
        self.assertAlmostEqual(offset, p['c'], delta=4 * p_err['c'])
        self.assertAlmostEqual(phi % (2 * np.pi),
                               p['phi'] % (2 * np.pi),
                               delta=4 * p_err['phi'])
        self.assertGreaterEqual(t_dead, p['t_dead'] - 4 * p_err['t_dead'])

    def test_random_data_fixed_phi(self):
        n_sample = 40
        t_max = 1e-6
        amp = 0.1
        rel_noise = 0.1
        offset = 40
        phi = np.pi
        t_dead = 0.0
        omega = np.pi / t_max * 9.3

        t = np.linspace(0, t_max, n_sample)

        y = np.sin((t - t_dead) * omega + phi)
        y += rel_noise * np.random.normal(size=n_sample)
        y *= amp
        y += offset

        # hold constant during dead time
        y = np.where(t > t_dead, y, offset + amp * np.sin(phi))

        # fix these fit parameters to a specific value
        const_dict = {
            # 't_dead': t_dead,
            'phi': phi,
            # 'c': offset
        }
        p, p_err = sinusoid.fit(t,
                                y,
                                y_err=np.ones(y.shape) * amp * rel_noise,
                                evaluate_function=False,
                                evaluate_x_limit=[0, t_max],
                                constants=const_dict)

        self.assertAlmostEqual(omega, p['omega'], delta=4 * p_err['omega'])
        self.assertAlmostEqual(amp, p['a'], delta=4 * p_err['a'])
        self.assertAlmostEqual(offset, p['c'], delta=4 * p_err['c'])
        self.assertAlmostEqual(phi % (2 * np.pi),
                               p['phi'] % (2 * np.pi),
                               delta=4 * p_err['phi'])
        self.assertGreaterEqual(t_dead, p['t_dead'] - 4 * p_err['t_dead'])

    def test_pi_pulse(self):
        n_sample = 21
        t_max = 1e-4
        amp = 0.5
        rel_noise = 0.1
        offset = 0.5
        phi = np.pi / 2
        t_dead = 0.0
        omega = 1.3 * np.pi / t_max

        t = np.linspace(0, t_max, n_sample)

        y = np.sin((t - t_dead) * omega + phi)
        y += rel_noise * np.random.normal(size=n_sample)
        y *= amp
        y += offset

        # hold constant during dead time
        y = np.where(t > t_dead, y, offset + amp * np.sin(phi))

        # fix these fit parameters to a specific value
        const_dict = {
            't_dead': t_dead,
            # 'phi': phi,
            # 'c': offset
        }
        p, p_err = sinusoid.fit(t,
                                y,
                                y_err=np.ones(y.shape) * amp * rel_noise,
                                evaluate_function=False,
                                evaluate_x_limit=[0, t_max],
                                constants=const_dict)

        self.assertAlmostEqual(omega, p['omega'], delta=4 * p_err['omega'])
        self.assertAlmostEqual(amp, p['a'], delta=4 * p_err['a'])
        self.assertAlmostEqual(offset, p['c'], delta=4 * p_err['c'])
        self.assertAlmostEqual(phi % (2 * np.pi),
                               p['phi'] % (2 * np.pi),
                               delta=4 * p_err['phi'])
        self.assertGreaterEqual(t_dead, p['t_dead'] - 4 * p_err['t_dead'])

    def test_pi_pulse_with_t_dead(self):
        n_sample = 21
        t_max = 1e-2
        amp = 0.5
        rel_noise = 0.05
        offset = 0.5
        phi = np.pi / 2
        t_dead = t_max * 0.24
        omega = 1.3 * np.pi / (t_max - t_dead)

        t = np.linspace(0, t_max, n_sample)

        y = np.sin((t - t_dead) * omega + phi)
        y += rel_noise * np.random.normal(size=n_sample)
        y *= amp
        y += offset

        # hold constant during dead time
        y = np.where(t > t_dead, y, offset + amp * np.sin(phi))

        # fix these fit parameters to a specific value
        const_dict = {
            # 't_dead': t_dead,
            'phi': phi,
            # 'c': offset
        }
        p, p_err = sinusoid.fit(t,
                                y,
                                y_err=np.ones(y.shape) * amp * rel_noise,
                                evaluate_function=False,
                                evaluate_x_limit=[0, t_max],
                                constants=const_dict)

        self.assertAlmostEqual(omega, p['omega'], delta=4 * p_err['omega'])
        self.assertAlmostEqual(amp, p['a'], delta=4 * p_err['a'])
        self.assertAlmostEqual(offset, p['c'], delta=4 * p_err['c'])
        self.assertAlmostEqual(phi % (2 * np.pi),
                               p['phi'] % (2 * np.pi),
                               delta=4 * p_err['phi'])
        self.assertAlmostEqual(t_dead, p['t_dead'], delta=4 * p_err['t_dead'])

    def test_delayed_split_data_with_t_dead(self):
        n0 = 40
        n1 = 30
        t0 = 1e-2
        t1 = t0 + 12.3e-4
        t2 = 2.1e-2
        t3 = t2 + 5.2e-4

        amp = 1.5
        rel_noise = 0.1
        offset = -5
        phi = np.pi / 2
        t_dead = (t3 - t2) * 0.14 + t0
        omega = np.pi / (t3 - t2) * 4.2

        t = np.concatenate((np.linspace(t0, t1, n0), np.linspace(t2, t3, n1)), axis=0)

        y = np.sin((t - t_dead) * omega + phi)
        y += rel_noise * np.random.normal(size=n0 + n1)
        y *= amp
        y += offset

        # hold constant during dead time
        y = np.where(t > t_dead, y, offset + amp * np.sin(phi))

        # fix these fit parameters to a specific value
        const_dict = {
            # 't_dead': t_dead,
            'phi': phi,
            # 'c': offset
        }
        p, p_err = sinusoid.fit(t,
                                y,
                                y_err=np.ones(y.shape) * amp * rel_noise,
                                evaluate_function=False,
                                evaluate_x_limit=[0, t3],
                                constants=const_dict)

        self.assertAlmostEqual(omega, p['omega'], delta=4 * p_err['omega'])
        self.assertAlmostEqual(amp, p['a'], delta=4 * p_err['a'])
        self.assertAlmostEqual(offset, p['c'], delta=4 * p_err['c'])
        self.assertAlmostEqual(phi % (2 * np.pi),
                               p['phi'] % (2 * np.pi),
                               delta=4 * p_err['phi'])
        self.assertAlmostEqual(t_dead, p['t_dead'], delta=4 * p_err['t_dead'])


if __name__ == '__main__':
    unittest.main()
