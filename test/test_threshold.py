import unittest
from oitg.threshold import *


class ThresholdTest(unittest.TestCase):
    def test_threshold(self):
        bright = 4e4
        dark = 2e4
        t_bin = 2e-3
        p_bright = 0.5
        calc_thresh_rate(bright, dark, t_bin=t_bin, p_bright=p_bright)

    def test_calc_p_error(self):
        bright = 4e4
        dark = 2e4
        t_bin = 2e-3
        p_bright = 0.5
        calc_p_error(bright, dark, t_bin, p_bright)

    def test_bin_time(self):
        bright = 4e4
        dark = 2e4
        p_bright = 0.5
        error_target = 1e-3

        t_bin, thresh_rate = calc_target_bin_time(bright,
                                                  dark,
                                                  error_target,
                                                  p_bright=p_bright)

        self.assertAlmostEqual(error_target,
                               calc_p_error(bright, dark, t_bin, p_bright),
                               delta=0.01 * error_target)
        self.assertAlmostEqual(
            thresh_rate, calc_thresh_rate(bright, dark, t_bin=t_bin, p_bright=p_bright))


if __name__ == '__main__':
    unittest.main()
