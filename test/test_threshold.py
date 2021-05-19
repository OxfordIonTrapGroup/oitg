import unittest
from oitg.threshold import *


class ThresholdTest(unittest.TestCase):
    def test_optimise_readout(self):
        bright_rate = 9.e4
        dark_rate = 3e4
        p_bright = 0.5
        dark_to_bright_rate = 1 / 1.168
        t_bin, threshold, p_error = optimise_readout(
            bright_rate,
            dark_rate,
            dark_to_bright_rate=dark_to_bright_rate,
            p_bright=p_bright,
        )
        self.assertAlmostEqual(0.0010438517809917116,
                               t_bin,
                               delta=0.01 * 0.0010438517809917116)
        self.assertEqual(58, threshold)
        self.assertAlmostEqual(0.000285363868745659,
                               p_error,
                               delta=0.01 * 0.000285363868745659)

    def test_optimise_t_bin(self):
        bright_rate = 9.e4
        dark_rate = 3e4
        p_bright = 0.5
        dark_to_bright_rate = 1 / 1.168
        t_bin, p_error = optimise_t_bin(
            bright_rate,
            dark_rate,
            58,
            dark_to_bright_rate=dark_to_bright_rate,
            p_bright=p_bright,
        )
        self.assertAlmostEqual(0.0010438517809917116,
                               t_bin,
                               delta=0.01 * 0.0010438517809917116)
        self.assertAlmostEqual(0.000285363868745659,
                               p_error,
                               delta=0.01 * 0.000285363868745659)


if __name__ == '__main__':
    unittest.main()
