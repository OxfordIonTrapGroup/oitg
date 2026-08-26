import unittest

from oitg.uncertainty_to_string import uncertainty_to_string

NAN = float("nan")
INF = float("inf")


class UncertaintyToStringTest(unittest.TestCase):
    def test_shortest_representation(self):
        self.assertEqual(uncertainty_to_string(0, 1e-4), "0(1)e-4")
        self.assertEqual(uncertainty_to_string(12.34567, 0.00123, 2), "12.3457(12)")
        self.assertEqual(
            uncertainty_to_string(-0.123456, 0.000123, 3), "-0.123456(123)"
        )
        self.assertEqual(
            uncertainty_to_string(0.001234560000, 0.000000012345, 4),
            "0.00123456000(1234)",
        )
        self.assertEqual(
            uncertainty_to_string(-0.0000123456, 0.0000001234), "-1.23(1)e-5"
        )

    def test_value_smaller_than_uncertainty(self):
        # The nominal value is rounded to the last significant digit of the
        # uncertainty, so it collapses to zero if it is small compared to the latter.
        self.assertEqual(uncertainty_to_string(10, 100), "0(100)")
        self.assertEqual(uncertainty_to_string(-10, 100), "0(100)")
        self.assertEqual(uncertainty_to_string(60, 100), "1(1)e2")

    def test_negative_uncertainty(self):
        self.assertEqual(
            uncertainty_to_string(12.34567, -0.00123, 2),
            uncertainty_to_string(12.34567, 0.00123, 2),
        )

    def test_zero_uncertainty_rejected(self):
        with self.assertRaises(AssertionError):
            uncertainty_to_string(1.0, 0.0)

    def test_nan_value(self):
        self.assertEqual(uncertainty_to_string(NAN, 1.0), "NaN")
        self.assertEqual(uncertainty_to_string(NAN, NAN), "NaN")
        self.assertEqual(uncertainty_to_string(NAN, INF), "NaN")

    def test_nan_uncertainty(self):
        self.assertEqual(uncertainty_to_string(1.5, NAN), "1.5(NaN)")
        self.assertEqual(uncertainty_to_string(0, NAN), "0(NaN)")
        self.assertEqual(uncertainty_to_string(INF, NAN), "inf(NaN)")

    def test_inf(self):
        self.assertEqual(uncertainty_to_string(INF, 1.0), "inf")
        self.assertEqual(uncertainty_to_string(INF, INF), "inf")
        self.assertEqual(uncertainty_to_string(1.0, INF), "1.0(inf)")


if __name__ == "__main__":
    unittest.main()
