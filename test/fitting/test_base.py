import numpy as np
import unittest
from oitg.fitting import cos, FitError


class TestUndefinedParam(unittest.TestCase):
    def test_undefined_constant(self):
        with self.assertRaises(FitError):
            cos.fit([1, 2, 3, 4], [5, 6, 7, 8], constants={"this_does_not_exist": 0.0})

    def test_undefined_initial_value(self):
        with self.assertRaises(FitError):
            cos.fit([1, 2, 3, 4], [5, 6, 7, 8], initialise={"this_does_not_exist": 0.0})
