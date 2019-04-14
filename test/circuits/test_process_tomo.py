import numpy as np
import unittest
from numpy.testing import assert_allclose
from oitg.circuits.protocols.process_tomo.tools import *


class ProcessTomoCase(unittest.TestCase):
    def test_round_trip_conversion(self):
        """Test converting between different matrix representations."""

        for num_qubits in range(1, 5):
            # Since we know the implementations don't depend on well-formed input data,
            # just generate a random d^2 x d^2 matrix as test data for all conversions.
            dim = 2**(2 * num_qubits)
            mat = np.random.rand(dim, dim)
            assert_allclose(mat, vec2mat(mat2vec(mat)))
            assert_allclose(liou2choi(choi2liou(mat)), mat)
            assert_allclose(choi2liou(liou2choi(mat)), mat)
