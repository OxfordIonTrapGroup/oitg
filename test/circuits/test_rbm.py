import unittest

from oitg.circuits.clifford import *
from oitg.circuits.clifford_sym_qutrit import *
from oitg.circuits.gate import *
from oitg.circuits.protocols.rbm.generate import generate_rbm_experiment


class SymSubspaceTest(unittest.TestCase):
    def test_sequence_gen(self):
        g = make_sym_qutrit_clifford_group(get_sym_qutrit_clifford_xx_implementation)
        # For a smoke test of all the math, just make sure sequence generation doesn't
        # fail – an inverse being found means a lot of catastrophic events …
        generate_rbm_experiment(g, [2, 3, 100, 1001], 100)
