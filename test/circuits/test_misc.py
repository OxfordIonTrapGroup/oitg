import numpy as np
import unittest
from numpy.testing import assert_almost_equal

from oitg.circuits.clifford import *
from oitg.circuits.composite_pulses import *
from oitg.circuits.gate import *
from oitg.circuits.qasm import *
from oitg.circuits.to_matrix import *


class QasmTest(unittest.TestCase):
    def test_gate(self):
        self.assertEqual(list(gate_to_qasm(Gate("rx", (0.4, ), (1, )))),
                         ["rx(0.4) q[1]"])
        self.assertEqual(list(gate_to_qasm(Gate("cz", (), (
            0,
            1,
        )))), ["cz q[0], q[1]"])

    def test_gate_experiment(self):
        exp = [
            Gate("rx", (0.4, ), (2, )),
            Gate("cz", (), (0, 1)),
        ]
        self.assertEqual(list(gate_experiment_to_qasm(exp)), [
            "OPENQASM 2.0", "include \"qelib1.inc\"", "qreg q[3]", "creg r[3]",
            "rx(0.4) q[2]", "cz q[0], q[1]", "measure q -> r"
        ])


def is_inverse_up_to_phase(a, b):
    return np.isclose(a.shape[0], np.abs(np.trace(np.dot(a, b))), atol=1e-6)


class UnitaryTest(unittest.TestCase):
    def test_fill(self):
        assert_almost_equal(
            single_gate_matrix(Gate("rx", (np.pi, ), (0, )), 2),
            [[0, 0, -1j, 0], [0, 0, 0, -1j], [-1j, 0, 0, 0], [0, -1j, 0, 0]])
        assert_almost_equal(
            single_gate_matrix(Gate("rx", (np.pi, ), (1, )), 2),
            [[0, -1j, 0, 0], [-1j, 0, 0, 0], [0, 0, 0, -1j], [0, 0, -1j, 0]])

    def test_permute(self):
        assert_almost_equal(single_gate_matrix(Gate("cx", (), (0, 1)), 2),
                            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
        assert_almost_equal(single_gate_matrix(Gate("cx", (), (1, 0)), 2),
                            [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])

    def test_rxy(self):
        assert_almost_equal(local_matrix(Gate("rxy", (0, 0.1234), (0, ))),
                            local_matrix(Gate("rx", (0.1234, ), (0, ))))
        assert_almost_equal(local_matrix(Gate("rxy", (np.pi / 2, 0.1234), (0, ))),
                            local_matrix(Gate("ry", (0.1234, ), (0, ))))

    def test_gate_sequence(self):
        cz_seq = [
            Gate("h", (), (0, )),
            Gate("cz", (), (0, 1)),
            Gate("h", (), (0, )),
        ]
        cx_seq = [Gate("cx", (), (1, 0))]
        assert_almost_equal(gate_sequence_matrix(cz_seq), gate_sequence_matrix(cx_seq))


class CompositePulseTest(unittest.TestCase):
    def test_correctness(self):
        for method in bb1, reduced_c_in_sk:
            for _ in range(100):
                phase, amount = np.random.rand(2) * np.pi
                raw_gate = Gate("rxy", (phase, amount), (0, ))
                composite_gate = list(method(raw_gate))
                self.assertTrue(
                    is_inverse_up_to_phase(gate_sequence_matrix(composite_gate),
                                           np.conj(local_matrix(raw_gate)).T))

    def test_expand_bb1(self):
        gates = [
            Gate("rx", (np.pi, ), (0, )),
            Gate("cz", (), (0, 1)),
            Gate("ry", (np.pi, ), (0, ))
        ]
        result = list(expand_using(bb1, gates))
        self.assertEqual(len(result), 11)
        result = list(expand_using(bb1, gates, insert_barriers=False))
        self.assertEqual(len(result), 9)
        expand_using(bb1, gates, ignore_unsupported_gates=False)
        self.assertRaises(
            UnsupportedGate, lambda: list(
                expand_using(bb1, gates, ignore_unsupported_gates=False)))
