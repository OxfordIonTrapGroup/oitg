from itertools import chain, product
from typing import List
import numpy as np
from ...gate import Gate, GateSequence, remap_operands


def generate_process_tomography_sequences(target: GateSequence,
                                          num_qubits: int) -> List[GateSequence]:
    """Return a list of gate sequences to perform process tomography on the given target
    sequence.

    The system is prepared in a tensor product of single-qubit Pauli-operator
    eigenstates before applying the target sequence and measuring the expectation value
    of a tensor product of Pauli operators.

    For state preparation, all six Pauli eigenstates are created (i.e. ±x, ±y, ±z). Even
    though this yields an over-complete set of input states (:math:`6^n` instead of the
    :math:`4^n` required ones), this is a small price to pay for the resulting
    symmetry – even for two-qubit gates this only just more than doubles (:math:`9 / 4`)
    the number of sequences, so we can always just take fewer shots per sequence to
    compensate.

    :param target: The gate sequence to perform tomography on.
    :param num_qubits: The number of qubits making up the Hilbert space of interest.
    """

    fiducials = [
        (Gate("ry", (np.pi / 2, ), (0, )), ),  # +x
        (Gate("rx", (-np.pi / 2, ), (0, )), ),  # +y
        (),  # +z
        (Gate("ry", (-np.pi / 2, ), (0, )), ),  # -x
        (Gate("rx", (np.pi / 2, ), (0, )), ),  # -y
        (Gate("rx", (np.pi, ), (0, )), ),  # -z
    ]

    def product_fiducials(locals):
        return [
            tuple(
                chain.from_iterable(
                    remap_operands(seq, {0: i}) for (i, seq) in enumerate(seqs)))
            for seqs in product(locals, repeat=num_qubits)
        ]

    return [
        tuple(chain(prep, target, measure[::-1]))
        for prep in product_fiducials(fiducials)
        for measure in product_fiducials(fiducials[:3])
    ]
