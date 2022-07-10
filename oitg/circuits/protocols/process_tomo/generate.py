from itertools import chain, product
from typing import List, Tuple
import numpy as np
from ...gate import Gate, GateSequence, remap_operands


def generate_process_tomography_sequences(target: GateSequence,
                                          num_qubits: int) -> List[GateSequence]:
    """Return a list of gate sequences to perform process tomography on the given target
    sequence.

    The system is prepared in a tensor product of single-qubit Pauli-operator
    eigenstates before applying the target sequence and measuring the expectation value
    of a tensor product of Pauli operators. 
    See ``generate_process_tomography_fiducial_pairs()``.

    :param target: The gate sequence to perform tomography on.
    :param num_qubits: The number of qubits making up the Hilbert space of interest.
    """
    fiducial_pairs = generate_process_tomography_fiducial_pairs(num_qubits)
    return wrap_target_in_process_tomography_fiducials(target, fiducial_pairs)


def wrap_target_in_process_tomography_fiducials(
        target: GateSequence,
        fiducial_pairs: List[Tuple[GateSequence, GateSequence]]) -> List[GateSequence]:
    """Return a list of gate sequences to perform process tomography on the given target
    sequence.

    The system is prepared in a tensor product of single-qubit Pauli-operator
    eigenstates before applying the target sequence and measuring the expectation value
    of a tensor product of Pauli operators. 

    :param target: The gate sequence to perform tomography on.
    :param fiducial_pairs: See ``generate_process_tomography_fiducial_pairs()``
    """
    return [tuple(chain(prep, target, measure)) for prep, measure in fiducial_pairs]


def generate_process_tomography_fiducial_pairs(
        num_qubits: int) -> List[Tuple[GateSequence, GateSequence]]:
    """Return a list of tuples of gate sequences implementing the preparation and 
    measurement for process tomography.

    For state preparation, all six Pauli eigenstates are created (i.e. ±x, ±y, ±z). Even
    though this yields an over-complete set of input states (:math:`6^n` instead of the
    :math:`4^n` required ones), this is a small price to pay for the resulting
    symmetry – even for two-qubit gates this only just more than doubles (:math:`9 / 4`)
    the number of sequences, so we can always just take fewer shots per sequence to
    compensate.

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

    def fiducial_pairs(locals):
        return [
            tuple(
                chain.from_iterable(
                    remap_operands(seq, {0: i}) for (i, seq) in enumerate(seqs)))
            for seqs in product(locals, repeat=num_qubits)
        ]

    return [(prep, measure[::-1]) for prep in fiducial_pairs(fiducials)
            for measure in fiducial_pairs(fiducials[:3])]