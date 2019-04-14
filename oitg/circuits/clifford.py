r"""Brute-force Clifford group helpers.

The Gottesmann—Knill theorem states that circuits consisting of gates from the
Clifford group :math:`\mathcal{C}_n = \operatorname{Aut}(\mathrm{Pauli}_n)` can be
efficiently simulated clasically, typically using some implementation of the stabiliser
formalism.

This module is emphatically *not* about that. Instead, it contains utilities for
directly manipulating elements of a group of unitaries as gate strings, without making
any use of its structure to make the calculations easier. This is only really practical
for the single- and two-qubit Clifford groups :math:`\mathcal{C}_1` and
:math:`\mathcal{C}_2`.

TODO (DPN): Instead of explicitly listing more than one Clifford group decomposition,
this should just directly integrate my decomposition search code and transparently use
:mod:`oitg.cache` for caching. In the meantime, just ask if you want to use a different
gate set.
"""

from enum import Enum, unique
from typing import Callable, Dict, List
import numpy as np
from .gate import Gate, GateGenerator, GateSequence, remap_operands
from .to_matrix import gate_sequence_matrix


class GateGroup:
    r"""A group (in the mathematical sense) of gates, where each element is represented
    by its index in some canonical order.

    :param num_qubits: The number of qubits the gates operate on.
    :param gate_sequences: The :class:`.GateSequence`\ s corresponding to each
        group element.
    :param matrices: The unitary matrices describing the action of each group element.
    :param inverse_idxs: The index of the inverse element for each group element, given
        as a map of canonical matrix key (see :meth:`to_canonical_matrix_key`) to
        element index.
    """

    def __init__(self, num_qubits: int, gate_sequences: List[GateSequence],
                 matrices: List[np.ndarray], inverse_idxs: Dict[bytearray, int]):
        self.num_qubits = num_qubits

        # Exhaustive list of gate sequences for all elements and their unitary matrices.
        self._gate_sequences = gate_sequences
        self._matrices = matrices
        assert len(self._matrices) == len(self._gate_sequences)

        # Maps canonical matrix (in `bytes` form) to the index of the inverse element.
        self._inverse_idxs = inverse_idxs
        assert len(self._inverse_idxs) == len(self._gate_sequences)

    def num_elements(self) -> int:
        """Return the total number of elements in the group."""
        return len(self._gate_sequences)

    def gates_for_idx(self, idx: int) -> GateSequence:
        """Return the gate sequence for the given element index."""
        return self._gate_sequences[idx]

    def matrix_for_idx(self, idx: int) -> np.ndarray:
        """Return the unitary matrix for the given element index."""
        return self._matrices[idx]

    def find_inverse_idx(self, matrix: np.ndarray) -> int:
        """Look up the index of the element that is the inverse of the given unitary
        matrix."""
        return self._inverse_idxs[to_canonical_matrix_key(matrix)]


def to_canonical_matrix(gate_matrix: np.ndarray) -> np.ndarray:
    """Convert the given gate matrix to a canonical form, which is exactly the same
    no matter the global phase or rounding errors.

    Note that this rounds to 4 decimal places for robustness, which is plenty for
    Clifford group calculations, but might not be sufficient for other applications.
    """
    # Divide by the phase of the first non-small element in the first row to normalise
    # out the global phase. This is pretty arbitrary (we just need a phase convention
    # that avoids dividing by zero), and rather silly looking, but works.
    u = gate_matrix.copy()
    i = 0
    while np.abs(u[0, i]) < 1e-6:
        i += 1
    phase = u[0, i] / np.abs(u[0, i])
    return np.array(np.around(u / phase, decimals=4) + 0.0, dtype='complex64')


def to_canonical_matrix_key(gate_matrix: np.ndarray) -> bytearray:
    """Return the canonical matrix corresponding to `gate_matrix` in a form suitable
    for use as a dictionary key."""
    # TODO: Could hash here directly for performance.
    key = to_canonical_matrix(gate_matrix).tobytes()
    return key


CliffordImpl = Callable[[int], GateGenerator]


def make_clifford_group(num_qubits: int, implementation: CliffordImpl) -> GateGroup:
    """Construct a :class:`GateGroup` instance which enumerates all elements of the
    ``num_qubit``-qubit Clifford group, with elements decomposed according to the given
    ``implementation``.
    """

    if num_qubits == 1:
        num_elements = 24
    elif num_qubits == 2:
        num_elements = 11520
    else:
        raise ValueError("Unsupported number of qubits: {}".format(num_qubits))

    gate_sequences = []
    matrices = []
    inverse_idxs = {}
    for idx in range(num_elements):
        gates = list(implementation(idx))
        gate_sequences.append(gates)
        u_gate = gate_sequence_matrix(gates, num_qubits)
        matrices.append(u_gate)
        inverse_idxs[to_canonical_matrix_key(np.linalg.inv(u_gate))] = idx

    return GateGroup(num_qubits, gate_sequences, matrices, inverse_idxs)


# Indices of various well-known single-qubit Clifford group elements.
# TODO: Look up by unitary for dry-ness.
C1_HADAMARD_IDX = 9
C1_Y_PI_BY_2_IDX = 11
C1_X_PI_IDX = 23

_clifford_1q_xypm_implementations = [
    [],
    [0, 3, 2],
    [0, 0, 1, 1],
    [0, 1, 2],
    [0, 0],
    [0, 3, 0],
    [1, 1],
    [0, 1, 0],
    [1, 0],
    [0, 0, 3],
    [1, 2],
    [1],
    [3, 2],
    [0, 0, 1],
    [3, 0],
    [3],
    [2, 3],
    [2],
    [2, 1],
    [2, 1, 1],
    [0, 3],
    [0, 1, 1],
    [0, 1],
    [0],
]


def get_clifford_1q_xypm_implementation(idx: int) -> GateGenerator:
    """Return an implementation of the 1-qubit Clifford group element with the given
    index as ±π/2 rotations about the x and y axes.

    :return: A minimal-length gate string implementing the given Clifford.
    """
    for e in _clifford_1q_xypm_implementations[idx]:
        if e == 0:
            yield Gate("rx", (np.pi / 2, ), (0, ))
        elif e == 1:
            yield Gate("ry", (np.pi / 2, ), (0, ))
        elif e == 2:
            yield Gate("rx", (-np.pi / 2, ), (0, ))
        elif e == 3:
            yield Gate("ry", (-np.pi / 2, ), (0, ))
        else:
            assert False


_clifford_1q_xzpm2_implementations = [
    [],
    [1],
    [5],
    [3],
    [4],
    [1, 4],
    [4, 5],
    [3, 4],
    [0, 1],
    [0, 1, 0],
    [2, 3],
    [0, 1, 2],
    [2, 1],
    [0, 3, 0],
    [0, 3],
    [0, 3, 2],
    [3, 2],
    [2],
    [1, 2],
    [0, 5],
    [1, 0],
    [2, 5],
    [3, 0],
    [0],
]


def get_clifford_1q_xzpm2_implementation(idx: int) -> GateGenerator:
    """Return an implementation of the 1-qubit Clifford group element with the given
    index as ±π/2 and π rotations about the x and z axes.

    :return: A minimal-length gate string implementing the given Clifford.
    """
    for e in _clifford_1q_xzpm2_implementations[idx]:
        if e == 0:
            yield Gate("rx", (np.pi / 2, ), (0, ))
        elif e == 1:
            yield Gate("rz", (np.pi / 2, ), (0, ))
        elif e == 2:
            yield Gate("rx", (-np.pi / 2, ), (0, ))
        elif e == 3:
            yield Gate("rz", (-np.pi / 2, ), (0, ))
        elif e == 4:
            yield Gate("rx", (np.pi, ), (0, ))
        elif e == 5:
            yield Gate("rz", (np.pi, ), (0, ))
        else:
            assert False


@unique
class EntanglingGate(Enum):
    """Specifies the equivalence class (up to local single-qubit Cliffords) of a
    non-trivial two-qubit Clifford operation.
    """

    #:
    cz_like = 0

    #:
    iswap_like = 1

    #:
    swap_like = 2


def get_clifford_2q_implementation(
        idx: int, clifford_1q_impl: CliffordImpl,
        entangling_gates_impl: Callable[[EntanglingGate, CliffordImpl], GateGenerator]
) -> GateGenerator:
    """Generate an implementation of the 2-qubit Clifford group element with the given
    index.

    :param idx: The element index.
    :param clifford_1q_impl: The single-qubit Clifford gate implementation to use
        (acting on qubit 0).
    :param entangling_gates_impl: The entangling gate implementation to use.
    """

    # TODO: This implementation correctly enumerates the two-qubit Clifford group for
    # any valid entangling_gates_impl, but the element indices aren't stable across
    # different choices of entangling gates.

    def local_idx_to_gates(i):
        assert 0 <= i < 24**2
        c0, c1 = divmod(i, 24)
        yield from clifford_1q_impl(c0)
        yield from remap_operands(clifford_1q_impl(c1), {0: 1})

    axis_permute_clifford_1q_elems = [0, 12, 22]

    def perm_idx_to_gates(i):
        assert 0 <= i < 3**2
        p0, p1 = divmod(i, 3)
        yield from clifford_1q_impl(axis_permute_clifford_1q_elems[p0])
        yield from remap_operands(clifford_1q_impl(axis_permute_clifford_1q_elems[p1]),
                                  {0: 1})

    i = idx
    NUM_LOCAL = 24**2
    if i < NUM_LOCAL:
        # Local operation class.
        yield from local_idx_to_gates(i)
        return
    i -= NUM_LOCAL

    NUM_CZ_LIKE = 24**2 * 3**2
    if i < NUM_CZ_LIKE:
        # CZ-like class.
        local_idx, perm_idx = divmod(i, 3**2)
        yield from local_idx_to_gates(local_idx)
        yield from entangling_gates_impl(EntanglingGate.cz_like, clifford_1q_impl)
        yield from perm_idx_to_gates(perm_idx)
        return
    i -= NUM_CZ_LIKE

    NUM_ISWAP_LIKE = 24**2 * 3**2
    if i < NUM_ISWAP_LIKE:
        # iSWAP-like class.
        local_idx, perm_idx = divmod(i, 3**2)
        yield from local_idx_to_gates(local_idx)
        yield from entangling_gates_impl(EntanglingGate.iswap_like, clifford_1q_impl)
        yield from perm_idx_to_gates(perm_idx)
        return
    i -= NUM_ISWAP_LIKE

    NUM_SWAP_LIKE = 24**2
    if i < NUM_SWAP_LIKE:
        # Local operation class.
        yield from local_idx_to_gates(i)
        yield from entangling_gates_impl(EntanglingGate.swap_like, clifford_1q_impl)
        return

    raise ValueError("Invalid 2-qubit Clifford gate index: {}".format(idx))


def get_cz_entangling_gate_implementation(kind: EntanglingGate,
                                          clifford_1q_impl_0: CliffordImpl
                                          ) -> GateGenerator:
    """Generate an implementation of the given entangling gate category using CZ gates
    and the given single-qubit gate implementation.
    """

    def clifford_1q_impl_1(idx):
        return remap_operands(clifford_1q_impl_0(idx), {0: 1})

    if kind == EntanglingGate.cz_like:
        yield Gate("cz", (), (0, 1))
    elif kind == EntanglingGate.iswap_like:
        yield Gate("cz", (), (0, 1))
        yield from clifford_1q_impl_0(C1_X_PI_IDX)
        yield from clifford_1q_impl_1(C1_X_PI_IDX)
        yield Gate("cz", (), (0, 1))
    elif kind == EntanglingGate.swap_like:
        yield Gate("cz", (), (0, 1))
        yield from clifford_1q_impl_0(C1_X_PI_IDX)
        yield from clifford_1q_impl_1(C1_X_PI_IDX)
        yield Gate("cz", (), (0, 1))
        yield from clifford_1q_impl_0(C1_X_PI_IDX)
        yield from clifford_1q_impl_1(C1_X_PI_IDX)
        yield Gate("cz", (), (0, 1))
    else:
        assert False


def get_clifford_2q_xypm_cz_implementation(idx: int) -> GateGenerator:
    """Generate an implementation of the given 2-qubit Clifford group element using CZ
    gates and local single-qubit ±π/2 rotations about the x and y axes.
    """
    return get_clifford_2q_implementation(idx, get_clifford_1q_xypm_implementation,
                                          get_cz_entangling_gate_implementation)


def get_clifford_2q_xzpm2_cz_implementation(idx: int) -> GateGenerator:
    """Generate an implementation of the given 2-qubit Clifford group element using CZ
    gates and local single-qubit ±π/2 and π rotations about the x and z axes.
    """
    return get_clifford_2q_implementation(idx, get_clifford_1q_xzpm2_implementation,
                                          get_cz_entangling_gate_implementation)
