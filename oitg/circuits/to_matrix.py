"""Tools for converting common gates to unitary matrices."""

import numpy as np
from typing import Union
from .gate import Gate, GateGenerator, collect_operands

PAULI_OPERATORS = [[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]],
                   [[1, 0], [0, -1]]]


def rxy(phase, amount):
    x = np.array(PAULI_OPERATORS[1])
    y = np.array(PAULI_OPERATORS[2])
    return (np.cos(amount / 2) * np.eye(2) - 1j * np.sin(amount / 2) *
            (np.cos(phase) * x + np.sin(phase) * y))


LOCAL_MATRICES = {
    "rx":
    lambda amount: [[np.cos(amount / 2), -1j * np.sin(amount / 2)],
                    [-1j * np.sin(amount / 2),
                     np.cos(amount / 2)]],
    "ry":
    lambda amount: [[np.cos(amount / 2), -np.sin(amount / 2)],
                    [np.sin(amount / 2), np.cos(amount / 2)]],
    "rz":
    lambda amount: [[np.exp(-1j * amount / 2), 0], [0, np.exp(1j * amount / 2)]],
    "rxy":
    rxy,
    "h":
    lambda: np.array([[1, 1], [1, -1]]) / np.sqrt(2),
    "cx":
    lambda: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
    "cz":
    lambda: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]],
    "barrier":
    None,
}


def local_matrix(gate: Gate) -> np.ndarray:
    r"""Return the unitary matrix that describes ``gate`` on its operands only.

    The resulting matrix is the same no matter which target qubit(s) the gate operates
    on; the first copy of :math:`\mathbb{C}^2` corresponds to the first operand, etc.

    See :meth:`single_gate_matrix` for creating the unitary that acts on the given
    operands in a larger system.
    """
    try:
        mat_fun = LOCAL_MATRICES[gate.kind]
        if mat_fun is None:
            # No-op.
            return None
    except KeyError:
        raise ValueError("Unsupported gate of kind '{}'".format(gate.kind))
    return np.array(mat_fun(*gate.parameters))


def _num_qubits_from_state_dimension(dim):
    return dim.bit_length() - 1


def single_gate_matrix(gate: Gate, num_qubits: int) -> np.ndarray:
    """Return the unitary matrix that describes the action of the given gate.

    :param num_qubits: The number of qubits of the target Hilbert space.

    :return: A unitary matrix acting on the `2^num_qubits`-dimensional Hilbert space.
    """
    u_local = local_matrix(gate)

    if u_local is not None:
        num_unitary_targets = len(gate.operands)
        assert num_unitary_targets == _num_qubits_from_state_dimension(u_local.shape[0])
    else:
        # No-op (barrier, ...).
        num_unitary_targets = 1
        u_local = np.eye(2)

    # Construct global unitary by first tensoring with enough copies of the
    # identity and then permuting the bases.
    u_permuted = u_local
    for _ in range(num_qubits - num_unitary_targets):
        u_permuted = np.kron(u_permuted, np.eye(2))

    system_perm = np.array(range(num_qubits), dtype=int)
    for i, o in enumerate(gate.operands):
        system_perm[i] = o
        system_perm[o] = i

    # Shuffle around the axis indices. I wish numpy had a built-in method for this,
    # but just writing down the nested form manually isn't too bad for only qubits.
    # Dragging in qutip just for this seems wrong too.
    u = u_permuted.reshape([2, 2] * num_qubits)
    u = np.transpose(u, axes=np.concatenate((system_perm, system_perm + num_qubits)))
    return np.array(u.reshape(u_permuted.shape))


def gate_sequence_matrix(gates: GateGenerator,
                         num_qubits: Union[int, None] = None) -> np.ndarray:
    """Return the unitary matrix that describes the action of the given gate sequence.

    :param num_qubits: The number of qubits of the target Hilbert space. If ``None``,
        the number of qubits is inferred from the gate sequence (i.e., from the largest
        qubit index used).

    :return: A unitary matrix acting on the `2^num_qubits`-dimensional Hilbert space.
    """
    if num_qubits is None:
        num_qubits = max(collect_operands(gates)) + 1
    u = np.eye(2**num_qubits, dtype=np.complex64)
    for g in gates:
        u = single_gate_matrix(g, num_qubits) @ u
    return u


def apply_gate_sequence(gates: GateGenerator, initial_state: np.ndarray) -> np.ndarray:
    """Apply a gate sequence to the given initial state and return the resulting state
    vector.
    """
    num_qubits = _num_qubits_from_state_dimension(initial_state.shape[0])
    state = np.array(initial_state)
    for g in gates:
        state = single_gate_matrix(g, num_qubits) @ state
    return state
