"""Randomised benchmarking sequence generation.

In the same spirit as :mod:`oitg.circuits.clifford`, this implements pretty much the
most pedestrian scheme possible and just tracks the total unitary generated instead of
implementing stabilizer calculations.
"""

from typing import List, Iterable, Tuple
import numpy as np
from ...clifford import GateGroup
from ...gate import Gate, GateSequence
from ...to_matrix import gate_sequence_matrix, PAULI_OPERATORS


def generate_rbm_experiment(group: GateGroup,
                            sequence_lengths: Iterable[int],
                            randomisations_per_length: int,
                            pauli_randomize_last=True,
                            interleave_gate=None,
                            derive_shorter_by_truncation=False,
                            seed=None) -> List[Tuple[List[int], GateSequence, int]]:
    """
    :param sequence_lengths: List of sequence lengths to generate. For each length k, a
        number of sequences will be generated with k Clifford group elements (or
        `2 * k - 1` when interleaving a gate).
    :param randomisations_per_length: Number of random gate sequences for each given
        sequence length (twice that for interleaved benchmarking).
    :param pauli_randomize_last: If ``True``, the last Clifford element is chosen to
        invert the previous only gates up to a randomly selected Pauli group element,
        thus randomising the outcome between 0 and 1. This should always be used, as
        the analysis is susceptible to SPAM errors otherwise.
    :param interleave_gate: If not ``None``, the given gate is interleaved between each
        two Clifford gates in a second copy of every sequence (to implement interleaved
        benchmarking).
    :param derive_shorter_by_truncation: Derive shorter sequences by truncating longer
        sequences (and then appending the appropriate inverse gate) instead of
        generating fresh random sequences. This can be handy for debugging when
        initially working out phase relationships in the experiment.
    :param seed: Base seed for random number generator. If not provided, an
        unpredictable seed is used.
    :return: A list of tuples `(clifford_idxs, gates, expected_result)` giving for each
        experiment to run the corresponding list of Clifford elements (`-1` for the
        interleaved gate), the gate sequences, and the respective expected results
        (as the canonical integer representation of the binary string of results, where
        for each qubit 0 indicates the initial state, and 1 the one orthogonal to that).
    """

    rng = np.random.RandomState(seed=seed)

    sequence_lengths = np.sort(sequence_lengths)[::-1]
    if sequence_lengths[-1] < 2:
        raise ValueError("RBM sequences need to be at least 2 Clifford gates long")
    unfinished_sequences = []
    for length in sequence_lengths:
        for _rand in range(randomisations_per_length):
            unfinished_sequences.append(
                [rng.randint(group.num_elements()) for _ in range(length - 1)])
        if derive_shorter_by_truncation:
            break
    if derive_shorter_by_truncation:
        truncated_sequences = []
        for length in sequence_lengths[1:]:
            for seq in unfinished_sequences:
                truncated_sequences.append(seq[:(length - 1)])
        unfinished_sequences += truncated_sequences

    INTERLEAVE_IDX = -1
    if interleave_gate is not None:
        interleaved_sequences = []
        for seq in unfinished_sequences:
            interleaved_seq = []
            for gates in seq:
                interleaved_seq.append(gates)
                interleaved_seq.append(INTERLEAVE_IDX)
            interleaved_sequences.append(interleaved_seq)
        unfinished_sequences += interleaved_sequences

    finished_sequences_descs = []
    for clifford_idxs in unfinished_sequences:

        def to_gates(idx):
            if idx == INTERLEAVE_IDX:
                gates = [interleave_gate]
            else:
                gates = group.gates_for_idx(idx)
            return gates + [Gate("barrier", (), tuple(range(group.num_qubits)))]

        gates = sum(map(to_gates, clifford_idxs), [])

        # TODO: Use cached matrices for performance.
        matrix_up_to_last = gate_sequence_matrix(gates, num_qubits=group.num_qubits)
        if pauli_randomize_last:
            # "Mess up" the state tracking by appending a random Pauli per qubit, so
            # that the expected result is random.
            paulis = [
                PAULI_OPERATORS[rng.randint(len(PAULI_OPERATORS))]
                for _ in range(group.num_qubits)
            ]
            a = paulis[0]
            for b in paulis[1:]:
                a = np.kron(a, b)
        else:
            a = np.identity(matrix_up_to_last.shape[0])

        last_idx = group.find_inverse_idx(a @ matrix_up_to_last)
        clifford_idxs.append(last_idx)
        last_gates = to_gates(last_idx)
        gates += last_gates

        # Calculate final state and convert expected measurement result to binary
        # string.
        final_matrix = (gate_sequence_matrix(last_gates, num_qubits=group.num_qubits)
                        @ matrix_up_to_last)
        final_state = final_matrix[:, 0]
        result_idx = np.argmax(np.abs(final_state))
        assert np.isclose(abs(final_state[result_idx]), 1)

        finished_sequences_descs.append((clifford_idxs, tuple(gates), result_idx))

    return finished_sequences_descs


if __name__ == '__main__':
    # Demonstrate sequence generation.
    from contexttimer import Timer
    from ....cache import cache_to_pickle_file
    from ...clifford import *
    from ...visualisation import save_circuit_pdf

    @cache_to_pickle_file("clifford_2q_xzpm2_cz")
    def get_c2():
        print("Regenerating 2-qubit Clifford group...")
        with Timer() as t:
            c2 = make_clifford_group(2, get_clifford_2q_xzpm2_cz_implementation)
        print("...done ({:.3f} s).".format(t.elapsed))
        return c2

    lengths = [2, 4, 8, 16, 32, 64]
    num_randomisations = 3

    with Timer() as t:
        c1 = make_clifford_group(1, get_clifford_1q_xypm_implementation)
    print("Generated 1-qubit Clifford group in {} s".format(t.elapsed))
    with Timer() as t:
        result = generate_rbm_experiment(c1, lengths, num_randomisations)
    print("Generated {} 1-qubit sequences in {} s".format(
        len(lengths) * num_randomisations, t.elapsed))

    with Timer() as t:
        c2 = get_c2()
    print("Loaded 2-qubit Clifford group in {} s".format(t.elapsed))
    with Timer() as t:
        result = generate_rbm_experiment(c2, lengths, num_randomisations)
    print("Generated {} 2-qubit sequences in {} s".format(
        len(lengths) * num_randomisations, t.elapsed))

    save_circuit_pdf("rbm.pdf", result[1])
