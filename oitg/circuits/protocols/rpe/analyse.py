import numpy as np
from typing import Dict
from ...gate import GateSequence


def analyse(outcomes: Dict[GateSequence, np.ndarray]) -> float:
    r"""Analyse a Robust Phase Estimation data set and return the gate rotation angle
    estimate.

    Follows the algorithm described in [RKLM17]_.

    :param outcomes: A dictionary mapping gate sequences to number of observed outcomes,
        as returned by :meth:`oitg.results.collect_outcomes`.

    :return: The final estimate for the target gate rotation angle, in
        :math:`[0, 2\pi)`.
    """
    seqs = list(outcomes.keys())
    lengths = np.array([len(s) for s in seqs])
    idxs = np.argsort(lengths)

    target_seq = seqs[idxs[0]]

    # There will be two sequences of length 2; 2*target as well as target+pi_2. Find the
    # right one and extract the pi_2 sequence.
    for i in range(1, 3):
        target_and_pi_2_seq = seqs[idxs[i]]
        if target_and_pi_2_seq[:len(target_seq)] != target_seq:
            raise ValueError("Second shortest sequence should be the target gate string"
                             "plus the π/2 implementation")
        pi_2_seq = target_and_pi_2_seq[len(target_seq):]
        if pi_2_seq != target_seq:
            break
    else:
        raise ValueError("π/2 implementation not found")

    max_repeats, r = divmod(lengths[idxs[-2]], len(target_seq))
    if r != 0 or (max_repeats & (max_repeats - 1)) != 0:
        raise ValueError("Longest sequences should be power-of-two repeats of target")

    max_power_of_two = np.log2(max_repeats).astype(np.int64)

    estimate = 0.0
    for i in range(max_power_of_two + 1):
        current_len = 2**i

        cos_outcomes = outcomes[target_seq * current_len]
        xhat = cos_outcomes[0] / sum(cos_outcomes) - 1 / 2

        sin_outcomes = outcomes[target_seq * current_len + pi_2_seq]
        yhat = sin_outcomes[0] / sum(sin_outcomes) - 1 / 2

        next_estimate = -np.arctan2(yhat, xhat) / current_len

        # Shift estimated angle into permissible region.
        while next_estimate < estimate - np.pi / current_len:
            next_estimate += 2 * np.pi / current_len
        while next_estimate > estimate + np.pi / current_len:
            next_estimate -= 2 * np.pi / current_len
        estimate = next_estimate
    return estimate % (2 * np.pi)
