from typing import Iterable
from ...gate import GateSequence


def generate_rpe_sequences(target: GateSequence, pi_2: GateSequence,
                           max_len_exponent: int) -> Iterable[GateSequence]:
    r"""Generate sequences for robust phase estimation, up to a total sequence length
    of :math:`2^{\mathrm{max\_len\_exponent}} + 1`.

    :param target: A gate sequence implementing the target rotation.
    :param pi_2: A gate sequence implementing a π/2 pulse along the same axis as the
        target gate. (Can be same as `target` for measuring a π/2 pulse).
    :param max_len_exponent: The number of power-of-two refinement steps.

    :return: A generator yielding all the gate sequences to run.
    """
    for repeats in (2**i for i in range(max_len_exponent + 1)):
        yield target * repeats
        yield target * repeats + pi_2
