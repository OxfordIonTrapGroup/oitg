from typing import List, Sequence
from ...gate import GateSequence
from . import GSTSpec


def generate_std_gst_sequences(spec: GSTSpec,
                               max_len_exponent: int) -> List[GateSequence]:
    r"""Return a list of sequences to run to perform gate set tomography according to
    the given spec, with standard power-of-two sequence lengths up to (approximately)
    :math:`2^{\mathrm{max\_len\_exponent}}`.

    See :meth:`generate_gst_sequences`.

    :param spec: The fiducials and germs to use.
    :param max_len_exponent: The number of long-sequence refinement steps.
    """
    return generate_gst_sequences(spec, (2**i for i in range(max_len_exponent + 1)))


def generate_gst_sequences(spec: GSTSpec,
                           target_lens: Sequence[int]) -> List[GateSequence]:
    r"""Return a list of sequences to run to perform gate set tomography according to
    the given spec, with number of gates per sequence limited to the given lengths.

    Matches pyGSTi's default method of truncating sequences, where germs are only
    repeated in whole (rounding down target lengths divided by germ lengths), and
    fiducial lengths do not count.

    :param spec: The fiducials and germs to use.
    :param target_lens: The target sequence lengths.

    :return: A concatenated list of gate sequences to run (de-duplicated).
    """
    # Collect sequences in set for deduplication.
    seqs = set()
    for target_len in target_lens:
        for prep in spec.prep_fiducials:
            for meas in spec.meas_fiducials:
                for germ in spec.germs:
                    num_repeats = target_len // len(germ)
                    seqs.add(prep + germ * num_repeats + meas)
    return list(seqs)
