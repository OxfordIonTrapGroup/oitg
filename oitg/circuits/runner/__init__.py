"""Common interface for gate sequence execution.

Many experimental protocols consist of acquiring outcome statistics for a number of gate
sequences (for instance those implemented in :mod:`oitg.protocols`). Traditionally, a
considerable fraction of the time spent on implementing these protocols in one's
experiment comes from having to write code to convert between different data formats
for specifying gate sequences and results, fetching result data from result files, etc.
This module addresses this by specifying a common interface to sequence runners, that
is, code that acquires outcome data for a list of given gate sequences.

This interface is given by the :class:`SequenceRunner` abstract base class, and the
:class:`SequenceRunnerOptions` structure, which implementations of the former should use
to accept settings giving the details of how (often) to execute the sequences (in place
of, say, an unwieldly long list of constructor arguments).

There are two main aspects of acquiring experimental data that this interface addresses
beyond just running sequentially through a long list of gate strings. The first is the
fact that experiments will tend to run as kernels on an ARTIQ core device. Kernels are
slow to start, and so is, to a lesser degree, any subsequent network communication.
Hence, it is advisable to compile/run more than one gate sequence at once. On the other
hand, core device resources are limited, so trying to e.g. keep DMA recordings for
thousands of long gate sequences in memory at the same time would be a bad idea. The
natural solution to this is to process the total list of gate sequences in **chunks** of
appropriate lengths. The chunk size is configurable, as it depends on the particulars of
the target system (keeping them about 1 s in wall clock duration might be a good
starting point). The exact interpretation will depend on the specifics of the sequence
runner implementation, but a common design would be to have a long-running kernel on the
core device, which fetches sequences from the host in slices of the configured chunk
length, acquires the data for them, and reports the results back to the host afterwards
(while fetching the next chunk).

The second aspect are **repeats** and **randomisation**. Typical experiments would
acquire a few hundred or thousand single-shot measurements per sequence, and might
comprise thousands of total sequences. For instance, a multi-qubit tomography experiment
might run for an hour or more. On these time scales, slow changes in the laboratory
environment, such as thermal drifts (with their typical time scale of ~10 min), are very
noticeable. To avoid any systematic shifts in the resulting data, a useful strategy is
to acquire the data points in random order to wash out any correlations. Randomly
permuting the sequences run shot-by-shot would be ideal, but possibly very expensive, as
mentioned above. A possible compromise is to a) acquire several shots per sequence at a
time, and/or b) choose at random only within each chunk, where precomputed sequences can
be quickly accessed.

The appropriate tradeoffs for both these aspects can be chosen by appropriately
configuring the  :class:`SequenceRunnerOptions`: The configurable chunk size allows to
limit the amount of resources needed on the core device (e.g. DMA sequences, size of
result arrays). The three (conceptual) nested loops (global repeats -> [chunking] ->
repeats per chunk -> shots per repeat) allow trading off randomisation quality vs.
performance: If switching between different sequences in the same chunk is cheap, choose
few shots per repeat, but many repeats per chunk. If you are concerned about slow
drifts, use fewer total shots per chunk, but more global repeats.

See e.g. ``oxart.circuits.runner`` for actual sequence runner implementations using
ARTIQ.
"""

from itertools import chain
from numpy import ndarray
from typing import Callable, Iterable, Union
from ..gate import GateGenerator, GateSequence
from ..qasm import gate_to_qasm


# TODO (Python 3.7+): Convert this into a data class.
class SequenceRunnerOptions:
    r"""Specifies common :class:`SequenceRunner` options.

    See the :mod:`module <oitg.circuits.runner>`\ -level docstring for background on
    the concepts.

    If you do not want to support the whole interface immediately while bringing up a
    new runner implementation (e.g. no per-chunk repeats or randomisation), consider
    emitting a warning and, in the case of repeats, folding the factor into other
    multipliers.

    :param num_global_repeats: The number of times to run through the list of sequences
        given.
    :param randomise_globally: Whether to randomise the order of sequences between
        global repeats (so that they are divided into different chunks, etc.).
    :param chunk_size: Number of sequences to execute at once (typically in one RPC
        cycle to the core device).
    :param num_repeats_per_chunk: The number of times to cycle through all sequences
        within each chunk.
    :param num_shots_per_repeat: The number of shots (single measurements) to acquire
        for each sequence per repeat.
    :param randomise_per_repeat: Whether to randomise the order of sequences within each
        repeat.
    """

    def __init__(self,
                 num_global_repeats: int = 1,
                 randomise_globally: bool = True,
                 chunk_size: int = 1,
                 num_repeats_per_chunk: int = 1,
                 num_shots_per_repeat: int = 100,
                 randomise_per_repeat: bool = True):
        self.num_global_repeats = num_global_repeats
        self.randomise_globally = randomise_globally
        self.chunk_size = chunk_size
        self.num_repeats_per_chunk = num_repeats_per_chunk
        self.num_shots_per_repeat = num_shots_per_repeat
        self.randomise_per_repeat = randomise_per_repeat


class SequenceRunner:
    r"""Executes a number of gate sequences to gather outcome statistics.

    See the :mod:`module <oitg.circuits.runner>`\ -level docstring for details.
    """

    def run_sequences(self,
                      sequences: Iterable[GateSequence],
                      num_qubits: Union[None, int] = None,
                      progress_callback: Callable[[ndarray, ndarray], None] = None,
                      progress_callback_interval: float = 5.0,
                      dataset_prefix: Union[None, str] = "data.circuits."):
        r"""Runs the given sequences and returns result statistics.

        :param sequences: The gate sequences to execute.
        :param num_qubits: The number of qubits in the circuit; used for readout. If not
            given, this will be inferred from the largest operand index used in the
            circuit.
        :param progress_callback: An optional callback to be invoked periodically
            throughout data acquisition, e.g. for updating some on-line plots/analysis.
            Two NumPy arrays are supplied as arguments, giving the sequence indices and
            outcomes acquired since the last time the callback was invoked (see return
            value).
        :param progress_callback_interval: The interval between invocations of the
            progress callback, in seconds.
        :param dataset_prefix: Prefix for dataset keys to write executed sequences and
            their results to. If ``None``, results are not written to datasets. Sequence
            runner implementations not embedded into an ARTIQ experiment can ignore
            this.

        :return: A tuple ``(run_order, outcomes)`` of NumPy arrays. ``run_order`` lists
            the order in which the sequences were executed (possibly with repeats) by
            their respective index in the passed collection. ``outcomes`` lists the
            number of occurrences of the respective measurement outcomes as a
            two-dimensional array, where the first index matches ``run_order``, and the
            second specifies the outcome in canonical binary order (i.e. corresponding
            to projection onto :math:`\left|0\ldots00\right>, \left|0\ldots01\right>,
            \ldots, \left|1\ldots11\right>`).
        """
        raise NotImplementedError


def stringify_gate_sequence(seq: GateGenerator):
    """Return the string used to represent ``seq`` in the result datasets."""
    return ";".join(chain.from_iterable(gate_to_qasm(g) for g in seq))
