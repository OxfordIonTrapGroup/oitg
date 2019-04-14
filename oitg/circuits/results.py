"""Common helpers for analysing experiment results.

Also see :mod:`oitg.results` for finding and loading HDF5 files.
"""

import numpy as np
from .gate import GateSequence
from .qasm import parse_gate_sequence_string
from typing import Any, Dict, List


def collect_outcomes(sequences: List[GateSequence], run_order: List[int],
                     outcomes: List[np.ndarray]):
    """Total up the number of observations per outcome for gate sequence runner
    experiments.

    This function works on the arrays passed to/returned by :mod:`oitg.circuits.runner`
    ``run_sequence()`` implementations; see :meth:`collect_outcomes_from_datasets` for
    processing the resulting datasets.

    :param sequences: A list of all sequences run.
    :param run_order: A list giving the index in ``sequences`` for each result.
    :param outcomes: A list of the same length as ``run_order``, giving the number of
        occurrences of each outcome for each respective gate sequence.

    :return: A dictionary mapping gate sequences to arrays giving the number of times
        each measurement outcome was observed after running them. Array elements
        represent the different qubit measurement outcomes in canonical order (i.e.
        `000`, `001`, `010`, …).
    """
    result = {}
    for sequence_idx, counts in zip(run_order, outcomes):
        total_counts = result.setdefault(sequences[sequence_idx], np.zeros_like(counts))
        total_counts += counts
    return result


def collect_outcomes_from_datasets(datasets: Dict[str, Any],
                                   prefix: str = "data.circuits."
                                   ) -> Dict[GateSequence, np.ndarray]:
    """Total up the number of observations per outcome for gate sequence runner
    experiments.

    This function works on the datasets written by :mod:`oitg.circuits.runner`
    implementations. See :meth:`collect_outcomes` for an implementation that directly
    operates on lists, and :mod:`oitg.results` for extracting datasets from an ARTIQ
    results file.

    :param datasets: A dictionary containing all the datasets written by the experiment.
    :param prefix: The dataset key prefix the circuit results were saved under.

    :return: A dictionary mapping gate sequences to arrays giving the number of times
        each measurement outcome was observed after running them; see
        :meth:`collect_outcomes`.
    """

    def d(key):
        return datasets[prefix + key]

    return collect_outcomes([parse_gate_sequence_string(s) for s in d("sequences")],
                            d("run_order"), d("outcomes"))
