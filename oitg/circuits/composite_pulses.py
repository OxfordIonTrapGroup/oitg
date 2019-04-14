"""Composite pulse implementations for replacing gates with sequences that are logically
equivalent, but have different behaviour under imperfections.

.. rubric:: References
.. [BIKN13] Bando, M., Ichikawa, T., Kondo, Y. & Nakahara, M.
    Concatenated composite pulses compensating simultaneous systematic errors.
    J. Phys. Soc. Jpn. 82, 014004 (2013).
.. [KGK+14] Kabytayev, C. et al. Robustness of composite pulses to time-dependent
    control noise. Phys. Rev. A 90, 012316 (2014).
.. [Wimp94] Wimperis, Stephen. Broadband, Narrowband, and Passband Composite Pulses for
    Use in Advanced NMR Experiments. Journal of Magnetic Resonance 109, 221–231 (1994).
"""

import numpy as np
from typing import Callable, Iterable
from .gate import Gate, GateGenerator


class UnsupportedGate(ValueError):
    """Raised if a given gate cannot be expanded in the requested form."""
    pass


def to_rxy(gate: Gate) -> Gate:
    """Canonicalise all single-qubit rotations in the xy-plane to ``rxy`` gates.

    :return: A ``rxy`` :class:`.Gate` with positive rotation amount.
    """
    if gate.kind == "rxy":
        return gate
    phase = {"rx": 0.0, "ry": np.pi / 2}.get(gate.kind, None)
    if phase is None:
        raise UnsupportedGate

    amount = gate.parameters[0]
    if amount < 0:
        phase += np.pi
        amount *= -1

    return Gate("rxy", (phase, amount), gate.operands)


def bb1(gate: Gate, symmetric: bool = False) -> GateGenerator:
    """Generate implementation of the given single-qubit rotation using a BB1 composite
    pulse.

    BB1, as per [Wimp94]_, is a broadband amplitude noise suppression sequence,
    cancelling error terms up to fourth order in amplitude offset.

    :param gate: The gate to implement.
    :param symmetric: Whether to implement the gate symmetrically using 5 pulses (2
        half-rotations around the 3-pulse BB1 identity sequence), or asymmetrically
        using 4 pulses. The latter has slightly nicer behaviour under detuning errors.
    """
    gate = to_rxy(gate)
    phase, amount = gate.parameters
    phi = np.arccos(-amount / (4 * np.pi))

    def xy(phase, amount):
        return Gate("rxy", (phase, amount), gate.operands)

    first_amount = 0.5 * amount if symmetric else amount
    yield xy(phase, first_amount)
    yield xy(phase + phi, np.pi)
    yield xy(phase + 3 * phi, 2 * np.pi)
    yield xy(phase + phi, np.pi)
    if symmetric:
        yield xy(phase, first_amount)


def reduced_c_in_sk(gate: Gate) -> GateGenerator:
    """Generate implementation of the given single-qubit rotation using a reduced
    CORPSE/SK1 concatenated composite pulse.

    Reduced CinSK, as described in [BIKN13]_ (and among those analysed in [KGK+14]_) is
    robust against both pulse amplitude and detuning errors. Note that the CORPSE part
    (that gives detuning robustness) requires non-multiple-of-π/2 gate rotation amounts.
    """
    gate = to_rxy(gate)
    phase, amount = gate.parameters

    k = np.arcsin(np.sin(amount / 2) / 2)
    phi = np.arccos(-amount / (4 * np.pi))

    def xy(phase, amount):
        return Gate("rxy", (phase, amount), gate.operands)

    yield xy(phase, 2 * np.pi + amount / 2 - k)
    yield xy(phase + np.pi, 2 * (np.pi - k))
    yield xy(phase, amount / 2 - k)
    yield xy(phase - phi, 2 * np.pi)
    yield xy(phase + phi, 2 * np.pi)


def expand_using(method: Callable,
                 gates: GateGenerator,
                 ignore_kinds: Iterable[str] = [],
                 ignore_unsupported_gates: bool = True,
                 insert_barriers: bool = True) -> GateGenerator:
    """Expand all gates in the given sequence using composite pulses.

    :param method: A callable implementing the chosen composite pulse type
        (e.g. :meth:`bb1`).
    :param ignore_kinds: A set of gate kinds not to attempt to expand.
    :param ignore_unsupported_gates: If ``True``, silently pass over unsupported gates
        without expanding them.
    :param insert_barriers: Insert a barrier after each composite pulse.
    """
    ignore_kinds = set(ignore_kinds)
    for gate in gates:
        try:
            if gate.kind in ignore_kinds:
                yield gate
                continue
            yield from method(gate)
            if insert_barriers:
                yield Gate("barrier", (), gate.operands)
        except UnsupportedGate:
            if not ignore_unsupported_gates:
                raise
            yield gate
