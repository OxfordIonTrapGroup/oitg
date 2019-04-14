r"""
Defines representations of quantum gates (and sequences of them), and a few basic
operations for manipulating them.

A :class:`Gate` is a named tuple
``(kind: str, parameters: Tuple[float, ...], operands: Tuple[int, ...])``.

``kind`` identifies the type of unitary (e.g. ``"rx"``, ``"cz"``). ``parameters`` is a
list of float values parametrising the chosen gate (e.g. rotation angles). ``operands``
is a list of integer indices denoting the target qubits.

This is deliberately just a plain piece of data to make gates directly representable in
ARTIQ kernels as well (with the tuples expressed as ``TList``\ s). Even apart from that,
coming up with an extensible design for representing both values and operations isn't
trivial in the first place (cf. expression problem). This sort of weakly typed design
(with gate types represented as arbitrary strings) isn't worse than many alternatives
anyway.

Using tuples not only represents the data semantics better than lists (parameters and
operands are plain collections with value semantics), but also has the advantage of
keeping Gate hashable (so it can be used in sets/dictionaries/…).

If more complex operations (e.g. decomposition into elementary gates, optimisations,
etc.) are desired in the future, we might want to directly integrate a QC library (Cirq,
Qiskit, pyQuil, …) into our code instead, though.

A :class:`GateSequence` is a tuple of :class:`Gate`\ s that represents the coherent
portion of a circuit on one or more qubits. Tuples are used instead of lists as most of
the time, gate sequences are not changed once built up, and tuples (with their value
semantics) are hashable.

A :class:`GateGenerator` produces a :class:`GateSequence` gate by gate through an
``Iterable`` interface. Note that this will in general *not* be a list that support
indexing, but for instance an iterator. Use e.g. ``tuple(…)`` (or similar) to convert
the result to a :class:`GateSequence`.
"""

from typing import Dict, Iterable, NamedTuple, Set, Tuple

# TODO: Make this a NamedTuple subclass (Python 3.6+) or dataclass (3.7+).
Gate = NamedTuple("Gate", [("kind", str), ("parameters", Tuple[float, ...]),
                           ("operands", Tuple[int, ...])])

#:
GateSequence = Tuple[Gate, ...]

#:
GateGenerator = Iterable[Gate]


def collect_operands(gates: GateGenerator) -> Set[int]:
    """Return all the qubit operands used in the given gate sequence."""
    qubits = set()
    for g in gates:
        qubits.update(g.operands)
    return qubits


def remap_operands(gates: GateGenerator, operand_map: Dict[int, int]) -> GateGenerator:
    """Change all the operand indices in a gate string according to the given map.

    For instance, ``remap_operands(sequence, {0: 1})`` will make all gates in the
    sequence that target qubit ``0`` act on qubit ``1`` instead (including cases where
    ``0`` is used as part of a multi-qubit gate).
    """
    for g in gates:
        yield Gate(g.kind, g.parameters,
                   tuple(operand_map.get(k, k) for k in g.operands))
