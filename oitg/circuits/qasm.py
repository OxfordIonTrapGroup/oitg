"""Functionality for interfacing with OpenQASM.

QASM is a straightforward text-based format for representing quantum computations in the
circuit model. At this point, we do not support any classical feedback, etc.; just
straight-line gate sequences starting with state preparation and ending with
measurement.

For more complex QASM support, we should integrate an external library.
"""

import math
import re
from typing import Callable, Iterable
from .gate import Gate, GateGenerator, collect_operands

Statements = Iterable[str]


def stringify_param(p) -> str:
    """Returns the string form of a parameter p; with fractions of π pretty-printed for
    readability."""
    if isinstance(p, float):
        sign = ""
        if p < 0:
            sign = "-"
            p *= -1
        if math.isclose(p, 2 * math.pi):
            return sign + "2 * pi"
        if math.isclose(p, math.pi):
            return sign + "pi"
        if math.isclose(p, math.pi / 2):
            return sign + "pi / 2"
        if math.isclose(p, math.pi / 4):
            return sign + "pi / 4"
    return str(p)


def gate_to_qasm(gate: Gate) -> Statements:
    """Returns a QASM statement corresponding to the given :class:`.Gate`."""
    result = gate.kind
    if gate.parameters:
        result += "("
        result += ", ".join(stringify_param(p) for p in gate.parameters)
        result += ")"
    result += " "
    result += ", ".join("q[{}]".format(o) for o in gate.operands)
    yield result


def default_prologue(num_qubits: int) -> Statements:
    # TODO: We currently use the Quantum Experience standard library for the gate
    # definitions; should replace with a taylor-made one.
    yield "OPENQASM 2.0"
    yield "include \"qelib1.inc\""
    # Use single-letter names `q`/`r` for brevity in plots.
    yield "qreg q[{}]".format(num_qubits)
    yield "creg r[{}]".format(num_qubits)


def default_epilogue(num_qubits: int) -> Statements:
    yield "measure q -> r"


def gate_experiment_to_qasm(
        gates: GateGenerator,
        prologue_fn: Callable[[int], Statements] = default_prologue,
        epilogue_fn: Callable[[int], Statements] = default_epilogue) -> Statements:
    num_qubits = max(collect_operands(gates)) + 1

    yield from prologue_fn(num_qubits)
    for g in gates:
        yield from gate_to_qasm(g)
    yield from epilogue_fn(num_qubits)


def stringify_qasm(stmts: Statements) -> str:
    """Concatenate/format a list of QASM statements into a single string."""
    result = ""
    for stmt in stmts:
        result += stmt + ";\n"
    return result


def parse_gate_sequence_string(string: str) -> GateGenerator:
    """Parse a simple gate sequence in ;-delimited string form into its constituent
    gates.
    """
    # This implementation is an example of what usually *not* to do; we should really
    # just pull in an external QASM parser if we want to support more complex strings.
    for statement in string.split(";"):
        if not statement:
            continue
        match = re.match(r"(\w+)\s*(\(.*\))?\s*((q\[\d+\],?\s?)*)", statement)
        if match is None:
            raise ValueError(
                "Not a valid gate string statement: '{}'".format(statement))
        kind, param_string, operand_string, _ = match.groups()

        parameters = ()
        if param_string is not None:
            parameters = eval(param_string, None, {"pi": math.pi})
            if not isinstance(parameters, tuple):
                parameters = (parameters, )

        operands = tuple(
            int(m.groups()[0]) for m in re.finditer(r"q\[(\d+)\]", operand_string))

        yield Gate(kind, parameters, operands)
