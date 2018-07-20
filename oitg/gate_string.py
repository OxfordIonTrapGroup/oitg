from enum import Enum, unique
from typing import List, Tuple


@unique
class GateType(Enum):
    identity = 0
    plus_pi_x = 1
    plus_pi_2_x = 2
    minus_pi_x = 3
    minus_pi_2_x = 4
    plus_pi_y = 5
    plus_pi_2_y = 6
    minus_pi_y = 7
    minus_pi_2_y = 8


class GateStringParseError(Exception):
    pass


class GateStringParser:
    """Parses a simple single qubit gate sequence specification in string form
    into an expanded list of gates."""

    def __init__(self, default_transition, alternate_transition):
        self.default_transition = default_transition
        self.alternate_transition = alternate_transition

    def parse(self, gate_string: str) -> List[Tuple[str, GateType]]:
        gates = []
        self._parse(None, iter(gate_string), False, False, gates)
        return gates


    def _parse(self, char, it, one_only, in_parens, gates):
        try:
            if char is None:
                char = next(it)
            while char:
                char = _skip_whitespace(char, it)

                # Grouping (subsequences).
                if char == "(":
                    char = self._parse(None, it, False, True, gates)
                    if one_only:
                        return char
                    continue

                if char == ")":
                    if not in_parens:
                        raise GateStringParseError("Stray ')'")
                    in_parens = False
                    return next(it)

                # Parse number of repeats.
                num_repeats = 0
                while char.isdigit():
                    num_repeats *= 10
                    num_repeats += int(char)
                    char = _ensure_next(it, "Premature end after repeat spec")

                if num_repeats > 0:
                    repeat_gates = []
                    char = self._parse(char, it, True, False, repeat_gates)
                    gates.extend(repeat_gates * num_repeats)
                    if one_only:
                        return char
                    continue

                char = _skip_whitespace(char, it, "Premature end after repeat spec")

                sign = "+"
                if _is_sign(char):
                    sign = char
                    char = _ensure_next(it, "Premature end after sign spec")

                char = _skip_whitespace(char, it, "Premature end after sign spec")

                gate_type = _lookup_gate_type(sign, char)
                transition = self.default_transition
                try:
                    # TODO: Support explicit `x[transition_1]Y[transition_2]` syntax.
                    char = next(it)
                    if char == "'":
                        transition = self.alternate_transition
                        char = next(it)
                finally:
                    gates.append((transition, gate_type))

                if one_only:
                    return char

            return next(it)
        except StopIteration:
            pass

        if in_parens:
            raise GateStringParseError("Missing ')'")


def _lookup_gate_type(sign, axis_angle):
    if not _is_sign(sign):
        raise ValueError("Invalid sign: {}".format(sign))
    pos = sign == "+"

    if axis_angle == "x":
        return GateType.plus_pi_2_x if pos else GateType.minus_pi_2_x
    if axis_angle == "y":
        return GateType.plus_pi_2_y if pos else GateType.minus_pi_2_y
    if axis_angle == "X":
        return GateType.plus_pi_x if pos else GateType.minus_pi_x
    if axis_angle == "Y":
        return GateType.plus_pi_y if pos else GateType.minus_pi_y
    raise ValueError("Invalid axis/angle: {}".format(axis_angle))


def _is_sign(char):
    return char in ["+", "-"]


def _skip_whitespace(char, it, error=None):
    while char.isspace():
        if error:
            char = _ensure_next(it, error)
        else:
            char = next(it)
    return char


def _ensure_next(it, error):
    try:
        return next(it)
    except StopIteration:
        raise GateStringParseError(error)
