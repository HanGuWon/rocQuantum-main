from __future__ import annotations

import re
from abc import ABC, abstractmethod
from numbers import Number
from typing import TYPE_CHECKING, Iterable, List, Sequence, Tuple

if TYPE_CHECKING:
    from .kernel import QuantumKernel


_PAULI_TOKEN_RE = re.compile(r"([IXYZixyz])(\d+)")


class QuantumOperator(ABC):
    """Abstract base class for quantum observables."""

    def __init__(self, coefficient: Number = 1.0):
        self.coefficient = complex(coefficient)

    def __mul__(self, other):
        if isinstance(other, Number):
            new_op = self.__class__.__new__(self.__class__)
            new_op.__dict__.update(self.__dict__)
            new_op.coefficient = self.coefficient * other
            return new_op
        raise NotImplementedError(f"Cannot multiply QuantumOperator by {type(other)}")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        if isinstance(other, QuantumOperator):
            return SumOperator([self, other])
        raise NotImplementedError(f"Cannot add QuantumOperator to {type(other)}")

    @abstractmethod
    def to_string(self) -> str:
        pass


class PauliOperator(QuantumOperator):
    """Represents a single Pauli-string term, e.g. ``0.5 * X0 Y1 Z2``."""

    def __init__(self, pauli_string: str, coefficient: Number = 1.0):
        super().__init__(coefficient)
        self.pauli_string = pauli_string
        _parse_pauli_string(pauli_string)

    def to_string(self) -> str:
        return f"{self.coefficient} * {self.pauli_string}"


class HermitianOperator(QuantumOperator):
    """Represents an operator defined by a Hermitian matrix."""

    def __init__(self, matrix, coefficient: Number = 1.0):
        super().__init__(coefficient)
        self.matrix = matrix

    def to_string(self) -> str:
        return f"{self.coefficient} * Hermitian(matrix)"


class SumOperator(QuantumOperator):
    """Represents a sum of quantum operators."""

    def __init__(self, operators: list[QuantumOperator], coefficient: Number = 1.0):
        super().__init__(coefficient)
        self.terms = operators

    def __add__(self, other):
        if isinstance(other, SumOperator):
            return SumOperator(list(self.terms) + list(other.terms))
        if isinstance(other, QuantumOperator):
            return SumOperator(list(self.terms) + [other])
        raise NotImplementedError

    def to_string(self) -> str:
        return " + ".join(f"({term.to_string()})" for term in self.terms)


def _parse_pauli_string(pauli_string: str) -> List[Tuple[str, int]]:
    if not isinstance(pauli_string, str):
        raise TypeError("Pauli strings must be strings.")

    compact = pauli_string.replace(" ", "").replace(",", "")
    if not compact or compact.upper() == "I":
        return []

    parsed: List[Tuple[str, int]] = []
    seen_qubits = set()
    position = 0
    while position < len(compact):
        match = _PAULI_TOKEN_RE.match(compact, position)
        if match is None:
            raise ValueError(
                f"Invalid Pauli-string syntax '{pauli_string}'. "
                "Expected tokens like 'X0', 'Y1', or 'Z2'."
            )

        pauli = match.group(1).upper()
        qubit = int(match.group(2))
        if qubit in seen_qubits and pauli != "I":
            raise ValueError(
                f"Pauli string '{pauli_string}' repeats qubit {qubit}. "
                "Each qubit may appear at most once per term."
            )

        if pauli != "I":
            parsed.append((pauli, qubit))
            seen_qubits.add(qubit)
        position = match.end()

    return parsed


def iter_pauli_terms(operator: QuantumOperator) -> List[Tuple[complex, List[Tuple[str, int]]]]:
    """Expand an operator into ``(coefficient, pauli-term)`` pairs."""

    if isinstance(operator, PauliOperator):
        return [(operator.coefficient, _parse_pauli_string(operator.pauli_string))]

    if isinstance(operator, SumOperator):
        terms: List[Tuple[complex, List[Tuple[str, int]]]] = []
        for term in operator.terms:
            for coefficient, paulis in iter_pauli_terms(term):
                terms.append((operator.coefficient * coefficient, paulis))
        return terms

    if isinstance(operator, HermitianOperator):
        raise NotImplementedError(
            "HermitianOperator expectation values are not wired to a native backend yet. "
            "Use PauliOperator or SumOperator for the current ROCm-native observe path."
        )

    raise TypeError(f"Unsupported quantum operator type: {type(operator)!r}")


def get_expectation_value(
    kernel: "QuantumKernel",
    operator: QuantumOperator,
    backend: str = "state_vector",
    **kwargs,
):
    """Compute the expectation value of an operator via the canonical runtime."""

    from .kernel import observe

    return observe(kernel, operator, backend=backend, **kwargs)
