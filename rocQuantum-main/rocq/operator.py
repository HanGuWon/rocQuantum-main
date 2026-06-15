from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from numbers import Integral, Number
from typing import TYPE_CHECKING, Iterable, List, Sequence, Tuple

if TYPE_CHECKING:
    from .kernel import QuantumKernel


_PAULI_TOKEN_RE = re.compile(r"([IXYZixyz])(\d+)")


def _validate_positive_integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer.")
    index = int(value)
    if index <= 0:
        raise ValueError(f"{name} must be positive.")
    return index


def _validate_nonnegative_integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a non-negative integer.")
    index = int(value)
    if index < 0:
        raise ValueError(f"{name} must be non-negative.")
    return index


def _normalize_observable_targets(targets, name: str):
    if targets is None:
        return None
    if isinstance(targets, bool) or isinstance(targets, (str, bytes)):
        raise ValueError(f"{name} must be an integer index or a sequence of integer indices.")
    if isinstance(targets, Integral):
        raw_targets = [targets]
    else:
        try:
            raw_targets = list(targets)
        except TypeError as exc:
            raise ValueError(
                f"{name} must be an integer index or a sequence of integer indices."
            ) from exc

    normalized = [_validate_nonnegative_integer(target, name) for target in raw_targets]
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name}s must be unique.")
    return normalized


def _normalize_sparse_shape(shape) -> tuple[int, int]:
    if isinstance(shape, (str, bytes)):
        raise ValueError("SparseHamiltonianOperator shape must have two dimensions.")
    try:
        raw_shape = list(shape)
    except TypeError as exc:
        raise ValueError("SparseHamiltonianOperator shape must have two dimensions.") from exc
    if len(raw_shape) != 2:
        raise ValueError("SparseHamiltonianOperator shape must have two dimensions.")

    rows = _validate_positive_integer(
        raw_shape[0],
        "SparseHamiltonianOperator shape dimension",
    )
    cols = _validate_positive_integer(
        raw_shape[1],
        "SparseHamiltonianOperator shape dimension",
    )
    if rows != cols:
        raise ValueError("SparseHamiltonianOperator shape must be square.")
    if rows & (rows - 1):
        raise ValueError("SparseHamiltonianOperator shape dimension must be a power of two.")
    return rows, cols


def _normalize_coefficient(value, name: str = "coefficient") -> complex:
    if isinstance(value, bool) or not isinstance(value, Number):
        raise ValueError(f"{name} must be a finite numeric value.")
    coefficient = complex(value)
    if not math.isfinite(coefficient.real) or not math.isfinite(coefficient.imag):
        raise ValueError(f"{name} must be finite.")
    return coefficient


class QuantumOperator(ABC):
    """Abstract base class for quantum observables."""

    def __init__(self, coefficient: Number = 1.0):
        self.coefficient = _normalize_coefficient(coefficient)

    def __mul__(self, other):
        if isinstance(other, Number):
            scalar = _normalize_coefficient(other, "scalar")
            new_op = self.__class__.__new__(self.__class__)
            new_op.__dict__.update(self.__dict__)
            new_op.coefficient = _normalize_coefficient(self.coefficient * scalar)
            return new_op
        if isinstance(other, QuantumOperator):
            return _multiply_operator_pauli_terms(self, other)
        raise NotImplementedError(f"Cannot multiply QuantumOperator by {type(other)}")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, Number):
            scalar = _normalize_coefficient(other, "divisor")
            if scalar == 0:
                raise ValueError("divisor must be non-zero.")
            return self * (1 / scalar)
        raise NotImplementedError(f"Cannot divide QuantumOperator by {type(other)}")

    def __add__(self, other):
        if isinstance(other, QuantumOperator):
            return SumOperator([self, other])
        if isinstance(other, Number):
            scalar = _normalize_coefficient(other, "scalar")
            if scalar == 0:
                return self
            return SumOperator([self, _identity_operator(scalar)])
        raise NotImplementedError(f"Cannot add QuantumOperator to {type(other)}")

    def __radd__(self, other):
        if isinstance(other, Number):
            scalar = _normalize_coefficient(other, "scalar")
            if scalar == 0:
                return self
            return SumOperator([_identity_operator(scalar), self])
        raise NotImplementedError(f"Cannot add {type(other)} to QuantumOperator")

    def __neg__(self):
        return -1 * self

    def __sub__(self, other):
        if isinstance(other, QuantumOperator):
            return self + (-other)
        if isinstance(other, Number):
            scalar = _normalize_coefficient(other, "scalar")
            if scalar == 0:
                return self
            return self + _identity_operator(-scalar)
        raise NotImplementedError(f"Cannot subtract {type(other)} from QuantumOperator")

    def __rsub__(self, other):
        if isinstance(other, Number):
            return _identity_operator(_normalize_coefficient(other, "scalar")) + (-self)
        raise NotImplementedError(f"Cannot subtract QuantumOperator from {type(other)}")

    @abstractmethod
    def to_string(self) -> str:
        pass


class PauliOperator(QuantumOperator):
    """Represents a single Pauli-string term, e.g. ``0.5 * X0 Y1 Z2``."""

    def __init__(self, pauli_string: str, coefficient: Number = 1.0):
        super().__init__(coefficient)
        self.pauli_string = pauli_string
        _parse_pauli_string(pauli_string)

    def __mul__(self, other):
        if isinstance(other, PauliOperator):
            phase, paulis = _multiply_pauli_terms(
                _parse_pauli_string(self.pauli_string),
                _parse_pauli_string(other.pauli_string),
            )
            return PauliOperator(_format_pauli_string(paulis), self.coefficient * other.coefficient * phase)
        return super().__mul__(other)

    def to_string(self) -> str:
        return f"{self.coefficient} * {self.pauli_string}"


def _identity_operator(coefficient: Number) -> PauliOperator:
    return PauliOperator("I", coefficient=coefficient)


class HermitianOperator(QuantumOperator):
    """Represents an operator defined by a Hermitian matrix."""

    def __init__(self, matrix, coefficient: Number = 1.0, targets=None):
        super().__init__(coefficient)
        self.matrix = matrix
        self.targets = _normalize_observable_targets(targets, "HermitianOperator target")

    def to_string(self) -> str:
        return f"{self.coefficient} * Hermitian(matrix)"


class SparseHamiltonianOperator(QuantumOperator):
    """Represents a full-state sparse Hamiltonian in CSR form."""

    def __init__(self, data, indices, indptr, shape, coefficient: Number = 1.0):
        super().__init__(coefficient)
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.shape = _normalize_sparse_shape(shape)

    def to_string(self) -> str:
        return f"{self.coefficient} * SparseHamiltonian(CSR, shape={self.shape})"


class SumOperator(QuantumOperator):
    """Represents a sum of quantum operators."""

    def __init__(self, operators: list[QuantumOperator], coefficient: Number = 1.0):
        super().__init__(coefficient)
        self.terms = operators

    def _add_terms(self) -> list[QuantumOperator]:
        if self.coefficient == 1:
            return list(self.terms)
        return [SumOperator(list(self.terms), coefficient=self.coefficient)]

    def __add__(self, other):
        if isinstance(other, SumOperator):
            return SumOperator(self._add_terms() + other._add_terms())
        if isinstance(other, QuantumOperator):
            return SumOperator(self._add_terms() + [other])
        if isinstance(other, Number):
            if other == 0:
                return self
            return SumOperator(self._add_terms() + [_identity_operator(other)])
        raise NotImplementedError

    def to_string(self) -> str:
        joined_terms = " + ".join(f"({term.to_string()})" for term in self.terms)
        if self.coefficient == 1:
            return joined_terms
        return f"{self.coefficient} * ({joined_terms})"


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


_PAULI_PRODUCT_TABLE = {
    ("X", "Y"): (1j, "Z"),
    ("Y", "X"): (-1j, "Z"),
    ("Y", "Z"): (1j, "X"),
    ("Z", "Y"): (-1j, "X"),
    ("Z", "X"): (1j, "Y"),
    ("X", "Z"): (-1j, "Y"),
}


def _multiply_pauli_terms(
    left: Sequence[Tuple[str, int]],
    right: Sequence[Tuple[str, int]],
) -> Tuple[complex, List[Tuple[str, int]]]:
    phase = 1.0 + 0.0j
    by_qubit = {int(qubit): pauli for pauli, qubit in left}

    for pauli, qubit in right:
        qubit = int(qubit)
        if qubit not in by_qubit:
            by_qubit[qubit] = pauli
            continue

        existing = by_qubit[qubit]
        if existing == pauli:
            del by_qubit[qubit]
            continue

        local_phase, product_pauli = _PAULI_PRODUCT_TABLE[(existing, pauli)]
        phase *= local_phase
        by_qubit[qubit] = product_pauli

    return phase, [(pauli, qubit) for qubit, pauli in sorted(by_qubit.items())]


def _format_pauli_string(paulis: Sequence[Tuple[str, int]]) -> str:
    if not paulis:
        return "I"
    return " ".join(f"{pauli}{int(qubit)}" for pauli, qubit in paulis)


def _multiply_operator_pauli_terms(left: QuantumOperator, right: QuantumOperator) -> QuantumOperator:
    product_terms = []
    for left_coefficient, left_paulis in iter_pauli_terms(left):
        for right_coefficient, right_paulis in iter_pauli_terms(right):
            phase, paulis = _multiply_pauli_terms(left_paulis, right_paulis)
            product_terms.append(
                PauliOperator(
                    _format_pauli_string(paulis),
                    coefficient=left_coefficient * right_coefficient * phase,
                )
            )

    if len(product_terms) == 1:
        return product_terms[0]
    return SumOperator(product_terms)


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

    if isinstance(operator, (HermitianOperator, SparseHamiltonianOperator)):
        raise NotImplementedError(
            f"{operator.__class__.__name__} cannot be expanded by iter_pauli_terms(). "
            "Use rocq.observe() or get_expectation_value() to evaluate matrix observables."
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
