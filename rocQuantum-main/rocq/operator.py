from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from numbers import Integral, Number
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

import numpy as np

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


def _normalize_hermitian_matrix(matrix):
    try:
        raw_matrix = np.asarray(matrix, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "HermitianOperator matrix must be a finite numeric square matrix."
        ) from exc

    if raw_matrix.ndim != 2 or raw_matrix.shape[0] != raw_matrix.shape[1]:
        raise ValueError("HermitianOperator matrix must be square.")

    matrix_dim = int(raw_matrix.shape[0])
    if matrix_dim <= 0 or matrix_dim & (matrix_dim - 1):
        raise ValueError("HermitianOperator matrix dimension must be a power of two.")

    normalized = []
    for value in raw_matrix.reshape(-1):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Number):
            raise ValueError("HermitianOperator matrix must contain finite numeric values.")
        scalar = complex(value)
        if not math.isfinite(scalar.real) or not math.isfinite(scalar.imag):
            raise ValueError("HermitianOperator matrix must be finite.")
        normalized.append(scalar)
    return np.asarray(normalized, dtype=np.complex128).reshape(raw_matrix.shape)


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


def _normalize_sparse_data(data):
    try:
        raw_data = np.asarray(data, dtype=object).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "SparseHamiltonianOperator CSR data must contain finite numeric values."
        ) from exc

    normalized = []
    for value in raw_data:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Number):
            raise ValueError(
                "SparseHamiltonianOperator CSR data must contain finite numeric values."
            )
        scalar = complex(value)
        if not math.isfinite(scalar.real) or not math.isfinite(scalar.imag):
            raise ValueError("SparseHamiltonianOperator CSR data must be finite.")
        normalized.append(scalar)
    return np.asarray(normalized, dtype=np.complex128)


def _normalize_sparse_index_vector(values, label: str):
    if isinstance(values, (str, bytes)):
        raise ValueError(f"SparseHamiltonianOperator CSR {label} must contain integer indices.")
    try:
        raw_values = np.asarray(values, dtype=object).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"SparseHamiltonianOperator CSR {label} must contain integer indices."
        ) from exc
    normalized = [
        _validate_nonnegative_integer(value, f"SparseHamiltonianOperator CSR {label}")
        for value in raw_values
    ]
    try:
        return np.asarray(normalized, dtype=np.int64)
    except (OverflowError, ValueError) as exc:
        raise ValueError(
            f"SparseHamiltonianOperator CSR {label} must fit in signed 64-bit integers."
        ) from exc


def _normalize_sparse_csr(data, indices, indptr, shape: tuple[int, int]):
    data_array = _normalize_sparse_data(data)
    indices_array = _normalize_sparse_index_vector(indices, "indices")
    indptr_array = _normalize_sparse_index_vector(indptr, "indptr")
    rows, cols = shape

    if data_array.size != indices_array.size:
        raise ValueError("SparseHamiltonianOperator CSR data and indices lengths must match.")
    if indptr_array.size != rows + 1:
        raise ValueError("SparseHamiltonianOperator CSR indptr length must equal rows + 1.")
    if (
        indptr_array.size == 0
        or int(indptr_array[0]) != 0
        or int(indptr_array[-1]) != data_array.size
    ):
        raise ValueError("SparseHamiltonianOperator CSR indptr must start at 0 and end at nnz.")
    if np.any(indptr_array[:-1] > indptr_array[1:]):
        raise ValueError("SparseHamiltonianOperator CSR indptr must be monotonic.")
    if np.any(indices_array < 0) or np.any(indices_array >= cols):
        raise ValueError("SparseHamiltonianOperator CSR column index is out of bounds.")
    return data_array, indices_array, indptr_array


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
        raise TypeError(f"Cannot multiply QuantumOperator by {type(other).__name__}.")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, Number):
            scalar = _normalize_coefficient(other, "divisor")
            if scalar == 0:
                raise ValueError("divisor must be non-zero.")
            return self * (1 / scalar)
        raise TypeError(f"Cannot divide QuantumOperator by {type(other).__name__}.")

    def __add__(self, other):
        if isinstance(other, QuantumOperator):
            return SumOperator([self, other])
        if isinstance(other, Number):
            scalar = _normalize_coefficient(other, "scalar")
            if scalar == 0:
                return self
            return SumOperator([self, _identity_operator(scalar)])
        raise TypeError(f"Cannot add QuantumOperator to {type(other).__name__}.")

    def __radd__(self, other):
        if isinstance(other, Number):
            scalar = _normalize_coefficient(other, "scalar")
            if scalar == 0:
                return self
            return SumOperator([_identity_operator(scalar), self])
        raise TypeError(f"Cannot add {type(other).__name__} to QuantumOperator.")

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
        raise TypeError(f"Cannot subtract {type(other).__name__} from QuantumOperator.")

    def __rsub__(self, other):
        if isinstance(other, Number):
            return _identity_operator(_normalize_coefficient(other, "scalar")) + (-self)
        raise TypeError(f"Cannot subtract QuantumOperator from {type(other).__name__}.")

    @abstractmethod
    def to_string(self) -> str:
        pass


class PauliOperator(QuantumOperator):
    """Represents a single Pauli-string term, e.g. ``0.5 * X0 Y1 Z2``."""

    def __init__(
        self,
        pauli_string: str,
        coefficient: Number = 1.0,
        *,
        num_qubits: Optional[int] = None,
    ):
        super().__init__(coefficient)
        self.pauli_string = pauli_string
        parsed = _parse_pauli_string(pauli_string)
        compact = pauli_string.replace(" ", "").replace(",", "")
        token_indices = [
            int(match.group(2)) for match in _PAULI_TOKEN_RE.finditer(compact)
        ]
        inferred_width = max(token_indices, default=-1) + 1
        if inferred_width <= 0:
            inferred_width = max((qubit for _, qubit in parsed), default=-1) + 1
        inferred_width = max(1, inferred_width)
        if num_qubits is None:
            self.num_qubits = inferred_width
        else:
            width = _validate_positive_integer(num_qubits, "num_qubits")
            if width < inferred_width:
                raise ValueError(
                    "num_qubits must include every Pauli-string qubit index."
                )
            self.num_qubits = width

    def __mul__(self, other):
        if isinstance(other, PauliOperator):
            phase, paulis = _multiply_pauli_terms(
                _parse_pauli_string(self.pauli_string),
                _parse_pauli_string(other.pauli_string),
            )
            return PauliOperator(
                _format_pauli_string(paulis),
                self.coefficient * other.coefficient * phase,
                num_qubits=max(self.num_qubits, other.num_qubits),
            )
        return super().__mul__(other)

    def to_string(self) -> str:
        return f"{self.coefficient} * {self.pauli_string}"

    def get_pauli_word(self) -> str:
        """Return the dense CUDA-Q-style I/X/Y/Z word for this term.

        Character position ``i`` corresponds to qubit ``i``.  Explicit
        ``num_qubits`` metadata preserves trailing identities, which is required
        by CUDA-QX code metadata such as seven-qubit Steane stabilizers.
        """

        word = ["I"] * self.num_qubits
        for pauli, qubit in _parse_pauli_string(self.pauli_string):
            word[int(qubit)] = pauli
        return "".join(word)


def _identity_operator(coefficient: Number) -> PauliOperator:
    return PauliOperator("I", coefficient=coefficient)


class HermitianOperator(QuantumOperator):
    """Represents an operator defined by a Hermitian matrix."""

    def __init__(self, matrix, coefficient: Number = 1.0, targets=None):
        super().__init__(coefficient)
        self.matrix = _normalize_hermitian_matrix(matrix)
        self.targets = _normalize_observable_targets(targets, "HermitianOperator target")

    def to_string(self) -> str:
        return f"{self.coefficient} * Hermitian(matrix)"


class SparseHamiltonianOperator(QuantumOperator):
    """Represents a full-state sparse Hamiltonian in CSR form."""

    def __init__(self, data, indices, indptr, shape, coefficient: Number = 1.0):
        super().__init__(coefficient)
        self.shape = _normalize_sparse_shape(shape)
        self.data, self.indices, self.indptr = _normalize_sparse_csr(
            data,
            indices,
            indptr,
            self.shape,
        )

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
            scalar = _normalize_coefficient(other, "scalar")
            if scalar == 0:
                return self
            return SumOperator(self._add_terms() + [_identity_operator(scalar)])
        raise TypeError(f"Cannot add SumOperator to {type(other).__name__}.")

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


def _operator_num_qubits(operator: QuantumOperator) -> int:
    """Infer the smallest full-register width required by ``operator``."""

    if isinstance(operator, PauliOperator):
        return operator.num_qubits

    if isinstance(operator, SumOperator):
        return max(
            (_operator_num_qubits(term) for term in operator.terms),
            default=1,
        )

    if isinstance(operator, HermitianOperator):
        local_qubits = int(round(math.log2(operator.matrix.shape[0])))
        if operator.targets is None:
            return max(1, local_qubits)
        if len(operator.targets) != local_qubits:
            raise ValueError(
                "HermitianOperator target count must match its matrix dimension."
            )
        return max(1, max(operator.targets, default=-1) + 1)

    if isinstance(operator, SparseHamiltonianOperator):
        return max(1, int(round(math.log2(operator.shape[0]))))

    raise TypeError(f"Unsupported quantum operator type: {type(operator)!r}")


def _embed_local_matrix(matrix: np.ndarray, targets: Sequence[int], num_qubits: int) -> np.ndarray:
    """Embed a local matrix using rocQuantum's qubit-0-is-LSB convention."""

    target_list = [int(target) for target in targets]
    target_count = len(target_list)
    local_dim = 1 << target_count
    full_dim = 1 << int(num_qubits)
    if matrix.shape != (local_dim, local_dim):
        raise ValueError("Local operator matrix dimension must equal 2**len(targets).")

    embedded = np.zeros((full_dim, full_dim), dtype=np.complex128)
    target_mask = sum(1 << target for target in target_list)
    for column in range(full_dim):
        local_column = 0
        for bit, target in enumerate(target_list):
            if (column >> target) & 1:
                local_column |= 1 << bit
        base = column & ~target_mask
        for local_row in range(local_dim):
            row = base
            for bit, target in enumerate(target_list):
                if (local_row >> bit) & 1:
                    row |= 1 << target
            embedded[row, column] = matrix[local_row, local_column]
    return embedded


def operator_to_matrix(operator: QuantumOperator, num_qubits: int = None) -> np.ndarray:
    """Return a dense matrix for a canonical quantum operator.

    The conversion is a CPU reference utility intended for small exact-oracle
    tests, solver construction, and dynamics.  Its basis ordering matches the
    runtime: qubit 0 is the least-significant computational-basis bit.
    """

    if not isinstance(operator, QuantumOperator):
        raise TypeError("operator_to_matrix() expects a QuantumOperator instance.")
    inferred_qubits = _operator_num_qubits(operator)
    if num_qubits is None:
        resolved_qubits = inferred_qubits
    else:
        resolved_qubits = _validate_positive_integer(num_qubits, "num_qubits")
        if resolved_qubits < inferred_qubits:
            raise ValueError(
                f"num_qubits={resolved_qubits} is too small for an operator requiring "
                f"{inferred_qubits} qubits."
            )

    dimension = 1 << resolved_qubits
    if isinstance(operator, SumOperator):
        result = np.zeros((dimension, dimension), dtype=np.complex128)
        for term in operator.terms:
            result += operator_to_matrix(term, resolved_qubits)
        return complex(operator.coefficient) * result

    if isinstance(operator, PauliOperator):
        result = np.zeros((dimension, dimension), dtype=np.complex128)
        local_matrices = {
            "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
            "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
            "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
        }
        for coefficient, paulis in iter_pauli_terms(operator):
            term = np.array([[1.0 + 0.0j]], dtype=np.complex128)
            by_qubit = {int(qubit): pauli for pauli, qubit in paulis}
            for qubit in reversed(range(resolved_qubits)):
                term = np.kron(
                    term,
                    local_matrices.get(
                        by_qubit.get(qubit),
                        np.eye(2, dtype=np.complex128),
                    ),
                )
            result += complex(coefficient) * term
        return result

    if isinstance(operator, HermitianOperator):
        matrix = np.asarray(operator.matrix, dtype=np.complex128)
        local_qubits = int(round(math.log2(matrix.shape[0])))
        targets = (
            list(range(local_qubits))
            if operator.targets is None
            else list(operator.targets)
        )
        return complex(operator.coefficient) * _embed_local_matrix(
            matrix,
            targets,
            resolved_qubits,
        )

    if isinstance(operator, SparseHamiltonianOperator):
        if operator.shape != (dimension, dimension):
            raise ValueError(
                "SparseHamiltonianOperator spans a fixed full register and cannot "
                "be embedded into a different num_qubits value."
            )
        dense = np.zeros(operator.shape, dtype=np.complex128)
        for row in range(operator.shape[0]):
            start = int(operator.indptr[row])
            end = int(operator.indptr[row + 1])
            dense[row, operator.indices[start:end]] = operator.data[start:end]
        return complex(operator.coefficient) * dense

    raise TypeError(f"Unsupported quantum operator type: {type(operator)!r}")


def get_expectation_value(
    kernel: "QuantumKernel",
    operator: QuantumOperator,
    backend: Optional[str] = None,
    **kwargs,
):
    """Compute the expectation value of an operator via the canonical runtime."""

    from .kernel import observe

    return observe(kernel, operator, backend=backend, **kwargs)
