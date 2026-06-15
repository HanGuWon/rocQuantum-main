from __future__ import annotations

from collections import Counter
import math
from numbers import Integral, Number, Real
from typing import Iterable, Sequence

import numpy as np


GATE_ALIASES = {
    "CNOT": "CNOT",
    "CCX": "MCX",
    "CX": "CNOT",
    "CSWAP": "CSWAP",
    "CZ": "CZ",
    "CRX": "CRX",
    "CRY": "CRY",
    "CRZ": "CRZ",
    "CP": "CP",
    "CPHASE": "CP",
    "CONTROLLEDPHASE": "CP",
    "FREDKIN": "CSWAP",
    "H": "H",
    "HADAMARD": "H",
    "I": "I",
    "ID": "I",
    "IDENTITY": "I",
    "MCX": "MCX",
    "PAULIX": "X",
    "PAULIY": "Y",
    "PAULIZ": "Z",
    "P": "P",
    "PHASE": "P",
    "RX": "RX",
    "RY": "RY",
    "RZ": "RZ",
    "S": "S",
    "SDG": "SDG",
    "SWAP": "SWAP",
    "T": "T",
    "TDAG": "TDG",
    "TDG": "TDG",
    "TOFFOLI": "MCX",
    "X": "X",
    "Y": "Y",
    "Z": "Z",
}

NO_OPS = {"barrier", "delay", "save_statevector", "snapshot"}


def normalize_gate_name(name: str) -> str:
    key = str(name).replace("_", "").replace("-", "").upper()
    return GATE_ALIASES.get(key, str(name).upper())


def normalize_targets(targets: Iterable[int]) -> list[int]:
    if isinstance(targets, (bool, np.bool_)) or isinstance(targets, (str, bytes)):
        raise ValueError("Operation targets must be a sequence of integer qubit indices.")
    try:
        raw_targets = list(targets)
    except TypeError as exc:
        raise TypeError("Operation targets must be a sequence of integer qubit indices.") from exc

    normalized = []
    for target in raw_targets:
        if isinstance(target, (bool, np.bool_)) or not isinstance(target, Integral):
            raise ValueError("Operation targets must be integer qubit indices.")
        normalized.append(int(target))
    return normalized


def normalize_pauli_expectation_payload(
    pauli_string: object,
    targets: Iterable[int],
    num_qubits: int | None = None,
) -> tuple[str, list[int]]:
    if not isinstance(pauli_string, str):
        raise ValueError("Pauli string must be a string.")

    normalized_targets = normalize_targets(targets)
    normalized_pauli = pauli_string.upper()
    if len(normalized_pauli) != len(normalized_targets):
        raise ValueError("Pauli string length must match target qubit count.")
    if any(pauli not in {"I", "X", "Y", "Z"} for pauli in normalized_pauli):
        raise ValueError("Pauli string may only contain I, X, Y, or Z.")
    if len(set(normalized_targets)) != len(normalized_targets):
        raise ValueError("Pauli expectation targets must be unique.")
    if any(target < 0 for target in normalized_targets):
        raise ValueError("Pauli expectation target is outside the qubit range.")
    if num_qubits is not None and any(target >= int(num_qubits) for target in normalized_targets):
        raise ValueError("Pauli expectation target is outside the qubit range.")
    return normalized_pauli, normalized_targets


def normalize_positive_integer(value: object, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be a positive integer.")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{label} must be positive.")
    return normalized


def normalize_shots(shots: object) -> int:
    return normalize_positive_integer(shots, "shots")


def normalize_params(params: Iterable[object] | None) -> list[float]:
    if params is None:
        return []
    normalized = []
    for param in params:
        if isinstance(param, (bool, np.bool_)) or isinstance(param, (str, bytes)):
            raise ValueError("Operation parameters must be finite real numeric values.")
        if hasattr(param, "bind"):
            raise TypeError(f"Unbound symbolic parameter {param!r} cannot be executed.")
        try:
            value = float(param)
        except (TypeError, ValueError) as exc:
            raise TypeError("Operation parameters must be real numeric values.") from exc
        if not math.isfinite(value):
            raise ValueError("Operation parameters must be finite real numeric values.")
        normalized.append(value)
    return normalized


def validate_finite_complex_array(values: object, label: str) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} must contain finite values.")


def as_complex_vector(values: object, label: str = "Statevector amplitudes") -> np.ndarray:
    try:
        raw = np.asarray(values, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain finite numeric values.") from exc

    normalized = []
    for value in raw.reshape(-1):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Number):
            raise ValueError(f"{label} must contain finite numeric values.")
        scalar = complex(value)
        if not math.isfinite(scalar.real) or not math.isfinite(scalar.imag):
            raise ValueError(f"{label} must contain finite values.")
        normalized.append(scalar)
    return np.ascontiguousarray(
        np.asarray(normalized, dtype=np.complex128).reshape(raw.shape)
    )


def as_complex_matrix(matrix: object, label: str = "Operation matrix") -> np.ndarray:
    try:
        raw = np.asarray(matrix, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain finite numeric values.") from exc

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError(f"{label} must be square.")

    normalized = []
    for value in raw.reshape(-1):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Number):
            raise ValueError(f"{label} must contain finite numeric values.")
        scalar = complex(value)
        if not math.isfinite(scalar.real) or not math.isfinite(scalar.imag):
            raise ValueError(f"{label} must contain finite values.")
        normalized.append(scalar)

    out = np.asarray(normalized, dtype=np.complex128).reshape(raw.shape)
    if out.ndim != 2 or out.shape[0] != out.shape[1]:
        raise ValueError(f"{label} must be square.")
    validate_finite_complex_array(out, label)
    return np.ascontiguousarray(out)


def validate_operation_targets(
    targets: Iterable[int],
    num_qubits: int,
    label: str = "Operation targets",
) -> list[int]:
    normalized = normalize_targets(targets)
    if not normalized:
        raise ValueError(f"{label} must include at least one qubit.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{label} must be unique.")
    for target in normalized:
        if target < 0 or target >= int(num_qubits):
            raise ValueError(f"{label} is out of range.")
    return normalized


def operation_matrix_for_targets(
    matrix: object,
    targets: Iterable[int],
    num_qubits: int,
    label: str,
) -> tuple[np.ndarray, list[int]]:
    normalized_targets = validate_operation_targets(targets, num_qubits, f"{label} targets")
    normalized_matrix = as_complex_matrix(matrix, label)
    dimension = 1 << len(normalized_targets)
    if normalized_matrix.shape != (dimension, dimension):
        raise ValueError(f"{label} dimension must be 2^len(targets).")
    return normalized_matrix, normalized_targets


def normalize_sparse_operation_csr(
    data: object,
    indices: object,
    indptr: object,
    shape: Iterable[int],
    targets: Iterable[int],
    num_qubits: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int], list[int]]:
    normalized_targets = validate_operation_targets(
        targets,
        num_qubits,
        "Sparse operation targets",
    )

    try:
        raw_data = np.asarray(data, dtype=object).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError("Sparse operation CSR data must contain finite numeric values.") from exc

    normalized_data_values = []
    for value in raw_data:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Number):
            raise ValueError("Sparse operation CSR data must contain finite numeric values.")
        scalar = complex(value)
        if not math.isfinite(scalar.real) or not math.isfinite(scalar.imag):
            raise ValueError("Sparse operation CSR data must contain finite values.")
        normalized_data_values.append(scalar)
    normalized_data = np.ascontiguousarray(
        np.asarray(normalized_data_values, dtype=np.complex128)
    )

    normalized_indices = _normalize_sparse_index_vector(indices, "indices")
    normalized_indptr = _normalize_sparse_index_vector(indptr, "indptr")
    normalized_shape = _normalize_sparse_shape(shape)
    local_dimension = 1 << len(normalized_targets)

    if normalized_shape != (local_dimension, local_dimension):
        raise ValueError("Sparse operation matrix shape must match target wires.")
    if normalized_indptr.size != local_dimension + 1:
        raise ValueError("Sparse operation CSR indptr length must equal rows + 1.")
    if normalized_data.size != normalized_indices.size:
        raise ValueError("Sparse operation CSR data and indices lengths must match.")
    if (
        normalized_indptr.size == 0
        or int(normalized_indptr[0]) != 0
        or int(normalized_indptr[-1]) != normalized_data.size
    ):
        raise ValueError("Sparse operation CSR indptr must start at 0 and end at nnz.")
    if np.any(normalized_indptr[:-1] > normalized_indptr[1:]):
        raise ValueError("Sparse operation CSR indptr must be monotonic.")
    if np.any(normalized_indices < 0) or np.any(normalized_indices >= local_dimension):
        raise ValueError("Sparse operation CSR column index is out of bounds.")

    return (
        normalized_data,
        normalized_indices,
        normalized_indptr,
        normalized_shape,
        normalized_targets,
    )


def _normalize_sparse_shape(shape: Iterable[int], prefix: str = "Sparse operation") -> tuple[int, int]:
    if isinstance(shape, (str, bytes)):
        raise ValueError(f"{prefix} matrix shape must have two dimensions.")
    try:
        raw_shape = list(shape)
    except TypeError as exc:
        raise ValueError(f"{prefix} matrix shape must have two dimensions.") from exc
    if len(raw_shape) != 2:
        raise ValueError(f"{prefix} matrix shape must have two dimensions.")

    rows = _normalize_positive_dimension(raw_shape[0], f"{prefix} matrix shape")
    cols = _normalize_positive_dimension(raw_shape[1], f"{prefix} matrix shape")
    if rows != cols:
        raise ValueError(f"{prefix} matrix must be square.")
    if rows & (rows - 1):
        raise ValueError(f"{prefix} matrix dimension must be a power of two.")
    return rows, cols


def _normalize_positive_dimension(value: object, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{label} dimensions must be positive integers.")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{label} dimensions must be positive.")
    return normalized


def _normalize_sparse_index_vector(
    values: object,
    label: str,
    prefix: str = "Sparse operation",
) -> np.ndarray:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{prefix} CSR {label} must contain integer indices.")
    try:
        raw_values = np.asarray(values, dtype=object).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{prefix} CSR {label} must contain integer indices."
        ) from exc

    normalized = []
    for value in raw_values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise ValueError(f"{prefix} CSR {label} must contain integer indices.")
        integer = int(value)
        if integer < 0:
            raise ValueError(f"{prefix} CSR {label} must be non-negative.")
        normalized.append(integer)
    return np.ascontiguousarray(np.asarray(normalized, dtype=np.int64))


def _reverse_bits(value: int, width: int) -> int:
    out = 0
    for bit in range(width):
        if (int(value) >> bit) & 1:
            out |= 1 << (width - 1 - bit)
    return out


def matrix_to_little_endian_wires(matrix: object) -> np.ndarray:
    """Convert a wire-ordered matrix to rocQuantum's local little-endian basis."""
    normalized = as_complex_matrix(matrix)
    dimension = normalized.shape[0]
    if dimension == 0 or dimension & (dimension - 1):
        raise ValueError("Operation matrix dimension must be a power of two.")

    num_wires = int(np.log2(dimension))
    if num_wires <= 1:
        return normalized

    permutation = np.array([_reverse_bits(index, num_wires) for index in range(dimension)])
    return np.ascontiguousarray(normalized[np.ix_(permutation, permutation)])


def sparse_matrix_to_little_endian_wires(sparse_matrix: object):
    """Convert a wire-ordered sparse matrix to rocQuantum's local little-endian basis."""
    if not hasattr(sparse_matrix, "tocsr"):
        raise TypeError("Sparse operation matrix must expose a CSR conversion.")

    normalized = sparse_matrix.tocsr()
    if normalized.ndim != 2 or normalized.shape[0] != normalized.shape[1]:
        raise ValueError("Sparse operation matrix must be square.")
    validate_finite_complex_array(np.asarray(normalized.data, dtype=np.complex128), "Sparse operation CSR data")

    dimension = int(normalized.shape[0])
    if dimension == 0 or dimension & (dimension - 1):
        raise ValueError("Sparse operation matrix dimension must be a power of two.")

    num_wires = int(np.log2(dimension))
    if num_wires <= 1:
        return normalized

    permutation = np.array([_reverse_bits(index, num_wires) for index in range(dimension)])
    return normalized[permutation, :][:, permutation].tocsr()


def statevector_to_little_endian_wires(statevector: object) -> np.ndarray:
    """Convert a full wire-ordered statevector to rocQuantum's little-endian basis."""
    normalized = as_complex_vector(statevector, "Statevector amplitudes").reshape(-1)
    dimension = normalized.shape[0]
    if dimension == 0 or dimension & (dimension - 1):
        raise ValueError("Statevector length must be a power of two.")

    num_wires = int(np.log2(dimension))
    if num_wires <= 1:
        return np.ascontiguousarray(normalized)

    permutation = np.array([_reverse_bits(index, num_wires) for index in range(dimension)])
    return np.ascontiguousarray(normalized[permutation])


def samples_to_binary_rows(raw_samples: Sequence[int], num_wires: int) -> np.ndarray:
    return np.array(
        [[(int(sample) >> bit) & 1 for bit in range(num_wires)] for sample in raw_samples],
        dtype=int,
    )


def sample_rows_from_statevector(statevector: Sequence[complex], shots: int, rng=None) -> np.ndarray:
    shots = normalize_shots(shots)
    state = as_complex_vector(statevector, "Statevector amplitudes").reshape(-1)
    probabilities = np.abs(state) ** 2
    total = float(np.sum(probabilities))
    if total <= 0.0:
        raise ValueError("Statevector probabilities sum to zero.")
    probabilities = probabilities / total

    if rng is None:
        rng = np.random
    counts = rng.multinomial(shots, probabilities)
    sample_indices = np.repeat(np.arange(len(probabilities)), counts)
    num_wires = int(np.log2(len(probabilities))) if len(probabilities) else 0
    return samples_to_binary_rows(sample_indices, num_wires)


def _as_real_probability_array(values: object, label: str) -> np.ndarray:
    try:
        raw = np.asarray(values, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain finite real numeric values.") from exc

    normalized = []
    for value in raw.reshape(-1):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(f"{label} must contain finite real numeric values.")
        probability = float(value)
        if not math.isfinite(probability):
            raise ValueError(f"{label} must contain finite values.")
        normalized.append(probability)
    return np.asarray(normalized, dtype=float).reshape(raw.shape)


def normalize_probability_vector(probabilities: Sequence[float], label: str = "Probability vector") -> np.ndarray:
    probabilities = _as_real_probability_array(probabilities, label).reshape(-1)
    if probabilities.size == 0:
        raise ValueError(f"{label} cannot be empty.")
    if np.any(probabilities < -1.0e-12):
        raise ValueError(f"{label} must be non-negative.")
    return np.ascontiguousarray(np.clip(probabilities, 0.0, None))


def normalize_probability_matrix(probabilities: Sequence[Sequence[float]], label: str = "Batched probability vector") -> np.ndarray:
    probabilities = _as_real_probability_array(probabilities, label)
    if probabilities.ndim != 2:
        raise ValueError(f"{label} must be a two-dimensional array.")
    if probabilities.shape[0] == 0:
        return np.empty((0, probabilities.shape[1]), dtype=float)
    return np.vstack(
        [normalize_probability_vector(row, label) for row in probabilities]
    ).astype(float, copy=False)


def sample_indices_from_probabilities(probabilities: Sequence[float], shots: int, rng=None) -> np.ndarray:
    shots = normalize_shots(shots)
    probabilities = normalize_probability_vector(probabilities, "Sampler probabilities")
    total = float(np.sum(probabilities))
    if total <= 0.0:
        raise ValueError("Sampler probabilities sum to zero.")
    probabilities = probabilities / total

    if rng is None:
        rng = np.random
    counts = rng.multinomial(shots, probabilities)
    return np.repeat(np.arange(probabilities.size, dtype=np.int64), counts)


def sample_indices_batch_from_probabilities(probabilities: Sequence[Sequence[float]], shots: int, rng=None) -> np.ndarray:
    shots = normalize_shots(shots)
    probabilities = normalize_probability_matrix(probabilities, "Batched sampler probabilities")
    if probabilities.shape[0] == 0:
        return np.empty((0, shots), dtype=np.int64)
    return np.vstack(
        [sample_indices_from_probabilities(row, shots, rng=rng) for row in probabilities]
    ).astype(np.int64, copy=False)


def probabilities_from_statevector(
    statevector: Sequence[complex],
    qubits: Iterable[int] | None = None,
) -> np.ndarray:
    state = as_complex_vector(statevector, "Statevector amplitudes").reshape(-1)
    probabilities = np.abs(state) ** 2
    total = float(np.sum(probabilities))
    if total <= 0.0:
        raise ValueError("Statevector probabilities sum to zero.")
    probabilities = probabilities / total

    num_qubits = int(np.log2(state.size)) if state.size else 0
    if state.size != (1 << num_qubits):
        raise ValueError("Statevector length must be a power of two.")

    if qubits is None:
        normalized_qubits = list(range(num_qubits))
    else:
        normalized_qubits = normalize_targets(qubits)

    if not normalized_qubits or normalized_qubits == list(range(num_qubits)):
        return np.ascontiguousarray(probabilities)

    marginal = np.zeros(1 << len(normalized_qubits), dtype=float)
    for basis_index, probability in enumerate(probabilities):
        marginal_index = 0
        for output_bit, qubit in enumerate(normalized_qubits):
            if (int(basis_index) >> int(qubit)) & 1:
                marginal_index |= 1 << output_bit
        marginal[marginal_index] += float(probability)
    return np.ascontiguousarray(marginal)


def qiskit_memory_from_samples(
    raw_samples: Sequence[int],
    measured_items: Sequence[tuple[int, int]],
    memory_width: int,
    sample_offsets: Sequence[int] | None = None,
) -> list[str]:
    if memory_width <= 0:
        memory_width = len(measured_items)
    if sample_offsets is None:
        sample_offsets = list(range(len(measured_items)))

    memory = []
    for sample in raw_samples:
        bits = ["0"] * memory_width
        for packed_bit, (classical_bit, _) in zip(sample_offsets, measured_items):
            output_index = memory_width - 1 - int(classical_bit)
            if 0 <= output_index < memory_width:
                bits[output_index] = "1" if ((int(sample) >> packed_bit) & 1) else "0"
        memory.append("".join(bits))
    return memory


def qiskit_sample_plan(measured_items: Sequence[tuple[int, int]]) -> tuple[list[int], list[int]]:
    qubit_offsets = {}
    sample_qubits = []
    sample_offsets = []
    for _, qubit in measured_items:
        normalized_qubit = int(qubit)
        if normalized_qubit not in qubit_offsets:
            qubit_offsets[normalized_qubit] = len(sample_qubits)
            sample_qubits.append(normalized_qubit)
        sample_offsets.append(qubit_offsets[normalized_qubit])
    return sample_qubits, sample_offsets


def counts_from_memory(memory: Sequence[str]) -> dict[str, int]:
    return dict(Counter(memory))


def expectation_from_statevector(
    statevector: Sequence[complex],
    pauli_string: str,
    targets: Sequence[int],
) -> float:
    state = as_complex_vector(statevector, "Statevector amplitudes").reshape(-1)

    num_qubits = int(np.log2(state.size)) if state.size else 0
    if state.size != (1 << num_qubits):
        raise ValueError("Statevector length must be a power of two.")

    normalized_pauli_string, normalized_targets = normalize_pauli_expectation_payload(
        pauli_string,
        targets,
        num_qubits,
    )
    if not normalized_targets:
        return 1.0

    result = 0.0 + 0.0j
    for basis_index, amplitude in enumerate(state):
        partner_index = basis_index
        phase = 1.0 + 0.0j
        for pauli, target in zip(normalized_pauli_string, normalized_targets):
            bit = (basis_index >> target) & 1
            if pauli == "X":
                partner_index ^= 1 << target
            elif pauli == "Y":
                partner_index ^= 1 << target
                phase *= 1j if bit == 0 else -1j
            elif pauli == "Z":
                phase *= 1.0 if bit == 0 else -1.0
        result += np.conj(state[partner_index]) * phase * amplitude
    return float(np.real_if_close(result).real)


def expectation_matrix_from_statevector(
    statevector: Sequence[complex],
    matrix: object,
    targets: Sequence[int],
) -> complex:
    state = as_complex_vector(statevector, "Statevector amplitudes").reshape(-1)
    num_qubits = int(np.log2(state.size)) if state.size else 0
    if state.size != (1 << num_qubits):
        raise ValueError("Statevector length must be a power of two.")

    normalized_targets = normalize_targets(targets)
    if not normalized_targets:
        raise ValueError("Dense matrix expectation requires at least one target qubit.")
    if len(set(normalized_targets)) != len(normalized_targets):
        raise ValueError("Dense matrix expectation targets must be unique.")
    for target in normalized_targets:
        if target < 0 or target >= num_qubits:
            raise ValueError("Dense matrix expectation target is outside the statevector qubit range.")

    normalized_matrix = np.asarray(matrix, dtype=np.complex128)
    dimension = 1 << len(normalized_targets)
    if normalized_matrix.shape != (dimension, dimension):
        raise ValueError("Dense expectation matrix dimension must be 2^len(targets).")
    validate_finite_complex_array(normalized_matrix, "Dense expectation matrix")

    result = 0.0 + 0.0j
    for row_index, amplitude in enumerate(state):
        row_target = 0
        base_index = int(row_index)
        for output_bit, target in enumerate(normalized_targets):
            mask = 1 << target
            if row_index & mask:
                row_target |= 1 << output_bit
            base_index &= ~mask
        for col_target in range(dimension):
            col_index = base_index
            for output_bit, target in enumerate(normalized_targets):
                if (col_target >> output_bit) & 1:
                    col_index |= 1 << target
            result += np.conj(amplitude) * normalized_matrix[row_target, col_target] * state[col_index]
    return complex(result)


def sparse_hamiltonian_moments_from_statevector(
    statevector: Sequence[complex],
    data: object,
    indices: object,
    indptr: object,
    shape: Sequence[int],
) -> tuple[complex, complex]:
    state = as_complex_vector(statevector, "Statevector amplitudes").reshape(-1)
    normalized_data, normalized_indices, normalized_indptr, normalized_shape = normalize_sparse_hamiltonian_csr(
        data,
        indices,
        indptr,
        shape,
        state.size,
    )
    rows, cols = normalized_shape

    h_state = np.zeros_like(state)
    for row in range(rows):
        start = int(normalized_indptr[row])
        end = int(normalized_indptr[row + 1])
        if start > end:
            raise ValueError("Sparse Hamiltonian CSR indptr must be monotonic.")
        for offset in range(start, end):
            col = int(normalized_indices[offset])
            if col < 0 or col >= cols:
                raise ValueError("Sparse Hamiltonian CSR column index is out of bounds.")
            h_state[row] += normalized_data[offset] * state[col]
    mean = np.vdot(state, h_state)
    second_moment = np.vdot(h_state, h_state)
    return complex(mean), complex(second_moment)


def normalize_sparse_hamiltonian_csr(
    data: object,
    indices: object,
    indptr: object,
    shape: Iterable[int],
    state_dimension: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    try:
        raw_data = np.asarray(data, dtype=object).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError("Sparse Hamiltonian CSR data must contain finite numeric values.") from exc

    normalized_data_values = []
    for value in raw_data:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Number):
            raise ValueError("Sparse Hamiltonian CSR data must contain finite numeric values.")
        scalar = complex(value)
        if not math.isfinite(scalar.real) or not math.isfinite(scalar.imag):
            raise ValueError("Sparse Hamiltonian CSR data must contain finite values.")
        normalized_data_values.append(scalar)
    normalized_data = np.ascontiguousarray(
        np.asarray(normalized_data_values, dtype=np.complex128)
    )

    normalized_indices = _normalize_sparse_index_vector(indices, "indices", "Sparse Hamiltonian")
    normalized_indptr = _normalize_sparse_index_vector(indptr, "indptr", "Sparse Hamiltonian")
    rows, cols = _normalize_sparse_shape(shape, "Sparse Hamiltonian")
    expected_dimension = None if state_dimension is None else int(state_dimension)

    if expected_dimension is not None and (rows != expected_dimension or cols != expected_dimension):
        raise ValueError("Sparse Hamiltonian shape must match the statevector length.")
    if normalized_indptr.size != rows + 1:
        raise ValueError("Sparse Hamiltonian CSR indptr length must equal rows + 1.")
    if normalized_data.size != normalized_indices.size:
        raise ValueError("Sparse Hamiltonian CSR data and indices lengths must match.")
    if (
        normalized_indptr.size == 0
        or int(normalized_indptr[0]) != 0
        or int(normalized_indptr[-1]) != normalized_data.size
    ):
        raise ValueError("Sparse Hamiltonian CSR indptr must start at 0 and end at nnz.")
    if np.any(normalized_indptr[:-1] > normalized_indptr[1:]):
        raise ValueError("Sparse Hamiltonian CSR indptr must be monotonic.")
    if np.any(normalized_indices < 0) or np.any(normalized_indices >= cols):
        raise ValueError("Sparse Hamiltonian CSR column index is out of bounds.")

    return normalized_data, normalized_indices, normalized_indptr, (rows, cols)


def apply_sparse_matrix_to_statevector(
    statevector: Sequence[complex],
    data: object,
    indices: object,
    indptr: object,
    shape: Sequence[int],
    targets: Iterable[int],
    num_qubits: int,
) -> np.ndarray:
    state = as_complex_vector(statevector, "Statevector amplitudes").reshape(-1)
    normalized_data = np.asarray(data, dtype=np.complex128).reshape(-1)
    normalized_indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    normalized_indptr = np.asarray(indptr, dtype=np.int64).reshape(-1)
    normalized_shape = tuple(int(dim) for dim in shape)
    normalized_targets = normalize_targets(targets)
    dimension = 1 << int(num_qubits)

    if state.size != dimension:
        raise ValueError("Statevector length must match the simulator qubit count.")
    if len(normalized_shape) != 2 or normalized_shape[0] != normalized_shape[1]:
        raise ValueError("Sparse operation matrix must be square.")
    local_dimension = 1 << len(normalized_targets)
    if normalized_shape != (local_dimension, local_dimension):
        raise ValueError("Sparse operation matrix shape must match target wires.")
    if len(set(normalized_targets)) != len(normalized_targets):
        raise ValueError("Sparse operation targets must be unique.")
    if any(target < 0 or target >= int(num_qubits) for target in normalized_targets):
        raise ValueError("Sparse operation target is out of range.")
    if normalized_indptr.size != local_dimension + 1:
        raise ValueError("Sparse operation CSR indptr length must equal rows + 1.")
    if normalized_data.size != normalized_indices.size:
        raise ValueError("Sparse operation CSR data and indices lengths must match.")
    if normalized_indptr[0] != 0 or normalized_indptr[-1] != normalized_data.size:
        raise ValueError("Sparse operation CSR indptr must start at 0 and end at nnz.")
    validate_finite_complex_array(normalized_data, "Sparse operation CSR data")

    out = np.zeros_like(state)
    for row_index in range(dimension):
        local_row = 0
        base_index = int(row_index)
        for output_bit, target in enumerate(normalized_targets):
            mask = 1 << target
            if row_index & mask:
                local_row |= 1 << output_bit
            base_index &= ~mask

        row_start = int(normalized_indptr[local_row])
        row_end = int(normalized_indptr[local_row + 1])
        if row_end < row_start:
            raise ValueError("Sparse operation CSR indptr must be monotonic.")
        for offset in range(row_start, row_end):
            local_col = int(normalized_indices[offset])
            if local_col < 0 or local_col >= local_dimension:
                raise ValueError("Sparse operation CSR column index is out of bounds.")
            col_index = base_index
            for output_bit, target in enumerate(normalized_targets):
                if (local_col >> output_bit) & 1:
                    col_index |= 1 << target
            out[row_index] += normalized_data[offset] * state[col_index]
    return np.ascontiguousarray(out)


class RocQuantumRuntime:
    """Small adapter around the public rocquantum_bind simulator surface."""

    def __init__(self, simulator):
        self.simulator = simulator
        self.preserve_global_phase = True

    @classmethod
    def from_bindings(cls, num_qubits: int, binding_module=None, batch_size: int = 1):
        if binding_module is None:
            try:
                import rocquantum_bind as binding_module
            except ImportError as exc:
                raise ImportError(
                    "rocquantum_bind is required for framework integrations. "
                    "Build/install rocQuantum with ROCQUANTUM_BUILD_BINDINGS=ON."
                ) from exc

        simulator_cls = getattr(binding_module, "QuantumSimulator", None)
        if simulator_cls is None:
            simulator_cls = getattr(binding_module, "QSim", None)
        if simulator_cls is None:
            raise ImportError("rocquantum_bind exposes neither QuantumSimulator nor QSim.")
        num_qubits = normalize_positive_integer(num_qubits, "num_qubits")
        batch_size = normalize_positive_integer(batch_size, "batch_size")
        if batch_size == 1:
            return cls(simulator_cls(num_qubits))
        return cls(simulator_cls(num_qubits, batch_size))

    def reset(self) -> None:
        reset = getattr(self.simulator, "reset", None)
        if callable(reset):
            reset()
            return
        self.simulator = type(self.simulator)(int(self.num_qubits()))

    def reset_qubit(self, target: int) -> None:
        native = getattr(self.simulator, "reset_qubit", None)
        if callable(native):
            native(int(target))
            return

        legacy = getattr(self.simulator, "ResetQubit", None)
        if callable(legacy):
            legacy(int(target))
            return

        raise NotImplementedError("The active rocQuantum binding does not expose qubit reset.")

    def measure_qubit(self, target: int) -> int:
        native = getattr(self.simulator, "measure_qubit", None)
        if callable(native):
            return int(native(int(target)))

        legacy = getattr(self.simulator, "MeasureQubit", None)
        if callable(legacy):
            return int(legacy(int(target)))

        raise NotImplementedError("The active rocQuantum binding does not expose state-collapsing qubit measurement.")

    def num_qubits(self) -> int:
        getter = getattr(self.simulator, "num_qubits", None)
        if callable(getter):
            return int(getter())
        return int(getattr(self.simulator, "num_qubits", 0))

    def batch_size(self) -> int:
        getter = getattr(self.simulator, "batch_size", None)
        if callable(getter):
            return int(getter())
        return int(getattr(self.simulator, "batch_size", 1))

    def apply_operation(
        self,
        name: str,
        targets: Iterable[int],
        params: Iterable[object] | None = None,
        matrix: object | None = None,
    ) -> None:
        if str(name).lower() in NO_OPS:
            return

        normalized_targets = normalize_targets(targets)
        normalized_params = normalize_params(params)
        normalized_name = normalize_gate_name(name)

        if normalized_name in {"UNITARY", "QUBITUNITARY"}:
            if matrix is None:
                raise ValueError(f"{name} requires a matrix.")
            self.apply_matrix(matrix, normalized_targets)
            return

        apply_gate = getattr(self.simulator, "apply_gate", None)
        if callable(apply_gate) and normalized_name in GATE_ALIASES.values():
            try:
                apply_gate(normalized_name, normalized_targets, normalized_params)
                return
            except (TypeError, RuntimeError, ValueError):
                if matrix is None:
                    raise

        if matrix is not None:
            self.apply_matrix(matrix, normalized_targets)
            return

        legacy_apply = getattr(self.simulator, "ApplyGate", None)
        if callable(legacy_apply) and normalized_name in GATE_ALIASES.values() and not normalized_params:
            if len(normalized_targets) == 1:
                legacy_apply(normalized_name, normalized_targets[0])
                return
            if len(normalized_targets) == 2:
                legacy_apply(normalized_name, normalized_targets[0], normalized_targets[1])
                return

        if matrix is not None:
            self.apply_matrix(matrix, normalized_targets)
            return

        raise NotImplementedError(f"Operation {name!r} is not supported by rocQuantum bindings.")

    def apply_operation_batch(
        self,
        name: str,
        targets: Iterable[int],
        params_by_batch: Iterable[object],
    ) -> None:
        normalized_targets = normalize_targets(targets)
        normalized_params = normalize_params(params_by_batch)
        normalized_name = normalize_gate_name(name)
        if len(normalized_params) != self.batch_size():
            raise ValueError("Batch parameter count must equal the simulator batch size.")

        batched_parametric_ops = {"RX", "RY", "RZ", "P", "CRX", "CRY", "CRZ", "CP"}

        apply_gate_batch = getattr(self.simulator, "apply_gate_batch", None)
        if callable(apply_gate_batch) and normalized_name in batched_parametric_ops:
            apply_gate_batch(normalized_name, normalized_targets, normalized_params)
            return

        legacy_apply_gate_batch = getattr(self.simulator, "ApplyGateBatch", None)
        if callable(legacy_apply_gate_batch) and normalized_name in batched_parametric_ops:
            legacy_apply_gate_batch(normalized_name, normalized_targets, normalized_params)
            return

        if self.batch_size() == 1:
            self.apply_operation(normalized_name, normalized_targets, [normalized_params[0]])
            return

        first_param = normalized_params[0]
        if all(param == first_param for param in normalized_params):
            self.apply_operation(normalized_name, normalized_targets, [first_param])
            return

        raise NotImplementedError("The active rocQuantum binding does not expose batched parameter gates.")

    def apply_matrix(self, matrix: object, targets: Iterable[int]) -> None:
        normalized_matrix, normalized_targets = operation_matrix_for_targets(
            matrix,
            targets,
            self.num_qubits(),
            "Operation matrix",
        )
        apply_matrix = getattr(self.simulator, "apply_matrix", None)
        if callable(apply_matrix):
            apply_matrix(normalized_matrix, normalized_targets)
            return

        legacy_apply = getattr(self.simulator, "ApplyGate", None)
        if callable(legacy_apply) and len(normalized_targets) == 1:
            legacy_apply(normalized_matrix, normalized_targets[0])
            return

        raise NotImplementedError("The active rocQuantum binding does not expose matrix application.")

    def apply_controlled_matrix(
        self,
        matrix: object,
        controls: Iterable[int],
        targets: Iterable[int],
    ) -> None:
        normalized_controls = validate_operation_targets(
            controls,
            self.num_qubits(),
            "Controlled operation controls",
        )
        normalized_matrix, normalized_targets = operation_matrix_for_targets(
            matrix,
            targets,
            self.num_qubits(),
            "Controlled operation matrix",
        )
        if set(normalized_controls).intersection(normalized_targets):
            raise ValueError("Controlled operation controls and targets must be disjoint.")

        native = getattr(self.simulator, "apply_controlled_matrix", None)
        if callable(native):
            native(normalized_matrix, normalized_controls, normalized_targets)
            return

        legacy = getattr(self.simulator, "ApplyControlledGate", None)
        if callable(legacy) and len(normalized_controls) == 1 and len(normalized_targets) == 1:
            legacy(normalized_matrix, normalized_controls[0], normalized_targets[0])
            return

        raise NotImplementedError("The active rocQuantum binding does not expose controlled matrix application.")

    def apply_sparse_matrix(
        self,
        data: object,
        indices: object,
        indptr: object,
        shape: Iterable[int],
        targets: Iterable[int],
    ) -> None:
        (
            normalized_data,
            normalized_indices,
            normalized_indptr,
            normalized_shape,
            normalized_targets,
        ) = normalize_sparse_operation_csr(
            data,
            indices,
            indptr,
            shape,
            targets,
            self.num_qubits(),
        )

        native = getattr(self.simulator, "apply_sparse_matrix", None)
        if callable(native):
            native(normalized_data, normalized_indices, normalized_indptr, normalized_shape, normalized_targets)
            return

        legacy = getattr(self.simulator, "ApplySparseMatrix", None)
        if callable(legacy):
            legacy(normalized_data, normalized_indices, normalized_indptr, normalized_shape, normalized_targets)
            return

        if self.batch_size() == 1:
            self.set_statevector(
                apply_sparse_matrix_to_statevector(
                    self.statevector(),
                    normalized_data,
                    normalized_indices,
                    normalized_indptr,
                    normalized_shape,
                    normalized_targets,
                    self.num_qubits(),
                )
            )
            return

        self.set_statevectors(
            [
                apply_sparse_matrix_to_statevector(
                    state,
                    normalized_data,
                    normalized_indices,
                    normalized_indptr,
                    normalized_shape,
                    normalized_targets,
                    self.num_qubits(),
                )
                for state in self.statevectors()
            ]
        )

    def supports_adjoint_jacobian(self) -> bool:
        return callable(getattr(self.simulator, "adjoint_jacobian", None)) or callable(
            getattr(self.simulator, "AdjointJacobian", None)
        )

    def adjoint_jacobian(
        self,
        operations: Sequence[dict],
        observables: Sequence[Sequence[dict]],
        trainable_params: Sequence[int],
    ):
        def _native_adjoint_unavailable(exc: Exception) -> bool:
            message = str(exc)
            return isinstance(exc, NotImplementedError) or "status 5" in message

        unavailable_error = None

        native = getattr(self.simulator, "adjoint_jacobian", None)
        if callable(native):
            try:
                return native(list(operations), list(observables), [int(param) for param in trainable_params])
            except Exception as exc:
                if not _native_adjoint_unavailable(exc):
                    raise
                unavailable_error = exc

        legacy = getattr(self.simulator, "AdjointJacobian", None)
        if callable(legacy):
            try:
                return legacy(list(operations), list(observables), [int(param) for param in trainable_params])
            except Exception as exc:
                if not _native_adjoint_unavailable(exc):
                    raise
                unavailable_error = exc

        if unavailable_error is not None:
            raise NotImplementedError("Native adjoint Jacobian is unavailable for this payload.") from unavailable_error
        raise NotImplementedError("The active rocQuantum binding does not expose native adjoint Jacobian.")

    def set_statevector(self, statevector: object) -> None:
        normalized_state = as_complex_vector(statevector, "Statevector amplitudes").reshape(-1)
        expected_size = 1 << self.num_qubits()
        if normalized_state.size != expected_size:
            raise ValueError("Statevector length must match the simulator qubit count.")
        setter = getattr(self.simulator, "set_statevector", None)
        if callable(setter):
            setter(normalized_state)
            return

        legacy_setter = getattr(self.simulator, "SetStateVector", None)
        if callable(legacy_setter):
            legacy_setter(normalized_state)
            return

        raise NotImplementedError("The active rocQuantum binding does not expose statevector upload.")

    def set_statevectors(self, statevectors: object) -> None:
        normalized_states = as_complex_vector(statevectors, "Statevector amplitudes").reshape(-1)
        expected_size = self.batch_size() * (1 << self.num_qubits())
        if normalized_states.size != expected_size:
            raise ValueError("Batched statevector length must match simulator batch and qubit count.")
        setter = getattr(self.simulator, "set_statevectors", None)
        if callable(setter):
            setter(normalized_states)
            return

        if self.batch_size() == 1:
            self.set_statevector(normalized_states)
            return

        raise NotImplementedError("The active rocQuantum binding does not expose batched statevector upload.")

    def statevector(self, batch_index: int = 0) -> np.ndarray:
        getter = getattr(self.simulator, "get_statevector", None)
        if not callable(getter):
            getter = getattr(self.simulator, "GetStateVector", None)
        if not callable(getter):
            raise NotImplementedError("The active rocQuantum binding does not expose state readback.")
        if batch_index == 0:
            try:
                return as_complex_vector(getter(batch_index), "Statevector amplitudes").reshape(-1)
            except TypeError:
                return as_complex_vector(getter(), "Statevector amplitudes").reshape(-1)
        return as_complex_vector(getter(batch_index), "Statevector amplitudes").reshape(-1)

    def statevectors(self) -> np.ndarray:
        getter = getattr(self.simulator, "get_statevectors", None)
        if not callable(getter):
            getter = getattr(self.simulator, "GetStateVectors", None)
        if callable(getter):
            states = as_complex_vector(getter(), "Statevector amplitudes").reshape(-1)
            return states.reshape(self.batch_size(), 1 << self.num_qubits())
        if self.batch_size() == 1:
            return self.statevector().reshape(1, -1)
        raise NotImplementedError("The active rocQuantum binding does not expose batched state readback.")

    def measure(self, qubits: Iterable[int], shots: int) -> list[int]:
        shots = normalize_shots(shots)
        measure = getattr(self.simulator, "measure", None)
        if not callable(measure):
            raise NotImplementedError("The active rocQuantum binding does not expose native sampling.")
        return [int(sample) for sample in measure(normalize_targets(qubits), shots)]

    def measure_batch(self, qubits: Iterable[int], shots: int) -> np.ndarray:
        normalized_qubits = normalize_targets(qubits)
        shots = normalize_shots(shots)

        native = getattr(self.simulator, "measure_batch", None)
        if not callable(native):
            native = getattr(self.simulator, "MeasureBatch", None)
        if callable(native):
            samples = np.asarray(native(normalized_qubits, shots), dtype=np.int64)
            return samples.reshape(self.batch_size(), shots)

        if self.batch_size() == 1:
            try:
                return np.asarray([self.measure(normalized_qubits, shots)], dtype=np.int64)
            except NotImplementedError:
                pass

        probabilities = self.probabilities_batch(normalized_qubits)
        return sample_indices_batch_from_probabilities(probabilities, shots)

    def probabilities(self, qubits: Iterable[int] | None = None) -> np.ndarray:
        normalized_qubits = None if qubits is None else normalize_targets(qubits)
        native_qubits = [] if normalized_qubits is None else normalized_qubits

        def _native_probabilities_unavailable(exc: Exception) -> bool:
            message = str(exc)
            return isinstance(exc, NotImplementedError) or "status 5" in message or "at most 20 target qubits" in message

        native = getattr(self.simulator, "probabilities", None)
        if callable(native):
            try:
                return normalize_probability_vector(native(native_qubits), "Probability vector")
            except Exception as exc:
                if not _native_probabilities_unavailable(exc):
                    raise

        legacy = getattr(self.simulator, "Probabilities", None)
        if callable(legacy):
            try:
                return normalize_probability_vector(legacy(native_qubits), "Probability vector")
            except Exception as exc:
                if not _native_probabilities_unavailable(exc):
                    raise

        return probabilities_from_statevector(self.statevector(), normalized_qubits)

    def probabilities_batch(self, qubits: Iterable[int] | None = None) -> np.ndarray:
        normalized_qubits = None if qubits is None else normalize_targets(qubits)
        native_qubits = [] if normalized_qubits is None else normalized_qubits

        def _native_probabilities_unavailable(exc: Exception) -> bool:
            message = str(exc)
            return isinstance(exc, NotImplementedError) or "status 5" in message or "at most 20 target qubits" in message

        native = getattr(self.simulator, "probabilities_batch", None)
        if callable(native):
            try:
                probabilities = _as_real_probability_array(
                    native(native_qubits),
                    "Batched probability vector",
                ).reshape(self.batch_size(), -1)
                return normalize_probability_matrix(probabilities, "Batched probability vector")
            except Exception as exc:
                if not _native_probabilities_unavailable(exc):
                    raise

        legacy = getattr(self.simulator, "ProbabilitiesBatch", None)
        if callable(legacy):
            try:
                probabilities = _as_real_probability_array(
                    legacy(native_qubits),
                    "Batched probability vector",
                ).reshape(self.batch_size(), -1)
                return normalize_probability_matrix(probabilities, "Batched probability vector")
            except Exception as exc:
                if not _native_probabilities_unavailable(exc):
                    raise

        if self.batch_size() == 1:
            return self.probabilities(normalized_qubits).reshape(1, -1)

        return np.vstack(
            [probabilities_from_statevector(state, normalized_qubits) for state in self.statevectors()]
        )

    def expectation_value(self, pauli: str, target: int) -> float:
        normalized_pauli, normalized_targets = normalize_pauli_expectation_payload(
            pauli,
            [target],
            None if self.num_qubits() <= 0 else self.num_qubits(),
        )
        native = getattr(self.simulator, "expectation_value", None)
        if callable(native):
            return float(native(normalized_pauli, normalized_targets[0]))

        legacy = getattr(self.simulator, "GetExpectationValue", None)
        if callable(legacy):
            return float(legacy(normalized_pauli, normalized_targets[0]))

        return self.expectation_pauli_string(normalized_pauli, normalized_targets)

    def expectation_pauli_string(self, pauli_string: str, targets: Iterable[int]) -> float:
        normalized_pauli, normalized_targets = normalize_pauli_expectation_payload(
            pauli_string,
            targets,
            None if self.num_qubits() <= 0 else self.num_qubits(),
        )

        native = getattr(self.simulator, "expectation_pauli_string", None)
        if callable(native):
            return float(native(normalized_pauli, normalized_targets))

        legacy = getattr(self.simulator, "GetExpectationPauliString", None)
        if callable(legacy):
            return float(legacy(normalized_pauli, normalized_targets))

        return expectation_from_statevector(self.statevector(), normalized_pauli, normalized_targets)

    def expectation_pauli_string_batch(self, pauli_string: str, targets: Iterable[int]) -> np.ndarray:
        normalized_pauli, normalized_targets = normalize_pauli_expectation_payload(
            pauli_string,
            targets,
            None if self.num_qubits() <= 0 else self.num_qubits(),
        )

        def _native_expectation_unavailable(exc: Exception) -> bool:
            message = str(exc)
            return isinstance(exc, NotImplementedError) or "status 5" in message

        native = getattr(self.simulator, "expectation_pauli_string_batch", None)
        if callable(native):
            try:
                return np.asarray(native(normalized_pauli, normalized_targets), dtype=float).reshape(self.batch_size())
            except Exception as exc:
                if not _native_expectation_unavailable(exc):
                    raise

        legacy = getattr(self.simulator, "GetExpectationPauliStringBatch", None)
        if callable(legacy):
            try:
                return np.asarray(legacy(normalized_pauli, normalized_targets), dtype=float).reshape(self.batch_size())
            except Exception as exc:
                if not _native_expectation_unavailable(exc):
                    raise

        if self.batch_size() == 1:
            return np.asarray([self.expectation_pauli_string(normalized_pauli, normalized_targets)], dtype=float)

        return np.asarray(
            [
                expectation_from_statevector(state, normalized_pauli, normalized_targets)
                for state in self.statevectors()
            ],
            dtype=float,
        )

    def _native_expectation_matrix(self, normalized_matrix, normalized_targets):
        def _native_expectation_unavailable(exc: Exception) -> bool:
            message = str(exc)
            return isinstance(exc, NotImplementedError) or "status 5" in message

        native = getattr(self.simulator, "expectation_matrix", None)
        if callable(native):
            try:
                return complex(native(normalized_matrix, normalized_targets))
            except Exception as exc:
                if not _native_expectation_unavailable(exc):
                    raise

        legacy = getattr(self.simulator, "ExpectationMatrix", None)
        if callable(legacy):
            try:
                return complex(legacy(normalized_matrix, normalized_targets))
            except Exception as exc:
                if not _native_expectation_unavailable(exc):
                    raise

        return None

    def _native_expectation_matrix_moments(self, normalized_matrix, normalized_targets):
        def _native_expectation_unavailable(exc: Exception) -> bool:
            message = str(exc)
            return isinstance(exc, NotImplementedError) or "status 5" in message

        native = getattr(self.simulator, "expectation_matrix_moments", None)
        if callable(native):
            try:
                mean, second_moment = native(normalized_matrix, normalized_targets)
                return complex(mean), complex(second_moment)
            except Exception as exc:
                if not _native_expectation_unavailable(exc):
                    raise

        legacy = getattr(self.simulator, "ExpectationMatrixMoments", None)
        if callable(legacy):
            try:
                mean, second_moment = legacy(normalized_matrix, normalized_targets)
                return complex(mean), complex(second_moment)
            except Exception as exc:
                if not _native_expectation_unavailable(exc):
                    raise

        return None

    def expectation_matrix(self, matrix: object, targets: Iterable[int]) -> complex:
        normalized_targets = normalize_targets(targets)
        normalized_matrix = as_complex_matrix(matrix, "Dense expectation matrix")

        native_result = self._native_expectation_matrix(normalized_matrix, normalized_targets)
        if native_result is not None:
            return native_result

        if self.batch_size() != 1:
            raise NotImplementedError("Use expectation_matrix_batch() for batched dense matrix fallback.")

        return expectation_matrix_from_statevector(self.statevector(), normalized_matrix, normalized_targets)

    def expectation_matrix_moments(self, matrix: object, targets: Iterable[int]) -> tuple[complex, complex]:
        normalized_targets = normalize_targets(targets)
        normalized_matrix = as_complex_matrix(matrix, "Dense expectation matrix")

        native_moments = self._native_expectation_matrix_moments(normalized_matrix, normalized_targets)
        if native_moments is not None:
            return native_moments

        squared_matrix = np.ascontiguousarray(normalized_matrix @ normalized_matrix)
        mean = self._native_expectation_matrix(normalized_matrix, normalized_targets)
        second_moment = (
            self._native_expectation_matrix(squared_matrix, normalized_targets) if mean is not None else None
        )
        if mean is not None and second_moment is not None:
            return mean, second_moment

        if self.batch_size() != 1:
            raise NotImplementedError("Use expectation_matrix_batch() for batched dense matrix moments.")

        state = self.statevector()
        return (
            expectation_matrix_from_statevector(state, normalized_matrix, normalized_targets),
            expectation_matrix_from_statevector(state, squared_matrix, normalized_targets),
        )

    def _native_expectation_matrix_batch(self, normalized_matrix, normalized_targets):
        def _native_expectation_unavailable(exc: Exception) -> bool:
            message = str(exc)
            return isinstance(exc, NotImplementedError) or "status 5" in message

        native = getattr(self.simulator, "expectation_matrix_batch", None)
        if callable(native):
            try:
                return np.asarray(native(normalized_matrix, normalized_targets), dtype=np.complex128).reshape(
                    self.batch_size()
                )
            except Exception as exc:
                if not _native_expectation_unavailable(exc):
                    raise

        legacy = getattr(self.simulator, "ExpectationMatrixBatch", None)
        if callable(legacy):
            try:
                return np.asarray(legacy(normalized_matrix, normalized_targets), dtype=np.complex128).reshape(
                    self.batch_size()
                )
            except Exception as exc:
                if not _native_expectation_unavailable(exc):
                    raise

        return None

    def expectation_matrix_batch(self, matrix: object, targets: Iterable[int]) -> np.ndarray:
        normalized_targets = normalize_targets(targets)
        normalized_matrix = as_complex_matrix(matrix, "Dense expectation matrix")

        native_result = self._native_expectation_matrix_batch(normalized_matrix, normalized_targets)
        if native_result is not None:
            return native_result

        if self.batch_size() == 1:
            return np.asarray([self.expectation_matrix(normalized_matrix, normalized_targets)], dtype=np.complex128)

        return np.asarray(
            [
                expectation_matrix_from_statevector(state, normalized_matrix, normalized_targets)
                for state in self.statevectors()
            ],
            dtype=np.complex128,
        )

    def expectation_matrix_moments_batch(self, matrix: object, targets: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
        normalized_targets = normalize_targets(targets)
        normalized_matrix = as_complex_matrix(matrix, "Dense expectation matrix")

        def _native_expectation_unavailable(exc: Exception) -> bool:
            message = str(exc)
            return isinstance(exc, NotImplementedError) or "status 5" in message

        native = getattr(self.simulator, "expectation_matrix_moments_batch", None)
        if callable(native):
            try:
                means, second_moments = native(normalized_matrix, normalized_targets)
                return (
                    np.asarray(means, dtype=np.complex128).reshape(self.batch_size()),
                    np.asarray(second_moments, dtype=np.complex128).reshape(self.batch_size()),
                )
            except Exception as exc:
                if not _native_expectation_unavailable(exc):
                    raise

        legacy = getattr(self.simulator, "ExpectationMatrixMomentsBatch", None)
        if callable(legacy):
            try:
                means, second_moments = legacy(normalized_matrix, normalized_targets)
                return (
                    np.asarray(means, dtype=np.complex128).reshape(self.batch_size()),
                    np.asarray(second_moments, dtype=np.complex128).reshape(self.batch_size()),
                )
            except Exception as exc:
                if not _native_expectation_unavailable(exc):
                    raise

        squared_matrix = np.ascontiguousarray(normalized_matrix @ normalized_matrix)
        native_means = self._native_expectation_matrix_batch(normalized_matrix, normalized_targets)
        native_second_moments = self._native_expectation_matrix_batch(squared_matrix, normalized_targets)
        if native_means is not None and native_second_moments is not None:
            return native_means, native_second_moments

        if self.batch_size() == 1:
            mean, second_moment = self.expectation_matrix_moments(normalized_matrix, normalized_targets)
            return (
                np.asarray([mean], dtype=np.complex128),
                np.asarray([second_moment], dtype=np.complex128),
            )

        states = self.statevectors()
        return (
            np.asarray(
                [
                    expectation_matrix_from_statevector(state, normalized_matrix, normalized_targets)
                    for state in states
                ],
                dtype=np.complex128,
            ),
            np.asarray(
                [
                    expectation_matrix_from_statevector(state, squared_matrix, normalized_targets)
                    for state in states
                ],
                dtype=np.complex128,
            ),
        )

    def sparse_hamiltonian_moments(
        self,
        data: object,
        indices: object,
        indptr: object,
        shape: Iterable[int],
    ) -> tuple[complex, complex]:
        num_qubits = self.num_qubits()
        normalized_data, normalized_indices, normalized_indptr, normalized_shape = normalize_sparse_hamiltonian_csr(
            data,
            indices,
            indptr,
            shape,
            None if num_qubits <= 0 else 1 << num_qubits,
        )

        def _native_sparse_unavailable(exc: Exception) -> bool:
            message = str(exc)
            return isinstance(exc, NotImplementedError) or "status 5" in message

        native = getattr(self.simulator, "sparse_hamiltonian_moments", None)
        if callable(native):
            try:
                mean, second_moment = native(
                    normalized_data,
                    normalized_indices,
                    normalized_indptr,
                    normalized_shape,
                )
                return complex(mean), complex(second_moment)
            except Exception as exc:
                if not _native_sparse_unavailable(exc):
                    raise

        legacy = getattr(self.simulator, "SparseHamiltonianMoments", None)
        if callable(legacy):
            try:
                mean, second_moment = legacy(
                    normalized_data,
                    normalized_indices,
                    normalized_indptr,
                    normalized_shape,
                )
                return complex(mean), complex(second_moment)
            except Exception as exc:
                if not _native_sparse_unavailable(exc):
                    raise

        if self.batch_size() != 1:
            raise NotImplementedError("Use sparse_hamiltonian_moments_batch() for batched sparse fallback.")

        return sparse_hamiltonian_moments_from_statevector(
            self.statevector(),
            normalized_data,
            normalized_indices,
            normalized_indptr,
            normalized_shape,
        )

    def sparse_hamiltonian_moments_batch(
        self,
        data: object,
        indices: object,
        indptr: object,
        shape: Iterable[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        num_qubits = self.num_qubits()
        normalized_data, normalized_indices, normalized_indptr, normalized_shape = normalize_sparse_hamiltonian_csr(
            data,
            indices,
            indptr,
            shape,
            None if num_qubits <= 0 else 1 << num_qubits,
        )

        def _native_sparse_unavailable(exc: Exception) -> bool:
            message = str(exc)
            return isinstance(exc, NotImplementedError) or "status 5" in message

        native = getattr(self.simulator, "sparse_hamiltonian_moments_batch", None)
        if callable(native):
            try:
                means, second_moments = native(
                    normalized_data,
                    normalized_indices,
                    normalized_indptr,
                    normalized_shape,
                )
                return (
                    np.asarray(means, dtype=np.complex128).reshape(self.batch_size()),
                    np.asarray(second_moments, dtype=np.complex128).reshape(self.batch_size()),
                )
            except Exception as exc:
                if not _native_sparse_unavailable(exc):
                    raise

        legacy = getattr(self.simulator, "SparseHamiltonianMomentsBatch", None)
        if callable(legacy):
            try:
                means, second_moments = legacy(
                    normalized_data,
                    normalized_indices,
                    normalized_indptr,
                    normalized_shape,
                )
                return (
                    np.asarray(means, dtype=np.complex128).reshape(self.batch_size()),
                    np.asarray(second_moments, dtype=np.complex128).reshape(self.batch_size()),
                )
            except Exception as exc:
                if not _native_sparse_unavailable(exc):
                    raise

        means = []
        second_moments = []
        for state in self.statevectors():
            mean, second_moment = sparse_hamiltonian_moments_from_statevector(
                state,
                normalized_data,
                normalized_indices,
                normalized_indptr,
                normalized_shape,
            )
            means.append(mean)
            second_moments.append(second_moment)
        return np.asarray(means, dtype=np.complex128), np.asarray(second_moments, dtype=np.complex128)
