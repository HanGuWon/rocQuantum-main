from __future__ import annotations

import math
import os
import warnings
from numbers import Integral, Real
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .operator import HermitianOperator, PauliOperator, SparseHamiltonianOperator, SumOperator, iter_pauli_terms

try:
    import _rocq_hip_backend as hip_backend
except ImportError:
    hip_backend = None

try:
    import rocq_hip as dm_backend
except ImportError:
    dm_backend = None


_MOCK_ENV_VAR = "ROCQ_ENABLE_MOCK_BACKENDS"
_DISABLE_FUSION_ENV_VAR = "ROCQ_DISABLE_GATE_FUSION"
_FUSABLE_SINGLE_QUBIT_GATES = {"x", "y", "z", "h", "s", "t", "rx", "ry", "rz"}
_DISTRIBUTED_RUNTIME_SWITCHES = (
    "ROCQ_DISTRIBUTED_COMM",
    "ROCQ_REQUIRE_RCCL",
    "ROCQ_DISABLE_RCCL",
    "ROCQ_DISTRIBUTED_FALLBACK_MODE",
    "ROCQ_ENABLE_DISTRIBUTED_HOST_FALLBACK",
)
_DISTRIBUTED_SUPPORTED_FEATURES = (
    "distributed handle/allocation metadata",
    "local-domain state-vector gate application",
    "swap-localized non-local named single/control/CNOT/CZ gates",
    "swap-localized non-local MCX/CSWAP gates",
    "swap-localized 1-4 target dense matrix application",
    "local-domain sparse matrix application",
    "local-domain selected-qubit sampling/probabilities",
    "local-domain Pauli, dense-matrix, and rank-local CSR expectation reductions",
    "optional RCCL all-reduce/send-recv paths when native bindings are built with RCCL",
    "explicit slow/debug host fallback when enabled",
)
_DISTRIBUTED_UNSUPPORTED_FEATURES = (
    "multi-node distributed allocation",
    "production-grade multi-GPU performance parity",
    "general non-local controlled dense matrices",
    "general non-local sparse matrix application",
    "non-local arbitrary sparse expectation reductions",
    "slice-domain measured-bit sampling/probabilities beyond covered swap-localized paths",
    "distributed scheduler or multi-QPU async execution",
    "local hardware proof without self-hosted ROCm CI artifacts",
)
_MOCK_BACKEND_NOTE = (
    "{backend_name} is using the Python mock fallback because {env_var}=1 and the "
    "native ROCm binding is unavailable. This path is for local smoke tests only; "
    "it does not validate native ROCm/cuQuantum-style execution or performance."
)
_PAULI_MUL_TABLE = {
    ("I", "I"): (1, "I"),
    ("I", "X"): (1, "X"),
    ("I", "Y"): (1, "Y"),
    ("I", "Z"): (1, "Z"),
    ("X", "I"): (1, "X"),
    ("X", "X"): (1, "I"),
    ("X", "Y"): (1j, "Z"),
    ("X", "Z"): (-1j, "Y"),
    ("Y", "I"): (1, "Y"),
    ("Y", "X"): (-1j, "Z"),
    ("Y", "Y"): (1, "I"),
    ("Y", "Z"): (1j, "X"),
    ("Z", "I"): (1, "Z"),
    ("Z", "X"): (1j, "Y"),
    ("Z", "Y"): (-1j, "X"),
    ("Z", "Z"): (1, "I"),
}


class MockBackendWarning(RuntimeWarning):
    """Raised when a canonical backend falls back to the Python mock implementation."""


def _mock_backends_enabled() -> bool:
    return os.environ.get(_MOCK_ENV_VAR, "").strip().lower() in {"1", "true", "yes", "on"}


def _warn_mock_backend(backend_name: str) -> None:
    warnings.warn(
        _MOCK_BACKEND_NOTE.format(backend_name=backend_name, env_var=_MOCK_ENV_VAR),
        MockBackendWarning,
        stacklevel=3,
    )


def distributed_capabilities() -> Dict[str, object]:
    """Return the advertised canonical distributed runtime contract."""

    return {
        "status": "partial",
        "native_binding_available": hip_backend is not None,
        "native_backend_query_available": (
            hip_backend is not None
            and hasattr(hip_backend, "get_distributed_backend")
        ),
        "runtime_switches": list(_DISTRIBUTED_RUNTIME_SWITCHES),
        "supported_features": list(_DISTRIBUTED_SUPPORTED_FEATURES),
        "unsupported_features": list(_DISTRIBUTED_UNSUPPORTED_FEATURES),
        "guide": "rocquantum/src/hipStateVec/MULTI_GPU_GUIDE.md",
        "performance_note": (
            "Distributed support is experimental and correctness-oriented; "
            "ROCm multi-GPU performance proof requires self-hosted ROCm CI or real hardware."
        ),
    }


def _native_backend_error(module_name: str, backend_name: str) -> RuntimeError:
    return RuntimeError(
        f"The '{backend_name}' backend requires the native Python module '{module_name}', "
        "but it is not installed or importable in this environment. "
        f"Build/install rocQuantum with native bindings, or set {_MOCK_ENV_VAR}=1 only for tests."
    )


def _coerce_complex64_matrix(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix, dtype=np.complex64, order="C")


def _format_sample_counts(raw_results: Iterable[int], width: int) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for outcome in raw_results:
        bitstring = format(int(outcome), f"0{width}b")
        counts[bitstring] = counts.get(bitstring, 0) + 1
    return counts


def _validate_positive_integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer.")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be positive.")
    return normalized


def _validate_optional_boolean_option(value, name: str) -> Optional[bool]:
    if value is not None and not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean when provided.")
    return value


def _validate_backend_name(backend_name, supported: Dict[str, type]) -> str:
    if not isinstance(backend_name, str):
        raise ValueError(f"backend_name must be one of: {list(supported.keys())}.")
    if backend_name not in supported:
        raise ValueError(f"Unsupported backend '{backend_name}'. Supported backends are: {list(supported.keys())}")
    return backend_name


def _validate_gate_angle(op_name: str, angle) -> float:
    if isinstance(angle, bool) or not isinstance(angle, Real):
        raise ValueError(f"Gate '{op_name}' angle must be a finite real number.")
    value = float(angle)
    if not math.isfinite(value):
        raise ValueError(f"Gate '{op_name}' angle must be finite.")
    return value


def _validate_probability(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be between 0 and 1.")
    probability = float(value)
    if not math.isfinite(probability) or probability < 0.0 or probability > 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")
    return probability


def _normalize_sample_qubits(qubits, num_qubits: int) -> List[int]:
    if qubits is None:
        return list(range(int(num_qubits)))
    if isinstance(qubits, (str, bytes)):
        raise TypeError("sample qubits must be a sequence of integer indices.")

    try:
        raw_qubits = list(qubits)
    except TypeError as exc:
        raise TypeError("sample qubits must be a sequence of integer indices.") from exc

    if not raw_qubits:
        raise ValueError("sample qubits must include at least one qubit.")

    normalized = []
    for qubit in raw_qubits:
        if isinstance(qubit, bool) or not isinstance(qubit, Integral):
            raise ValueError("sample qubits must be integer indices.")
        index = int(qubit)
        if index < 0 or index >= int(num_qubits):
            raise ValueError(f"Sample qubit index {index} is out of range for {num_qubits} qubits.")
        normalized.append(index)

    if len(set(normalized)) != len(normalized):
        raise ValueError("sample qubits must be unique.")
    return normalized


def _normalize_gate_targets(
    op_name: str,
    targets,
    num_qubits: int,
    *,
    exact_count: Optional[int] = None,
    min_count: Optional[int] = None,
) -> List[int]:
    if isinstance(targets, bool) or isinstance(targets, (str, bytes)):
        raise TypeError("Gate targets must be a sequence of integer qubit indices.")
    try:
        raw_targets = list(targets)
    except TypeError as exc:
        raise TypeError("Gate targets must be a sequence of integer qubit indices.") from exc

    if exact_count is not None and len(raw_targets) != exact_count:
        raise ValueError(f"Gate '{op_name}' expects {exact_count} target(s).")
    if min_count is not None and len(raw_targets) < min_count:
        raise ValueError(f"Gate '{op_name}' expects at least {min_count} target(s).")
    if not raw_targets:
        raise ValueError("Gate targets must include at least one qubit.")

    normalized = []
    for target in raw_targets:
        if isinstance(target, bool) or not isinstance(target, Integral):
            raise ValueError("Gate target must be an integer qubit index.")
        index = int(target)
        if index < 0 or index >= int(num_qubits):
            raise ValueError(f"Gate target index {index} is out of range for {num_qubits} qubits.")
        normalized.append(index)

    if len(set(normalized)) != len(normalized):
        raise ValueError(f"Gate '{op_name}' target qubits must be distinct.")
    return normalized


def _finalize_expectation(value: complex):
    return value.real if abs(value.imag) < 1e-12 else value


def _combined_pauli_terms(operator, num_qubits: Optional[int] = None):
    qubit_count = None if num_qubits is None else int(num_qubits)
    combined = {}
    order = []
    for coefficient, paulis in iter_pauli_terms(operator):
        key_terms = []
        for pauli, qubit in paulis:
            qubit_index = int(qubit)
            if qubit_count is not None and (
                qubit_index < 0 or qubit_index >= qubit_count
            ):
                raise ValueError(
                    f"Pauli observable qubit index {qubit_index} is out of range "
                    f"for {qubit_count} qubits."
                )
            key_terms.append((pauli, qubit_index))
        key = tuple(key_terms)
        if key not in combined:
            combined[key] = 0.0 + 0.0j
            order.append(key)
        combined[key] += coefficient
    return [
        (combined[key], list(key))
        for key in order
        if combined[key] != 0
    ]


def _multiply_pauli_strings(
    left_phase: complex,
    left: Sequence[str],
    right_phase: complex,
    right: Sequence[str],
) -> tuple[complex, tuple[str, ...]]:
    if len(left) != len(right):
        raise ValueError("Pauli strings must have the same length.")
    phase = left_phase * right_phase
    output = []
    for lhs, rhs in zip(left, right):
        factor, pauli = _PAULI_MUL_TABLE[(lhs, rhs)]
        phase *= factor
        output.append(pauli)
    return phase, tuple(output)


def _matrix_expectation_cache_key(operator, num_qubits: int):
    if isinstance(operator, HermitianOperator):
        matrix, targets = _normalize_matrix_targets(operator.matrix, operator.targets, int(num_qubits))
        matrix = np.ascontiguousarray(matrix, dtype=np.complex128)
        return (
            "hermitian",
            tuple(int(target) for target in targets),
            matrix.dtype.str,
            matrix.shape,
            matrix.tobytes(),
        )

    if isinstance(operator, SparseHamiltonianOperator):
        data, indices, indptr, shape = _normalize_sparse_hamiltonian(operator, int(num_qubits))
        data = np.ascontiguousarray(data, dtype=np.complex128)
        indices = np.ascontiguousarray(indices, dtype=np.uintp)
        indptr = np.ascontiguousarray(indptr, dtype=np.uintp)
        return (
            "sparse",
            tuple(int(value) for value in shape),
            data.dtype.str,
            data.shape,
            data.tobytes(),
            indices.dtype.str,
            indices.shape,
            indices.tobytes(),
            indptr.dtype.str,
            indptr.shape,
            indptr.tobytes(),
        )

    return None


def _expect_mixed_sum_operator(operator: SumOperator, num_qubits: int, expectation_func):
    if operator.coefficient == 0:
        return 0.0

    total = 0.0 + 0.0j
    expectation_cache = {}
    pauli_terms = []
    for term in operator.terms:
        if getattr(term, "coefficient", None) == 0:
            continue

        cache_key = _matrix_expectation_cache_key(term, int(num_qubits))
        if cache_key is None:
            try:
                pauli_terms.extend(iter_pauli_terms(term))
                continue
            except (NotImplementedError, TypeError):
                value = expectation_func(term)
        else:
            if cache_key not in expectation_cache:
                expectation_cache[cache_key] = expectation_func(term * (1 / term.coefficient))
            value = term.coefficient * expectation_cache[cache_key]
        total += operator.coefficient * value

    if pauli_terms:
        pauli_operators = [
            PauliOperator(
                "I" if not paulis else " ".join(f"{pauli}{int(qubit)}" for pauli, qubit in paulis),
                coefficient=coefficient,
            )
            for coefficient, paulis in pauli_terms
            if coefficient != 0
        ]
        if pauli_operators:
            pauli_operator = pauli_operators[0] if len(pauli_operators) == 1 else SumOperator(pauli_operators)
            total += operator.coefficient * expectation_func(pauli_operator)
    return _finalize_expectation(total)


def _normalize_matrix_targets(matrix, targets, num_qubits: int):
    matrix_array = np.asarray(matrix, dtype=np.complex128)
    if matrix_array.ndim != 2 or matrix_array.shape[0] != matrix_array.shape[1]:
        raise ValueError("HermitianOperator matrix must be square.")

    matrix_dim = int(matrix_array.shape[0])
    if matrix_dim <= 0 or matrix_dim & (matrix_dim - 1):
        raise ValueError("HermitianOperator matrix dimension must be a power of two.")

    target_count = int(math.log2(matrix_dim))
    if targets is None:
        if target_count != int(num_qubits):
            raise ValueError(
                "HermitianOperator targets must be provided when the matrix does not span all qubits."
            )
        normalized_targets = list(range(int(num_qubits)))
    else:
        normalized_targets = [int(target) for target in targets]
        if len(normalized_targets) != target_count:
            raise ValueError("HermitianOperator target count must match matrix dimension.")
        if len(set(normalized_targets)) != len(normalized_targets):
            raise ValueError("HermitianOperator targets must be unique.")
        if any(target < 0 or target >= int(num_qubits) for target in normalized_targets):
            raise ValueError("HermitianOperator target is out of range for the backend.")

    return matrix_array, normalized_targets


def _statevector_expectation_matrix(statevector, matrix, targets: Sequence[int], num_qubits: int):
    state = np.asarray(statevector, dtype=np.complex128).reshape(-1)
    expected_state_dim = 1 << int(num_qubits)
    if state.size != expected_state_dim:
        raise ValueError("Statevector dimension does not match backend qubit count.")

    target_count = len(targets)
    matrix_dim = 1 << target_count
    total = 0.0 + 0.0j
    for row_index in range(expected_state_dim):
        row_target = 0
        base_index = row_index
        for bit, qubit in enumerate(targets):
            mask = 1 << int(qubit)
            if row_index & mask:
                row_target |= 1 << bit
            base_index &= ~mask

        accum = 0.0 + 0.0j
        for col_target in range(matrix_dim):
            col_index = base_index
            for bit, qubit in enumerate(targets):
                if (col_target >> bit) & 1:
                    col_index |= 1 << int(qubit)
            accum += matrix[row_target, col_target] * state[col_index]
        total += np.conj(state[row_index]) * accum

    return total


def _statevector_pauli_expectation(statevector, paulis: Sequence[tuple[str, int]], num_qubits: int):
    state = np.asarray(statevector, dtype=np.complex128).reshape(-1)
    expected_state_dim = 1 << int(num_qubits)
    if state.size != expected_state_dim:
        raise ValueError("Statevector dimension does not match backend qubit count.")

    transformed = np.zeros_like(state)
    for basis_index, amplitude in enumerate(state):
        if amplitude == 0:
            continue
        output_index = int(basis_index)
        phase = 1.0 + 0.0j
        for pauli, qubit in paulis:
            qubit_index = int(qubit)
            if qubit_index < 0 or qubit_index >= int(num_qubits):
                raise ValueError(
                    f"Pauli observable qubit index {qubit_index} is out of range "
                    f"for {num_qubits} qubits."
                )
            mask = 1 << qubit_index
            bit_set = bool(basis_index & mask)
            if pauli == "X":
                output_index ^= mask
            elif pauli == "Y":
                output_index ^= mask
                phase *= -1j if bit_set else 1j
            elif pauli == "Z":
                phase *= -1.0 if bit_set else 1.0
            else:
                raise NotImplementedError(f"Unsupported Pauli observable '{pauli}'.")
        transformed[output_index] += phase * amplitude
    return np.vdot(state, transformed)


def _density_matrix_expectation_matrix(density_matrix, matrix, targets: Sequence[int], num_qubits: int):
    density = np.asarray(density_matrix, dtype=np.complex128)
    expected_dim = 1 << int(num_qubits)
    if density.shape != (expected_dim, expected_dim):
        raise ValueError("Density matrix dimension does not match backend qubit count.")

    target_count = len(targets)
    matrix_dim = 1 << target_count
    total = 0.0 + 0.0j
    for row_index in range(expected_dim):
        row_target = 0
        base_index = row_index
        for bit, qubit in enumerate(targets):
            mask = 1 << int(qubit)
            if row_index & mask:
                row_target |= 1 << bit
            base_index &= ~mask

        for col_target in range(matrix_dim):
            col_index = base_index
            for bit, qubit in enumerate(targets):
                if (col_target >> bit) & 1:
                    col_index |= 1 << int(qubit)
            total += matrix[row_target, col_target] * density[col_index, row_index]

    return total


def _normalize_sparse_hamiltonian(operator: SparseHamiltonianOperator, num_qubits: int):
    data = np.asarray(operator.data, dtype=np.complex128).reshape(-1)
    indices = np.asarray(operator.indices, dtype=np.int64).reshape(-1)
    indptr = np.asarray(operator.indptr, dtype=np.int64).reshape(-1)
    rows, cols = (int(operator.shape[0]), int(operator.shape[1]))
    state_dim = 1 << int(num_qubits)

    if rows != state_dim or cols != state_dim:
        raise ValueError("SparseHamiltonianOperator shape must match the backend state dimension.")
    if data.size != indices.size:
        raise ValueError("SparseHamiltonianOperator CSR data and indices lengths must match.")
    if indptr.size != rows + 1:
        raise ValueError("SparseHamiltonianOperator CSR indptr length must equal rows + 1.")
    if indptr.size == 0 or int(indptr[0]) != 0 or int(indptr[-1]) != data.size:
        raise ValueError("SparseHamiltonianOperator CSR indptr must start at 0 and end at nnz.")
    if np.any(indptr[:-1] > indptr[1:]):
        raise ValueError("SparseHamiltonianOperator CSR indptr must be monotonic.")
    if np.any(indices < 0) or np.any(indices >= cols):
        raise ValueError("SparseHamiltonianOperator CSR column index is out of bounds.")

    return data, indices.astype(np.uintp), indptr.astype(np.uintp), (rows, cols)


def _density_matrix_sparse_hamiltonian_expectation(density_matrix, data, indices, indptr):
    density = np.asarray(density_matrix, dtype=np.complex128)
    rows, cols = density.shape
    if rows != cols:
        raise ValueError("Density matrix must be square.")

    total = 0.0 + 0.0j
    for row in range(rows):
        for offset in range(int(indptr[row]), int(indptr[row + 1])):
            col = int(indices[offset])
            total += data[offset] * density[col, row]
    return total


def _normalize_channel_targets(targets, num_qubits: int) -> List[int]:
    if isinstance(targets, bool) or isinstance(targets, (str, bytes)):
        raise TypeError("Kraus channel targets must be integer indices.")
    if isinstance(targets, Integral):
        raw_targets = [targets]
    else:
        try:
            raw_targets = list(targets)
        except TypeError as exc:
            raise TypeError("Kraus channel targets must be integer indices.") from exc

    if not raw_targets:
        raise ValueError("Kraus channel must target at least one qubit.")

    normalized = []
    for target in raw_targets:
        if isinstance(target, bool) or not isinstance(target, Integral):
            raise ValueError("Kraus channel targets must be integer indices.")
        normalized.append(int(target))
    if len(set(normalized)) != len(normalized):
        raise ValueError("Kraus channel targets must be unique.")
    if any(target < 0 or target >= int(num_qubits) for target in normalized):
        raise ValueError("Kraus channel target is out of range for the backend.")
    return normalized


def _normalize_kraus_matrices(kraus_matrices, targets: Sequence[int]) -> np.ndarray:
    matrices = np.asarray(kraus_matrices, dtype=np.complex64, order="C")
    target_dim = 1 << len(targets)

    if matrices.ndim == 2:
        matrices = matrices.reshape(1, matrices.shape[0], matrices.shape[1])
    if matrices.ndim != 3:
        raise ValueError("Kraus matrices must have shape (num_kraus, dim, dim).")
    if matrices.shape[0] <= 0:
        raise ValueError("Kraus channel must include at least one matrix.")
    if matrices.shape[1:] != (target_dim, target_dim):
        raise ValueError(
            "Kraus matrix dimensions must equal 2**len(targets) for the selected channel targets."
        )
    return np.ascontiguousarray(matrices, dtype=np.complex64)


def _probability_mixed_kraus_matrices(kraus_matrices, targets: Sequence[int], probability: float) -> np.ndarray:
    if kraus_matrices is None:
        raise ValueError("Kraus noise channels require kraus_matrices.")
    matrices = _normalize_kraus_matrices(kraus_matrices, targets)
    prob = _validate_probability(probability, "Kraus channel probability")
    if prob == 1.0:
        return matrices

    target_dim = 1 << len(targets)
    identity = math.sqrt(1.0 - prob) * np.eye(target_dim, dtype=np.complex64)
    scaled = math.sqrt(prob) * matrices
    return np.ascontiguousarray(np.concatenate([identity.reshape(1, target_dim, target_dim), scaled], axis=0))


def _basis_parts(index: int, targets: Sequence[int]) -> Tuple[int, int]:
    base_index = int(index)
    target_index = 0
    for bit, qubit in enumerate(targets):
        mask = 1 << int(qubit)
        if index & mask:
            target_index |= 1 << bit
        base_index &= ~mask
    return base_index, target_index


def _embed_target_bits(base_index: int, target_index: int, targets: Sequence[int]) -> int:
    result = int(base_index)
    for bit, qubit in enumerate(targets):
        if (int(target_index) >> bit) & 1:
            result |= 1 << int(qubit)
    return result


def _apply_density_kraus_host(density_matrix, targets: Sequence[int], kraus_matrices) -> np.ndarray:
    density = np.asarray(density_matrix, dtype=np.complex128)
    dim = density.shape[0]
    target_dim = 1 << len(targets)
    output = np.zeros_like(density, dtype=np.complex128)
    matrices = np.asarray(kraus_matrices, dtype=np.complex128)

    for matrix in matrices:
        for row in range(dim):
            row_base, row_target = _basis_parts(row, targets)
            for col in range(dim):
                col_base, col_target = _basis_parts(col, targets)
                total = 0.0 + 0.0j
                for local_row in range(target_dim):
                    source_row = _embed_target_bits(row_base, local_row, targets)
                    left = matrix[row_target, local_row]
                    if left == 0:
                        continue
                    for local_col in range(target_dim):
                        source_col = _embed_target_bits(col_base, local_col, targets)
                        total += left * density[source_row, source_col] * np.conj(matrix[col_target, local_col])
                output[row, col] += total
    return output.astype(np.complex64)


def _statevector_sparse_hamiltonian_moments(statevector, data, indices, indptr):
    state = np.asarray(statevector, dtype=np.complex128).reshape(-1)
    h_state = np.zeros_like(state)
    for row in range(state.size):
        accum = 0.0 + 0.0j
        for offset in range(int(indptr[row]), int(indptr[row + 1])):
            accum += data[offset] * state[int(indices[offset])]
        h_state[row] = accum

    mean = np.vdot(state, h_state)
    second_moment = np.vdot(h_state, h_state)
    return mean, second_moment


class _MockStateVectorState:
    def __init__(self, n_qubits: int):
        self._num_qubits = n_qubits
        self._state = np.zeros(1 << n_qubits, dtype=np.complex64)
        self._state[0] = 1.0 + 0.0j

    def _validate_qubit(self, qubit: int) -> int:
        if isinstance(qubit, bool) or not isinstance(qubit, Integral):
            raise ValueError("Gate target must be an integer qubit index.")
        normalized = int(qubit)
        if normalized < 0 or normalized >= int(self._num_qubits):
            raise ValueError(f"Qubit index {normalized} is out of range for {self._num_qubits} qubits.")
        return normalized

    def _validate_qubits(self, qubits: Sequence[int], name: str) -> List[int]:
        normalized = [self._validate_qubit(qubit) for qubit in qubits]
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"{name} must be unique.")
        return normalized

    def _angle(self, op_name: str, params: Optional[Dict[str, float]], primary_key: str, secondary_key: str) -> float:
        if isinstance(params, dict):
            angle = params.get(primary_key, params.get(secondary_key))
        elif isinstance(params, (list, tuple)) and params:
            angle = params[0]
        else:
            angle = None
        if angle is None:
            raise ValueError(f"Gate '{op_name}' requires a rotation angle.")
        return _validate_gate_angle(op_name, angle)

    def _single_qubit_matrix(self, op_name: str, params: Optional[Dict[str, float]]) -> np.ndarray:
        op = op_name.lower()
        if op == "h":
            return (1.0 / math.sqrt(2.0)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)
        if op == "x":
            return np.array([[0, 1], [1, 0]], dtype=np.complex128)
        if op == "y":
            return np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        if op == "z":
            return np.array([[1, 0], [0, -1]], dtype=np.complex128)
        if op == "s":
            return np.array([[1, 0], [0, 1j]], dtype=np.complex128)
        if op == "sdg":
            return np.array([[1, 0], [0, -1j]], dtype=np.complex128)
        if op == "t":
            phase = math.pi / 4.0
            return np.array([[1, 0], [0, complex(math.cos(phase), math.sin(phase))]], dtype=np.complex128)
        if op in {"tdg", "tdag"}:
            phase = -math.pi / 4.0
            return np.array([[1, 0], [0, complex(math.cos(phase), math.sin(phase))]], dtype=np.complex128)
        if op == "rx":
            angle = self._angle(op_name, params, "theta", "phi")
            c = math.cos(angle / 2.0)
            s = math.sin(angle / 2.0)
            return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)
        if op == "ry":
            angle = self._angle(op_name, params, "theta", "phi")
            c = math.cos(angle / 2.0)
            s = math.sin(angle / 2.0)
            return np.array([[c, -s], [s, c]], dtype=np.complex128)
        if op == "rz":
            angle = self._angle(op_name, params, "phi", "theta")
            c = math.cos(angle / 2.0)
            s = math.sin(angle / 2.0)
            return np.array([[c - 1j * s, 0], [0, c + 1j * s]], dtype=np.complex128)
        if op in {"p", "phase"}:
            angle = self._angle(op_name, params, "phi", "theta")
            return np.array([[1, 0], [0, complex(math.cos(angle), math.sin(angle))]], dtype=np.complex128)
        raise ValueError(f"Gate '{op_name}' is not supported by the mock state-vector backend.")

    def _basis_parts(self, index: int, targets: Sequence[int]) -> tuple[int, int]:
        base_index = int(index)
        target_index = 0
        for bit, qubit in enumerate(targets):
            mask = 1 << int(qubit)
            if index & mask:
                target_index |= 1 << bit
            base_index &= ~mask
        return base_index, target_index

    def _embed_target_bits(self, base_index: int, target_index: int, targets: Sequence[int]) -> int:
        output = int(base_index)
        for bit, qubit in enumerate(targets):
            if (int(target_index) >> bit) & 1:
                output |= 1 << int(qubit)
        return output

    def _apply_matrix_to_targets(self, targets: Sequence[int], matrix: np.ndarray) -> None:
        targets = self._validate_qubits(targets, "matrix targets")
        if not targets:
            raise ValueError("matrix targets must include at least one qubit.")
        target_dim = 1 << len(targets)
        matrix = np.asarray(matrix, dtype=np.complex128)
        if matrix.shape != (target_dim, target_dim):
            raise ValueError(f"Matrix shape must be ({target_dim}, {target_dim}) for selected targets.")

        old_state = np.asarray(self._state, dtype=np.complex128)
        new_state = np.zeros_like(old_state)
        for basis_index, amplitude in enumerate(old_state):
            if amplitude == 0:
                continue
            base_index, col_target = self._basis_parts(basis_index, targets)
            for row_target in range(target_dim):
                output_index = self._embed_target_bits(base_index, row_target, targets)
                new_state[output_index] += matrix[row_target, col_target] * amplitude
        self._state = new_state.astype(np.complex64)

    def _apply_controlled_matrix_to_targets(
        self,
        controls: Sequence[int],
        targets: Sequence[int],
        matrix: np.ndarray,
    ) -> None:
        controls = self._validate_qubits(controls, "control qubits")
        targets = self._validate_qubits(targets, "target qubits")
        if set(controls).intersection(targets):
            raise ValueError("control qubits and target qubits must be disjoint.")

        target_dim = 1 << len(targets)
        matrix = np.asarray(matrix, dtype=np.complex128)
        if matrix.shape != (target_dim, target_dim):
            raise ValueError(f"Matrix shape must be ({target_dim}, {target_dim}) for selected targets.")

        control_mask = sum(1 << control for control in controls)
        old_state = np.asarray(self._state, dtype=np.complex128)
        new_state = old_state.copy()
        for basis_index, amplitude in enumerate(old_state):
            if amplitude == 0 or (basis_index & control_mask) != control_mask:
                continue
            base_index, col_target = self._basis_parts(basis_index, targets)
            new_state[basis_index] -= amplitude
            for row_target in range(target_dim):
                output_index = self._embed_target_bits(base_index, row_target, targets)
                new_state[output_index] += matrix[row_target, col_target] * amplitude
        self._state = new_state.astype(np.complex64)

    def _apply_mcx(self, controls: Sequence[int], target: int) -> None:
        controls = self._validate_qubits(controls, "control qubits")
        target = self._validate_qubit(target)
        if target in controls:
            raise ValueError("control qubits and target qubit must be disjoint.")
        self._apply_controlled_matrix_to_targets(controls, [target], self._single_qubit_matrix("x", None))

    def _apply_cswap(self, control: int, target_a: int, target_b: int) -> None:
        control = self._validate_qubit(control)
        target_a = self._validate_qubit(target_a)
        target_b = self._validate_qubit(target_b)
        if len({control, target_a, target_b}) != 3:
            raise ValueError("CSWAP control and targets must be distinct.")
        swap_matrix = np.array(
            [
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.complex128,
        )
        self._apply_controlled_matrix_to_targets([control], [target_a, target_b], swap_matrix)

    def apply_named_gate(self, op_name: str, targets: List[int], params: Optional[Dict[str, float]] = None):
        params = params or {}
        op = op_name.lower()
        if op in {"h", "x", "y", "z", "s", "sdg", "t", "tdg", "tdag", "rx", "ry", "rz", "p", "phase"}:
            if len(targets) != 1:
                raise ValueError(f"Gate '{op_name}' expects one target.")
            self._apply_matrix_to_targets([targets[0]], self._single_qubit_matrix(op_name, params))
            return None
        if op in {"cnot", "cx"}:
            if len(targets) != 2:
                raise ValueError(f"Gate '{op_name}' expects [control, target].")
            self._apply_mcx([targets[0]], targets[1])
            return None
        if op == "cz":
            if len(targets) != 2:
                raise ValueError(f"Gate '{op_name}' expects [control, target].")
            self._apply_controlled_matrix_to_targets([targets[0]], [targets[1]], self._single_qubit_matrix("z", None))
            return None
        if op == "swap":
            if len(targets) != 2:
                raise ValueError(f"Gate '{op_name}' expects two targets.")
            swap_matrix = np.array(
                [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                dtype=np.complex128,
            )
            self._apply_matrix_to_targets(targets, swap_matrix)
            return None
        if op in {"crx", "cry", "crz", "cp", "cphase"}:
            if len(targets) != 2:
                raise ValueError(f"Gate '{op_name}' expects [control, target].")
            base_name = "p" if op in {"cp", "cphase"} else op[1:]
            self._apply_controlled_matrix_to_targets(
                [targets[0]],
                [targets[1]],
                self._single_qubit_matrix(base_name, params),
            )
            return None
        if op in {"mcx", "ccx", "toffoli"}:
            if len(targets) < 2:
                raise ValueError(f"Gate '{op_name}' requires at least one control and one target.")
            self._apply_mcx(targets[:-1], targets[-1])
            return None
        if op in {"cswap", "fredkin"}:
            if len(targets) != 3:
                raise ValueError(f"Gate '{op_name}' expects [control, target_a, target_b].")
            self._apply_cswap(targets[0], targets[1], targets[2])
            return None
        raise ValueError(f"Gate '{op_name}' is not supported by the mock state-vector backend.")

    def apply_matrix(self, targets: Sequence[int], matrix: np.ndarray):
        self._apply_matrix_to_targets(targets, matrix)
        return None

    def apply_controlled_matrix(self, controls: Sequence[int], targets: Sequence[int], matrix: np.ndarray):
        self._apply_controlled_matrix_to_targets(controls, targets, matrix)
        return None

    def get_state_vector(self):
        return self._state.copy()

    def sample(self, measured_qubits: Sequence[int], num_shots: int):
        measured = self._validate_qubits(measured_qubits, "measured qubits")
        probs = np.zeros(1 << len(measured), dtype=np.float64)
        for basis_index, amplitude in enumerate(self._state):
            outcome = 0
            for bit, qubit in enumerate(measured):
                if (basis_index >> int(qubit)) & 1:
                    outcome |= 1 << bit
            probs[outcome] += float(abs(amplitude) ** 2)
        total = probs.sum()
        if total <= 0.0:
            raise RuntimeError("Statevector has no probability mass.")
        probs /= total
        return np.random.choice(len(probs), size=int(num_shots), p=probs).astype(np.uint64)

    def _pauli_expectation(self, paulis: Sequence[tuple[str, int]]) -> complex:
        return _statevector_pauli_expectation(self._state, paulis, self._num_qubits)

    def expectation(self, operator):
        if isinstance(operator, HermitianOperator):
            matrix, targets = _normalize_matrix_targets(operator.matrix, operator.targets, self._num_qubits)
            value = _statevector_expectation_matrix(self._state, matrix, targets, self._num_qubits)
            return _finalize_expectation(operator.coefficient * value)

        if isinstance(operator, SparseHamiltonianOperator):
            data, indices, indptr, _ = _normalize_sparse_hamiltonian(operator, self._num_qubits)
            mean, _ = _statevector_sparse_hamiltonian_moments(self._state, data, indices, indptr)
            return _finalize_expectation(operator.coefficient * mean)

        if isinstance(operator, SumOperator):
            try:
                terms = _combined_pauli_terms(operator, self._num_qubits)
            except NotImplementedError:
                return _expect_mixed_sum_operator(operator, self._num_qubits, self.expectation)
        else:
            terms = _combined_pauli_terms(operator, self._num_qubits)

        total = 0.0 + 0.0j
        for coefficient, paulis in terms:
            if not paulis:
                total += coefficient
            else:
                total += coefficient * self._pauli_expectation(paulis)
        return _finalize_expectation(total)


class _MockDensityMatrixState:
    def __init__(self, n_qubits: int):
        self._num_qubits = int(n_qubits)
        dim = 1 << n_qubits
        self._density = np.zeros((dim, dim), dtype=np.complex64)
        self._density[0, 0] = 1.0 + 0.0j

    def apply_gate_matrix(self, matrix: np.ndarray, target: int, adjoint: bool = False):
        return None

    def apply_controlled_gate(self, matrix: np.ndarray, control: int, target: int):
        return None

    def apply_cnot(self, control: int, target: int):
        return None

    def apply_depolarizing_channel(self, target: int, prob: float):
        return None

    def apply_bit_flip_channel(self, target: int, prob: float):
        return None

    def apply_phase_flip_channel(self, target: int, prob: float):
        return None

    def apply_amplitude_damping_channel(self, target: int, prob: float):
        return None

    def apply_channel(self, targets, kraus_matrices: np.ndarray):
        normalized_targets = _normalize_channel_targets(targets, self._num_qubits)
        matrices = _normalize_kraus_matrices(kraus_matrices, normalized_targets)
        self._density = _apply_density_kraus_host(self._density, normalized_targets, matrices)
        return None

    def compute_expectation(self, pauli, target: int):
        return 0.0

    def compute_z_product_expectation(self, targets: Sequence[int]):
        return 0.0

    def sample(self, measured_qubits: Sequence[int], num_shots: int):
        measured = [int(q) for q in measured_qubits]
        if not measured:
            raise ValueError("At least one qubit must be measured.")
        probs = np.zeros(1 << len(measured), dtype=np.float64)
        diagonal = np.real(np.diag(self._density))
        for basis, prob in enumerate(diagonal):
            outcome = 0
            for bit, qubit in enumerate(measured):
                if (basis >> qubit) & 1:
                    outcome |= 1 << bit
            probs[outcome] += max(float(prob), 0.0)
        total = probs.sum()
        if total <= 0.0:
            raise RuntimeError("Density matrix has no positive diagonal probability mass.")
        probs /= total
        return np.random.choice(len(probs), size=int(num_shots), p=probs).astype(np.uint64)

    def get_density_matrix(self):
        return self._density.copy()


class _HipStateVectorState:
    def __init__(self, n_qubits: int, enable_fusion: bool = True):
        if hip_backend is None:
            raise _native_backend_error("_rocq_hip_backend", "state_vector")

        self._handle = hip_backend.RocsvHandle()
        self._num_qubits = n_qubits
        self._d_state = hip_backend.allocate_state_internal(self._handle, n_qubits, 1)
        status = hip_backend.initialize_state(self._handle, self._d_state, n_qubits)
        if status != hip_backend.rocqStatus.SUCCESS:
            raise RuntimeError(f"rocsvInitializeState failed: {status}")

        self._fusion = None
        if enable_fusion and hasattr(hip_backend, "GateFusion"):
            self._fusion = hip_backend.GateFusion(self._handle, self._d_state, self._num_qubits)

    @property
    def handle(self):
        return self._handle

    @property
    def device_state(self):
        return self._d_state

    @property
    def num_qubits(self):
        return self._num_qubits

    def _call(self, op_name: str, func, *args):
        status = func(*args)
        if status != hip_backend.rocqStatus.SUCCESS:
            raise RuntimeError(f"hipStateVec call failed for '{op_name}': {status}")
        return status

    def _angle(self, op_name: str, params: Optional[Dict[str, float]], primary_key: str, secondary_key: str) -> float:
        if isinstance(params, dict):
            angle = params.get(primary_key, params.get(secondary_key))
        elif isinstance(params, (list, tuple)) and params:
            angle = params[0]
        else:
            angle = None
        if angle is None:
            raise ValueError(f"Gate '{op_name}' requires a rotation angle.")
        return _validate_gate_angle(op_name, angle)

    def apply_named_gate(self, op_name: str, targets: List[int], params: Optional[Dict[str, float]] = None):
        params = params or {}
        op = op_name.lower()

        if op == "h":
            targets = _normalize_gate_targets(op_name, targets, self._num_qubits, exact_count=1)
            return self._call(op_name, hip_backend.apply_h, self._handle, self._d_state, self._num_qubits, targets[0])
        if op == "x":
            targets = _normalize_gate_targets(op_name, targets, self._num_qubits, exact_count=1)
            return self._call(op_name, hip_backend.apply_x, self._handle, self._d_state, self._num_qubits, targets[0])
        if op == "y":
            targets = _normalize_gate_targets(op_name, targets, self._num_qubits, exact_count=1)
            return self._call(op_name, hip_backend.apply_y, self._handle, self._d_state, self._num_qubits, targets[0])
        if op == "z":
            targets = _normalize_gate_targets(op_name, targets, self._num_qubits, exact_count=1)
            return self._call(op_name, hip_backend.apply_z, self._handle, self._d_state, self._num_qubits, targets[0])
        if op == "s":
            targets = _normalize_gate_targets(op_name, targets, self._num_qubits, exact_count=1)
            return self._call(op_name, hip_backend.apply_s, self._handle, self._d_state, self._num_qubits, targets[0])
        if op == "sdg":
            targets = _normalize_gate_targets(op_name, targets, self._num_qubits, exact_count=1)
            return self._call(op_name, hip_backend.apply_sdg, self._handle, self._d_state, self._num_qubits, targets[0])
        if op == "t":
            targets = _normalize_gate_targets(op_name, targets, self._num_qubits, exact_count=1)
            return self._call(op_name, hip_backend.apply_t, self._handle, self._d_state, self._num_qubits, targets[0])
        if op in {"tdg", "tdag"}:
            targets = _normalize_gate_targets(op_name, targets, self._num_qubits, exact_count=1)
            return self._call(op_name, hip_backend.apply_tdg, self._handle, self._d_state, self._num_qubits, targets[0])
        if op == "rx":
            targets = _normalize_gate_targets(op_name, targets, self._num_qubits, exact_count=1)
            angle = self._angle(op_name, params, "theta", "phi")
            return self._call(op_name, hip_backend.apply_rx, self._handle, self._d_state, self._num_qubits, targets[0], angle)
        if op == "ry":
            targets = _normalize_gate_targets(op_name, targets, self._num_qubits, exact_count=1)
            angle = self._angle(op_name, params, "theta", "phi")
            return self._call(op_name, hip_backend.apply_ry, self._handle, self._d_state, self._num_qubits, targets[0], angle)
        if op == "rz":
            targets = _normalize_gate_targets(op_name, targets, self._num_qubits, exact_count=1)
            angle = self._angle(op_name, params, "phi", "theta")
            return self._call(op_name, hip_backend.apply_rz, self._handle, self._d_state, self._num_qubits, targets[0], angle)
        if op in {"p", "phase"}:
            targets = _normalize_gate_targets(op_name, targets, self._num_qubits, exact_count=1)
            angle = self._angle(op_name, params, "phi", "theta")
            return self._call(op_name, hip_backend.apply_p, self._handle, self._d_state, self._num_qubits, targets[0], angle)
        if op == "cnot":
            targets = _normalize_gate_targets(op_name, targets, self._num_qubits, exact_count=2)
            return self._call(op_name, hip_backend.apply_cnot, self._handle, self._d_state, self._num_qubits, targets[0], targets[1])
        if op == "cz":
            targets = _normalize_gate_targets(op_name, targets, self._num_qubits, exact_count=2)
            return self._call(op_name, hip_backend.apply_cz, self._handle, self._d_state, self._num_qubits, targets[0], targets[1])
        if op == "swap":
            targets = _normalize_gate_targets(op_name, targets, self._num_qubits, exact_count=2)
            return self._call(op_name, hip_backend.apply_swap, self._handle, self._d_state, self._num_qubits, targets[0], targets[1])
        if op == "crx":
            targets = _normalize_gate_targets(op_name, targets, self._num_qubits, exact_count=2)
            angle = self._angle(op_name, params, "theta", "phi")
            return self._call(op_name, hip_backend.apply_crx, self._handle, self._d_state, self._num_qubits, targets[0], targets[1], angle)
        if op == "cry":
            targets = _normalize_gate_targets(op_name, targets, self._num_qubits, exact_count=2)
            angle = self._angle(op_name, params, "theta", "phi")
            return self._call(op_name, hip_backend.apply_cry, self._handle, self._d_state, self._num_qubits, targets[0], targets[1], angle)
        if op == "crz":
            targets = _normalize_gate_targets(op_name, targets, self._num_qubits, exact_count=2)
            angle = self._angle(op_name, params, "phi", "theta")
            return self._call(op_name, hip_backend.apply_crz, self._handle, self._d_state, self._num_qubits, targets[0], targets[1], angle)
        if op in {"cp", "cphase"}:
            targets = _normalize_gate_targets(op_name, targets, self._num_qubits, exact_count=2)
            angle = self._angle(op_name, params, "phi", "theta")
            return self._call(op_name, hip_backend.apply_cp, self._handle, self._d_state, self._num_qubits, targets[0], targets[1], angle)
        if op == "mcx":
            targets = _normalize_gate_targets(op_name, targets, self._num_qubits, min_count=2)
            controls = targets[:-1]
            return self._call(op_name, hip_backend.apply_mcx, self._handle, self._d_state, self._num_qubits, controls, targets[-1])
        if op in {"ccx", "toffoli"}:
            targets = _normalize_gate_targets(op_name, targets, self._num_qubits, exact_count=3)
            controls = targets[:-1]
            return self._call(op_name, hip_backend.apply_mcx, self._handle, self._d_state, self._num_qubits, controls, targets[-1])
        if op == "cswap":
            targets = _normalize_gate_targets(op_name, targets, self._num_qubits, exact_count=3)
            return self._call(
                op_name,
                hip_backend.apply_cswap,
                self._handle,
                self._d_state,
                self._num_qubits,
                targets[0],
                targets[1],
                targets[2],
            )

        raise ValueError(f"Gate '{op_name}' is not supported by the hipStateVec backend.")

    def apply_matrix(self, targets: Sequence[int], matrix: np.ndarray):
        d_matrix = hip_backend.create_device_matrix_from_numpy(_coerce_complex64_matrix(matrix))
        return self._call(
            "apply_matrix",
            hip_backend.apply_matrix,
            self._handle,
            self._d_state,
            self._num_qubits,
            list(targets),
            d_matrix,
            1 << len(targets),
        )

    def apply_controlled_matrix(self, controls: Sequence[int], targets: Sequence[int], matrix: np.ndarray):
        d_matrix = hip_backend.create_device_matrix_from_numpy(_coerce_complex64_matrix(matrix))
        return self._call(
            "apply_controlled_matrix",
            hip_backend.apply_controlled_matrix,
            self._handle,
            self._d_state,
            self._num_qubits,
            list(controls),
            list(targets),
            d_matrix,
        )

    def get_state_vector(self):
        return hip_backend.get_state_vector_full(self._handle, self._d_state, self._num_qubits, 1)

    def sample(self, measured_qubits: Sequence[int], num_shots: int):
        return hip_backend.sample(
            self._handle,
            self._d_state,
            self._num_qubits,
            list(measured_qubits),
            int(num_shots),
        )

    def _expectation_matrix(self, operator: HermitianOperator):
        matrix, targets = _normalize_matrix_targets(operator.matrix, operator.targets, self._num_qubits)
        native = getattr(hip_backend, "get_expectation_matrix", None)
        if native is not None:
            value = native(
                self._handle,
                self._d_state,
                self._num_qubits,
                targets,
                _coerce_complex64_matrix(matrix),
            )
        else:
            value = _statevector_expectation_matrix(self.get_state_vector(), matrix, targets, self._num_qubits)
        return _finalize_expectation(operator.coefficient * value)

    def _sparse_hamiltonian_moments(self, operator: SparseHamiltonianOperator):
        data, indices, indptr, shape = _normalize_sparse_hamiltonian(operator, self._num_qubits)
        native = getattr(hip_backend, "get_sparse_matrix_moments", None)
        if native is not None:
            mean, second_moment = native(
                self._handle,
                self._d_state,
                self._num_qubits,
                np.asarray(data, dtype=np.complex64, order="C"),
                [int(index) for index in indices],
                [int(offset) for offset in indptr],
                shape[0],
                shape[1],
            )
        else:
            mean, second_moment = _statevector_sparse_hamiltonian_moments(
                self.get_state_vector(), data, indices, indptr
            )
        return mean, second_moment

    def expectation(self, operator):
        if isinstance(operator, HermitianOperator):
            return self._expectation_matrix(operator)

        if isinstance(operator, SparseHamiltonianOperator):
            mean, _ = self._sparse_hamiltonian_moments(operator)
            return _finalize_expectation(operator.coefficient * mean)

        if isinstance(operator, SumOperator):
            try:
                terms = _combined_pauli_terms(operator, self._num_qubits)
            except NotImplementedError:
                return _expect_mixed_sum_operator(operator, self._num_qubits, self.expectation)
        else:
            terms = _combined_pauli_terms(operator, self._num_qubits)

        total = 0.0 + 0.0j
        fallback_state = None

        def fallback_pauli_expectation(pauli_terms):
            nonlocal fallback_state
            if fallback_state is None:
                fallback_state = self.get_state_vector()
            return _statevector_pauli_expectation(
                fallback_state,
                pauli_terms,
                self._num_qubits,
            )

        for coefficient, paulis in terms:
            if not paulis:
                total += coefficient
                continue

            if len(paulis) == 1:
                pauli, qubit = paulis[0]
                if pauli == "X":
                    native = getattr(hip_backend, "get_expectation_value_x", None)
                elif pauli == "Y":
                    native = getattr(hip_backend, "get_expectation_value_y", None)
                elif pauli == "Z":
                    native = getattr(hip_backend, "get_expectation_value_z", None)
                else:
                    raise NotImplementedError(f"Unsupported Pauli observable '{pauli}'.")
                if callable(native):
                    value = native(self._handle, self._d_state, self._num_qubits, qubit)
                else:
                    value = fallback_pauli_expectation(paulis)
                total += coefficient * value
                continue

            if all(pauli == "Z" for pauli, _ in paulis):
                native = getattr(hip_backend, "get_expectation_value_pauli_product_z", None)
                if callable(native):
                    value = native(
                        self._handle,
                        self._d_state,
                        self._num_qubits,
                        [qubit for _, qubit in paulis],
                    )
                else:
                    value = fallback_pauli_expectation(paulis)
                total += coefficient * value
                continue

            pauli_string = "".join(pauli for pauli, _ in paulis)
            qubits = [qubit for _, qubit in paulis]
            native = getattr(hip_backend, "get_expectation_pauli_string", None)
            if callable(native):
                value = native(
                    self._handle,
                    self._d_state,
                    self._num_qubits,
                    pauli_string,
                    qubits,
                )
            else:
                value = fallback_pauli_expectation(paulis)
            total += coefficient * value

        return _finalize_expectation(total)

    def fusion_engine(self):
        return self._fusion


class _HipDensityMatrixState:
    def __init__(self, n_qubits: int):
        if dm_backend is None:
            raise _native_backend_error("rocq_hip", "density_matrix")
        self._state = dm_backend.DensityMatrixState(n_qubits)

    def apply_gate_matrix(self, matrix: np.ndarray, target: int, adjoint: bool = False):
        return self._state.apply_gate(_coerce_complex64_matrix(matrix), target, adjoint)

    def apply_controlled_gate(self, matrix: np.ndarray, control: int, target: int):
        return self._state.apply_controlled_gate(_coerce_complex64_matrix(matrix), control, target)

    def apply_cnot(self, control: int, target: int):
        return self._state.apply_cnot(control, target)

    def apply_depolarizing_channel(self, target: int, prob: float):
        return self._state.apply_depolarizing_channel(target, prob)

    def apply_bit_flip_channel(self, target: int, prob: float):
        return self._state.apply_bit_flip_channel(target, prob)

    def apply_phase_flip_channel(self, target: int, prob: float):
        return self._state.apply_phase_flip_channel(target, prob)

    def apply_amplitude_damping_channel(self, target: int, prob: float):
        return self._state.apply_amplitude_damping_channel(target, prob)

    def apply_channel(self, targets, kraus_matrices: np.ndarray):
        return self._state.apply_channel(targets, _coerce_complex64_matrix(kraus_matrices))

    def compute_expectation(self, pauli, target: int):
        return self._state.compute_expectation(pauli, target)

    def compute_z_product_expectation(self, targets: Sequence[int]):
        return self._state._compute_z_product_expectation(list(targets))

    def compute_expectation_matrix(self, matrix: np.ndarray, targets: Sequence[int]):
        if len(targets) > 4:
            raise NotImplementedError("Native density-matrix dense expectation supports at most four target qubits.")
        native = getattr(self._state, "compute_expectation_matrix", None)
        if not callable(native):
            raise NotImplementedError("The active density-matrix binding does not expose dense expectation.")
        return native(_coerce_complex64_matrix(matrix), list(targets))

    def sample(self, measured_qubits: Sequence[int], num_shots: int):
        return self._state.sample(list(measured_qubits), int(num_shots))

    def get_density_matrix(self):
        return self._state.get_density_matrix()


class _BaseBackend:
    """Abstract base class for a quantum simulation backend."""

    def __init__(self, num_qubits: int):
        self.num_qubits = _validate_positive_integer(num_qubits, "num_qubits")

    def run_ops(self, ops, noise_model=None):
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError

    def sample(self, shots: int, qubits: Optional[Sequence[int]] = None):
        raise NotImplementedError

    def expectation(self, operator):
        raise NotImplementedError

    def apply_noise(self, channel: str, targets: List[int], prob: float, kraus_matrices=None):
        raise NotImplementedError


class StabilizerBackend(_BaseBackend):
    """Small Clifford-only stabilizer/tableau backend for Pauli propagation."""

    _H = (1.0 / math.sqrt(2.0)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)
    _X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    _Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    _Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    _S = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
    _SDG = np.array([[1, 0], [0, -1j]], dtype=np.complex128)

    def __init__(self, num_qubits: int):
        super().__init__(num_qubits)
        self._state = np.zeros(1 << int(self.num_qubits), dtype=np.complex128)
        self._state[0] = 1.0 + 0.0j
        self._generators: list[tuple[complex, tuple[str, ...]]] = []
        for qubit in range(int(self.num_qubits)):
            paulis = ["I"] * int(self.num_qubits)
            paulis[qubit] = "Z"
            self._generators.append((1.0 + 0.0j, tuple(paulis)))

    def _validate_qubit(self, qubit: int) -> int:
        return _normalize_gate_targets(
            "stabilizer",
            [qubit],
            self.num_qubits,
            exact_count=1,
        )[0]

    def _single_pauli(self, qubit: int, pauli: str) -> tuple[str, ...]:
        out = ["I"] * int(self.num_qubits)
        out[int(qubit)] = pauli
        return tuple(out)

    def _apply_single_qubit_state(self, matrix: np.ndarray, target: int) -> None:
        target = self._validate_qubit(target)
        mask = 1 << target
        state = self._state.copy()
        for base in range(state.size):
            if base & mask:
                continue
            paired = base | mask
            amp0 = state[base]
            amp1 = state[paired]
            self._state[base] = matrix[0, 0] * amp0 + matrix[0, 1] * amp1
            self._state[paired] = matrix[1, 0] * amp0 + matrix[1, 1] * amp1

    def _apply_cnot_state(self, control: int, target: int) -> None:
        control = self._validate_qubit(control)
        target = self._validate_qubit(target)
        if control == target:
            raise ValueError("CNOT control and target must differ.")
        control_mask = 1 << control
        target_mask = 1 << target
        for index in range(self._state.size):
            if (index & control_mask) and not (index & target_mask):
                paired = index | target_mask
                self._state[index], self._state[paired] = self._state[paired], self._state[index]

    def _apply_cz_state(self, control: int, target: int) -> None:
        control = self._validate_qubit(control)
        target = self._validate_qubit(target)
        if control == target:
            raise ValueError("CZ control and target must differ.")
        mask = (1 << control) | (1 << target)
        for index in range(self._state.size):
            if (index & mask) == mask:
                self._state[index] *= -1.0

    def _apply_swap_state(self, first: int, second: int) -> None:
        first = self._validate_qubit(first)
        second = self._validate_qubit(second)
        if first == second:
            return
        first_mask = 1 << first
        second_mask = 1 << second
        for index in range(self._state.size):
            first_bit = bool(index & first_mask)
            second_bit = bool(index & second_mask)
            if first_bit or not second_bit:
                continue
            paired = (index | first_mask) & ~second_mask
            self._state[index], self._state[paired] = self._state[paired], self._state[index]

    def _apply_single_qubit_tableau(self, target: int, mapping: dict[str, tuple[complex, str]]) -> None:
        target = self._validate_qubit(target)
        transformed = []
        for phase, paulis in self._generators:
            pauli_list = list(paulis)
            factor, mapped = mapping[pauli_list[target]]
            pauli_list[target] = mapped
            transformed.append((phase * factor, tuple(pauli_list)))
        self._generators = transformed

    def _transform_cnot_factor(self, qubit: int, pauli: str, control: int, target: int) -> tuple[complex, tuple[str, ...]]:
        if pauli == "I":
            return 1.0 + 0.0j, tuple(["I"] * int(self.num_qubits))
        if qubit == control:
            if pauli == "X":
                out = list(self._single_pauli(control, "X"))
                out[target] = "X"
                return 1.0 + 0.0j, tuple(out)
            if pauli == "Y":
                out = list(self._single_pauli(control, "Y"))
                out[target] = "X"
                return 1.0 + 0.0j, tuple(out)
            return 1.0 + 0.0j, self._single_pauli(control, "Z")
        if qubit == target:
            if pauli == "X":
                return 1.0 + 0.0j, self._single_pauli(target, "X")
            if pauli == "Y":
                out = list(self._single_pauli(control, "Z"))
                out[target] = "Y"
                return 1.0 + 0.0j, tuple(out)
            out = list(self._single_pauli(control, "Z"))
            out[target] = "Z"
            return 1.0 + 0.0j, tuple(out)
        return 1.0 + 0.0j, self._single_pauli(qubit, pauli)

    def _apply_cnot_tableau(self, control: int, target: int) -> None:
        control = self._validate_qubit(control)
        target = self._validate_qubit(target)
        transformed = []
        identity = tuple(["I"] * int(self.num_qubits))
        for phase, paulis in self._generators:
            out_phase = phase
            out_paulis = identity
            for qubit, pauli in enumerate(paulis):
                factor, mapped = self._transform_cnot_factor(qubit, pauli, control, target)
                out_phase, out_paulis = _multiply_pauli_strings(out_phase, out_paulis, factor, mapped)
            transformed.append((out_phase, out_paulis))
        self._generators = transformed

    def _apply_swap_tableau(self, first: int, second: int) -> None:
        first = self._validate_qubit(first)
        second = self._validate_qubit(second)
        transformed = []
        for phase, paulis in self._generators:
            pauli_list = list(paulis)
            pauli_list[first], pauli_list[second] = pauli_list[second], pauli_list[first]
            transformed.append((phase, tuple(pauli_list)))
        self._generators = transformed

    def _apply_op(self, op) -> None:
        name = op.name.lower()
        targets = [self._validate_qubit(target) for target in op.targets]
        if name in {"h", "x", "y", "z", "s", "sdg"} and len(targets) != 1:
            raise ValueError(f"Gate '{op.name}' expects one target.")
        if name in {"cnot", "cz", "swap"} and len(targets) != 2:
            raise ValueError(f"Gate '{op.name}' expects two targets.")

        if name == "h":
            self._apply_single_qubit_state(self._H, targets[0])
            self._apply_single_qubit_tableau(targets[0], {"I": (1, "I"), "X": (1, "Z"), "Y": (-1, "Y"), "Z": (1, "X")})
            return
        if name == "x":
            self._apply_single_qubit_state(self._X, targets[0])
            self._apply_single_qubit_tableau(targets[0], {"I": (1, "I"), "X": (1, "X"), "Y": (-1, "Y"), "Z": (-1, "Z")})
            return
        if name == "y":
            self._apply_single_qubit_state(self._Y, targets[0])
            self._apply_single_qubit_tableau(targets[0], {"I": (1, "I"), "X": (-1, "X"), "Y": (1, "Y"), "Z": (-1, "Z")})
            return
        if name == "z":
            self._apply_single_qubit_state(self._Z, targets[0])
            self._apply_single_qubit_tableau(targets[0], {"I": (1, "I"), "X": (-1, "X"), "Y": (-1, "Y"), "Z": (1, "Z")})
            return
        if name == "s":
            self._apply_single_qubit_state(self._S, targets[0])
            self._apply_single_qubit_tableau(targets[0], {"I": (1, "I"), "X": (1, "Y"), "Y": (-1, "X"), "Z": (1, "Z")})
            return
        if name == "sdg":
            self._apply_single_qubit_state(self._SDG, targets[0])
            self._apply_single_qubit_tableau(targets[0], {"I": (1, "I"), "X": (-1, "Y"), "Y": (1, "X"), "Z": (1, "Z")})
            return
        if name == "cnot":
            self._apply_cnot_state(targets[0], targets[1])
            self._apply_cnot_tableau(targets[0], targets[1])
            return
        if name == "cz":
            self._apply_cz_state(targets[0], targets[1])
            self._apply_single_qubit_tableau(targets[1], {"I": (1, "I"), "X": (1, "Z"), "Y": (-1, "Y"), "Z": (1, "X")})
            self._apply_cnot_tableau(targets[0], targets[1])
            self._apply_single_qubit_tableau(targets[1], {"I": (1, "I"), "X": (1, "Z"), "Y": (-1, "Y"), "Z": (1, "X")})
            return
        if name == "swap":
            self._apply_swap_state(targets[0], targets[1])
            self._apply_swap_tableau(targets[0], targets[1])
            return

        raise NotImplementedError(
            f"Gate '{op.name}' is outside the stabilizer backend's Clifford-only subset."
        )

    def run_ops(self, ops, noise_model=None):
        if noise_model is not None:
            raise NotImplementedError("Noise models are not supported by the stabilizer backend.")
        for op in ops:
            self._apply_op(op)

    def apply_noise(self, channel: str, targets: List[int], prob: float, kraus_matrices=None):
        raise NotImplementedError("Noise models are not supported by the stabilizer backend.")

    def get_state(self):
        return self._state.copy()

    def sample(self, shots: int, qubits: Optional[Sequence[int]] = None):
        shots = _validate_positive_integer(shots, "shots")
        measured_qubits = _normalize_sample_qubits(qubits, self.num_qubits)
        probs = np.zeros(1 << len(measured_qubits), dtype=np.float64)
        for basis, amplitude in enumerate(self._state):
            outcome = 0
            for bit, qubit in enumerate(measured_qubits):
                if (basis >> int(qubit)) & 1:
                    outcome |= 1 << bit
            probs[outcome] += float(abs(amplitude) ** 2)
        total = probs.sum()
        if total <= 0.0:
            raise RuntimeError("Stabilizer state has no probability mass.")
        probs /= total
        raw_results = np.random.choice(len(probs), size=shots, p=probs).astype(np.uint64)
        return _format_sample_counts(raw_results, len(measured_qubits))

    def _stabilizer_group(self) -> dict[tuple[str, ...], complex]:
        group = {tuple(["I"] * int(self.num_qubits)): 1.0 + 0.0j}
        for generator_phase, generator in self._generators:
            additions = {}
            for paulis, phase in group.items():
                product_phase, product = _multiply_pauli_strings(phase, paulis, generator_phase, generator)
                additions[product] = product_phase
            group.update(additions)
        return group

    def _pauli_expectation(self, paulis: Sequence[tuple[str, int]]) -> complex:
        target = ["I"] * int(self.num_qubits)
        for pauli, qubit in paulis:
            if pauli not in {"X", "Y", "Z"}:
                raise NotImplementedError(f"Unsupported Pauli observable '{pauli}'.")
            target[self._validate_qubit(qubit)] = pauli
        phase = self._stabilizer_group().get(tuple(target))
        if phase is None:
            return 0.0
        if np.isclose(phase, 1.0 + 0.0j):
            return 1.0
        if np.isclose(phase, -1.0 + 0.0j):
            return -1.0
        raise RuntimeError(f"Invalid stabilizer phase for observable {target}: {phase}")

    def expectation(self, operator):
        if isinstance(operator, (HermitianOperator, SparseHamiltonianOperator)):
            raise NotImplementedError("The stabilizer backend supports Pauli observables only.")
        if isinstance(operator, SumOperator):
            terms = _combined_pauli_terms(operator, self.num_qubits)
        else:
            terms = _combined_pauli_terms(operator, self.num_qubits)

        total = 0.0 + 0.0j
        for coefficient, paulis in terms:
            if not paulis:
                total += coefficient
            else:
                total += coefficient * self._pauli_expectation(paulis)
        return _finalize_expectation(total)


class StateVectorBackend(_BaseBackend):
    """Simulates a quantum state vector by dispatching to hipStateVec."""

    def __init__(self, num_qubits: int, enable_fusion: Optional[bool] = None):
        super().__init__(num_qubits)
        enable_fusion = _validate_optional_boolean_option(enable_fusion, "enable_fusion")
        if enable_fusion is None:
            enable_fusion = os.environ.get(_DISABLE_FUSION_ENV_VAR, "").strip().lower() not in {"1", "true", "yes", "on"}

        if hip_backend is None:
            if not _mock_backends_enabled():
                raise _native_backend_error("_rocq_hip_backend", "state_vector")
            _warn_mock_backend("state_vector")
            self._state = _MockStateVectorState(self.num_qubits)
            self._uses_mock = True
        else:
            self._state = _HipStateVectorState(self.num_qubits, enable_fusion=enable_fusion)
            self._uses_mock = False

    def _apply_op(self, op):
        self._state.apply_named_gate(op.name, op.targets, op.params)

    def _is_cnot(self, op) -> bool:
        return op.name.lower() == "cnot" and len(op.targets) == 2

    def _is_fusable_neighbor(self, op, cnot_op) -> bool:
        return (
            op.name.lower() in _FUSABLE_SINGLE_QUBIT_GATES
            and len(op.targets) == 1
            and op.targets[0] in cnot_op.targets
        )

    def _single_qubit_fusion_span(self, ops, index: int):
        current = ops[index]
        if current.name.lower() not in _FUSABLE_SINGLE_QUBIT_GATES or len(current.targets) != 1:
            return None

        target = current.targets[0]
        end = index + 1
        while end < len(ops):
            candidate = ops[end]
            if (
                candidate.name.lower() not in _FUSABLE_SINGLE_QUBIT_GATES
                or len(candidate.targets) != 1
                or candidate.targets[0] != target
            ):
                break
            end += 1

        if end - index < 2:
            return None
        return ops[index:end]

    def _to_fusion_gate(self, op):
        gate = hip_backend.GateOp()
        gate.params = []
        gate.controls = []
        gate.targets = []
        lower = op.name.lower()

        if lower == "cnot":
            gate.name = "CNOT"
            gate.controls = [int(op.targets[0])]
            gate.targets = [int(op.targets[1])]
            return gate

        if len(op.targets) != 1 or lower not in _FUSABLE_SINGLE_QUBIT_GATES:
            raise ValueError(f"Operation '{op.name}' is not eligible for GateFusion.")

        gate.name = lower.upper()
        gate.targets = [int(op.targets[0])]
        if lower in {"rx", "ry"}:
            angle = op.params.get("theta", op.params.get("phi"))
            gate.params = [_validate_gate_angle(lower, angle)]
        elif lower == "rz":
            angle = op.params.get("phi", op.params.get("theta"))
            gate.params = [_validate_gate_angle(lower, angle)]
        return gate

    def _try_fuse(self, ops, index: int) -> int:
        if self._uses_mock:
            return 0

        fusion = self._state.fusion_engine()
        if fusion is None:
            return 0

        current = ops[index]
        queue = self._single_qubit_fusion_span(ops, index)

        if queue is None and self._is_cnot(current):
            if index + 1 < len(ops) and self._is_fusable_neighbor(ops[index + 1], current):
                queue = [current, ops[index + 1]]
        elif (
            queue is None
            and index + 1 < len(ops)
            and self._is_fusable_neighbor(current, ops[index + 1])
            and self._is_cnot(ops[index + 1])
        ):
            queue = [current, ops[index + 1]]
            if index + 2 < len(ops) and self._is_fusable_neighbor(ops[index + 2], ops[index + 1]):
                queue.append(ops[index + 2])

        if queue is None:
            return 0

        fusion_queue = [self._to_fusion_gate(op) for op in queue]
        status = fusion.process_queue(fusion_queue)
        if status != hip_backend.rocqStatus.SUCCESS:
            raise RuntimeError(f"GateFusion.process_queue failed: {status}")
        return len(queue)

    def run_ops(self, ops, noise_model=None):
        if noise_model is not None:
            raise NotImplementedError("Noise models are only supported by the 'density_matrix' backend.")

        index = 0
        while index < len(ops):
            span = self._try_fuse(ops, index)
            if span > 0:
                index += span
                continue
            self._apply_op(ops[index])
            index += 1

    def apply_noise(self, channel: str, targets: List[int], prob: float, kraus_matrices=None):
        raise NotImplementedError("Noise models are only supported by the 'density_matrix' backend.")

    def get_state(self):
        return self._state.get_state_vector()

    def sample(self, shots: int, qubits: Optional[Sequence[int]] = None):
        shots = _validate_positive_integer(shots, "shots")
        measured_qubits = _normalize_sample_qubits(qubits, self.num_qubits)
        raw_results = self._state.sample(measured_qubits, shots)
        return _format_sample_counts(raw_results, len(measured_qubits))

    def expectation(self, operator):
        return self._state.expectation(operator)


class DensityMatrixBackend(_BaseBackend):
    """Simulates a quantum system using a density matrix backend."""

    def __init__(self, num_qubits: int):
        super().__init__(num_qubits)
        if dm_backend is None:
            if not _mock_backends_enabled():
                raise _native_backend_error("rocq_hip", "density_matrix")
            _warn_mock_backend("density_matrix")
            self._state = _MockDensityMatrixState(self.num_qubits)
            self._uses_mock = True
        else:
            self._state = _HipDensityMatrixState(self.num_qubits)
            self._uses_mock = False

    def _gate_matrix(self, op: str, param: Optional[float] = None) -> np.ndarray:
        if op == "h":
            return (1.0 / math.sqrt(2.0)) * np.array([[1, 1], [1, -1]], dtype=np.complex64)
        if op == "x":
            return np.array([[0, 1], [1, 0]], dtype=np.complex64)
        if op == "y":
            return np.array([[0, -1j], [1j, 0]], dtype=np.complex64)
        if op == "z":
            return np.array([[1, 0], [0, -1]], dtype=np.complex64)
        if op == "s":
            return np.array([[1, 0], [0, 1j]], dtype=np.complex64)
        if op == "sdg":
            return np.array([[1, 0], [0, -1j]], dtype=np.complex64)
        if op == "t":
            phase = math.pi / 4.0
            return np.array([[1, 0], [0, math.cos(phase) + 1j * math.sin(phase)]], dtype=np.complex64)
        if op in {"tdg", "tdag"}:
            phase = -math.pi / 4.0
            return np.array([[1, 0], [0, math.cos(phase) + 1j * math.sin(phase)]], dtype=np.complex64)
        if op in {"p", "phase"} and param is not None:
            return np.array([[1, 0], [0, math.cos(param) + 1j * math.sin(param)]], dtype=np.complex64)
        if op in {"rx", "ry", "rz"} and param is not None:
            c = math.cos(param / 2.0)
            s = math.sin(param / 2.0)
            if op == "rx":
                return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex64)
            if op == "ry":
                return np.array([[c, -s], [s, c]], dtype=np.complex64)
            return np.array([[c - 1j * s, 0], [0, c + 1j * s]], dtype=np.complex64)
        raise ValueError(f"Gate '{op}' is not supported by the density matrix backend.")

    def _angle(self, op_name: str, params: Optional[Dict[str, float]]) -> Optional[float]:
        if isinstance(params, dict):
            value = params.get("theta", params.get("phi"))
        elif isinstance(params, (list, tuple)) and params:
            value = params[0]
        else:
            value = None
        return None if value is None else _validate_gate_angle(op_name, value)

    def _apply_single_qubit_gate_matrix(self, gate_name: str, target: int) -> None:
        self._state.apply_gate_matrix(self._gate_matrix(gate_name), target)

    def _apply_ccx_decomposition(self, control_a: int, control_b: int, target: int) -> None:
        self._apply_single_qubit_gate_matrix("h", target)
        self._state.apply_cnot(control_b, target)
        self._apply_single_qubit_gate_matrix("tdg", target)
        self._state.apply_cnot(control_a, target)
        self._apply_single_qubit_gate_matrix("t", target)
        self._state.apply_cnot(control_b, target)
        self._apply_single_qubit_gate_matrix("tdg", target)
        self._state.apply_cnot(control_a, target)
        self._apply_single_qubit_gate_matrix("t", control_b)
        self._apply_single_qubit_gate_matrix("t", target)
        self._apply_single_qubit_gate_matrix("h", target)
        self._state.apply_cnot(control_a, control_b)
        self._apply_single_qubit_gate_matrix("t", control_a)
        self._apply_single_qubit_gate_matrix("tdg", control_b)
        self._state.apply_cnot(control_a, control_b)

    def _apply_mcx_decomposition(self, targets: Sequence[int]) -> None:
        if len(targets) < 2:
            raise ValueError("Gate 'mcx' requires at least one control qubit and one target qubit.")
        if len(targets) == 2:
            self._state.apply_cnot(targets[0], targets[1])
            return
        if len(targets) == 3:
            self._apply_ccx_decomposition(targets[0], targets[1], targets[2])
            return
        raise NotImplementedError(
            "DensityMatrixBackend supports mcx with at most two controls; "
            "larger MCX needs an explicit ancilla/decomposition policy."
        )

    def _apply_cswap_decomposition(self, control: int, target_a: int, target_b: int) -> None:
        self._apply_ccx_decomposition(control, target_b, target_a)
        self._apply_ccx_decomposition(control, target_a, target_b)
        self._apply_ccx_decomposition(control, target_b, target_a)

    def _apply_op(self, op):
        params = op.params or {}
        name = op.name.lower()

        if name == "cnot":
            targets = _normalize_gate_targets(op.name, op.targets, self.num_qubits, exact_count=2)
            self._state.apply_cnot(targets[0], targets[1])
            return
        if name in {"ccx", "toffoli"}:
            targets = _normalize_gate_targets(op.name, op.targets, self.num_qubits, exact_count=3)
            self._apply_ccx_decomposition(targets[0], targets[1], targets[2])
            return
        if name == "mcx":
            targets = _normalize_gate_targets(op.name, op.targets, self.num_qubits, min_count=2)
            self._apply_mcx_decomposition(targets)
            return
        if name in {"cswap", "fredkin"}:
            targets = _normalize_gate_targets(op.name, op.targets, self.num_qubits, exact_count=3)
            self._apply_cswap_decomposition(targets[0], targets[1], targets[2])
            return
        if name == "swap":
            control, target = _normalize_gate_targets(op.name, op.targets, self.num_qubits, exact_count=2)
            self._state.apply_cnot(control, target)
            self._state.apply_cnot(target, control)
            self._state.apply_cnot(control, target)
            return
        if name in {"cz", "crx", "cry", "crz", "cp"}:
            targets = _normalize_gate_targets(op.name, op.targets, self.num_qubits, exact_count=2)
            controlled_name = "z" if name == "cz" else name[1:]
            matrix = self._gate_matrix(controlled_name, self._angle(name, params))
            self._state.apply_controlled_gate(matrix, targets[0], targets[1])
            return

        targets = _normalize_gate_targets(op.name, op.targets, self.num_qubits, exact_count=1)
        matrix = self._gate_matrix(name, self._angle(name, params))
        self._state.apply_gate_matrix(matrix, targets[0])

    def _iter_noise_channels(self, noise_model, op):
        if noise_model is None:
            return
        for channel in noise_model.get_channels():
            if channel["op"] and channel["op"] != op.name.lower():
                continue
            yield channel

    def run_ops(self, ops, noise_model=None):
        for op in ops:
            self._apply_op(op)
            for channel in self._iter_noise_channels(noise_model, op):
                targets = channel["qubits"] if channel["qubits"] else op.targets
                self.apply_noise(
                    channel["type"],
                    list(targets),
                    float(channel["prob"]),
                    channel.get("kraus_matrices"),
                )

    def apply_noise(self, channel_type: str, targets: List[int], prob: float, kraus_matrices=None):
        if not isinstance(channel_type, str) or not channel_type.strip():
            raise ValueError("Noise channel type must be a non-empty string.")
        channel_lower = channel_type.strip().lower()
        normalized_targets = _normalize_channel_targets(targets, self.num_qubits)
        prob = _validate_probability(prob, "Noise channel probability")
        if channel_lower != "kraus" and kraus_matrices is not None:
            raise ValueError("kraus_matrices may only be supplied for 'kraus' noise channels.")

        if channel_lower == "kraus":
            matrices = _probability_mixed_kraus_matrices(kraus_matrices, normalized_targets, prob)
            self._state.apply_channel(normalized_targets, matrices)
            return

        for target in normalized_targets:
            if channel_lower == "depolarizing":
                self._state.apply_depolarizing_channel(target, prob)
            elif channel_lower == "bit_flip":
                self._state.apply_bit_flip_channel(target, prob)
            elif channel_lower == "phase_flip":
                self._state.apply_phase_flip_channel(target, prob)
            elif channel_lower == "amplitude_damping":
                self._state.apply_amplitude_damping_channel(target, prob)
            else:
                raise ValueError(f"Noise channel '{channel_type}' is not supported by the DensityMatrixBackend.")

    def get_state(self):
        return self._state.get_density_matrix()

    def sample(self, shots: int, qubits: Optional[Sequence[int]] = None):
        shots = _validate_positive_integer(shots, "shots")
        measured_qubits = _normalize_sample_qubits(qubits, self.num_qubits)
        raw_results = self._state.sample(measured_qubits, shots)
        return _format_sample_counts(raw_results, len(measured_qubits))

    def expectation(self, operator):
        if isinstance(operator, HermitianOperator):
            matrix, targets = _normalize_matrix_targets(operator.matrix, operator.targets, self.num_qubits)
            value = None
            if not self._uses_mock:
                native = getattr(self._state, "compute_expectation_matrix", None)
                if callable(native) and len(targets) <= 4:
                    try:
                        value = native(matrix, targets)
                    except NotImplementedError:
                        value = None
            if value is None:
                value = _density_matrix_expectation_matrix(
                    self.get_state(),
                    matrix,
                    targets,
                    self.num_qubits,
                )
            return _finalize_expectation(operator.coefficient * value)

        if isinstance(operator, SparseHamiltonianOperator):
            data, indices, indptr, _ = _normalize_sparse_hamiltonian(operator, self.num_qubits)
            value = _density_matrix_sparse_hamiltonian_expectation(
                self.get_state(),
                data,
                indices,
                indptr,
            )
            return _finalize_expectation(operator.coefficient * value)

        if isinstance(operator, SumOperator):
            try:
                terms = _combined_pauli_terms(operator, self.num_qubits)
            except NotImplementedError:
                return _expect_mixed_sum_operator(operator, self.num_qubits, self.expectation)
        else:
            terms = _combined_pauli_terms(operator, self.num_qubits)

        total = 0.0 + 0.0j
        for coefficient, paulis in terms:
            if not paulis:
                total += coefficient
                continue

            if self._uses_mock:
                if len(paulis) == 1:
                    _, qubit = paulis[0]
                    total += coefficient * self._state.compute_expectation("mock", qubit)
                    continue
                if all(pauli == "Z" for pauli, _ in paulis):
                    total += coefficient * self._state.compute_z_product_expectation([qubit for _, qubit in paulis])
                    continue
                raise NotImplementedError(
                    "Mock density-matrix backends support only identity and Z-product expectation contracts."
                )

            if len(paulis) == 1:
                pauli, qubit = paulis[0]
                if pauli == "X":
                    value = self._state.compute_expectation(dm_backend.Pauli.X, qubit)
                elif pauli == "Y":
                    value = self._state.compute_expectation(dm_backend.Pauli.Y, qubit)
                elif pauli == "Z":
                    value = self._state.compute_expectation(dm_backend.Pauli.Z, qubit)
                else:
                    raise NotImplementedError(f"Unsupported Pauli observable '{pauli}'.")
                total += coefficient * value
                continue

            if all(pauli == "Z" for pauli, _ in paulis):
                value = self._state.compute_z_product_expectation([qubit for _, qubit in paulis])
                total += coefficient * value
                continue

            raise NotImplementedError(
                "The density-matrix backend currently supports only single-qubit X/Y/Z "
                "and Z-product expectation values."
            )

        return _finalize_expectation(total)


def get_backend(backend_name: str, num_qubits: int, *, enable_fusion: Optional[bool] = None) -> _BaseBackend:
    """Factory function to instantiate a simulation backend."""

    supported = {
        "state_vector": StateVectorBackend,
        "density_matrix": DensityMatrixBackend,
        "stabilizer": StabilizerBackend,
        "tableau": StabilizerBackend,
        "clifford": StabilizerBackend,
    }
    backend_name = _validate_backend_name(backend_name, supported)
    enable_fusion = _validate_optional_boolean_option(enable_fusion, "enable_fusion")
    if backend_name == "state_vector":
        return StateVectorBackend(num_qubits, enable_fusion=enable_fusion)
    if enable_fusion is not None:
        raise ValueError("enable_fusion only applies to the 'state_vector' backend.")
    return supported[backend_name](num_qubits)
