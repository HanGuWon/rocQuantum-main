from __future__ import annotations

import math
import os
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


def _mock_backends_enabled() -> bool:
    return os.environ.get(_MOCK_ENV_VAR, "").strip().lower() in {"1", "true", "yes", "on"}


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


def _finalize_expectation(value: complex):
    return value.real if abs(value.imag) < 1e-12 else value


def _combined_pauli_terms(operator):
    combined = {}
    order = []
    for coefficient, paulis in iter_pauli_terms(operator):
        key = tuple((pauli, int(qubit)) for pauli, qubit in paulis)
        if key not in combined:
            combined[key] = 0.0 + 0.0j
            order.append(key)
        combined[key] += coefficient
    return [
        (combined[key], list(key))
        for key in order
        if combined[key] != 0
    ]


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
    if isinstance(targets, (int, np.integer)):
        normalized = [int(targets)]
    else:
        normalized = [int(target) for target in targets]

    if not normalized:
        raise ValueError("Kraus channel must target at least one qubit.")
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
    prob = float(probability)
    if prob < 0.0 or prob > 1.0:
        raise ValueError("Kraus channel probability must be between 0 and 1.")
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

    def apply_named_gate(self, op_name: str, targets: List[int], params: Optional[Dict[str, float]] = None):
        return None

    def apply_matrix(self, targets: Sequence[int], matrix: np.ndarray):
        return None

    def apply_controlled_matrix(self, controls: Sequence[int], targets: Sequence[int], matrix: np.ndarray):
        return None

    def get_state_vector(self):
        return self._state.copy()

    def sample(self, measured_qubits: Sequence[int], num_shots: int):
        return np.zeros(num_shots, dtype=np.uint64)

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
                terms = _combined_pauli_terms(operator)
            except NotImplementedError:
                return _expect_mixed_sum_operator(operator, self._num_qubits, self.expectation)
        else:
            terms = _combined_pauli_terms(operator)

        total = 0.0 + 0.0j
        for coefficient, paulis in terms:
            if not paulis:
                total += coefficient
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
        return float(angle)

    def apply_named_gate(self, op_name: str, targets: List[int], params: Optional[Dict[str, float]] = None):
        params = params or {}
        op = op_name.lower()

        if op == "h":
            return self._call(op_name, hip_backend.apply_h, self._handle, self._d_state, self._num_qubits, targets[0])
        if op == "x":
            return self._call(op_name, hip_backend.apply_x, self._handle, self._d_state, self._num_qubits, targets[0])
        if op == "y":
            return self._call(op_name, hip_backend.apply_y, self._handle, self._d_state, self._num_qubits, targets[0])
        if op == "z":
            return self._call(op_name, hip_backend.apply_z, self._handle, self._d_state, self._num_qubits, targets[0])
        if op == "s":
            return self._call(op_name, hip_backend.apply_s, self._handle, self._d_state, self._num_qubits, targets[0])
        if op == "sdg":
            return self._call(op_name, hip_backend.apply_sdg, self._handle, self._d_state, self._num_qubits, targets[0])
        if op == "t":
            return self._call(op_name, hip_backend.apply_t, self._handle, self._d_state, self._num_qubits, targets[0])
        if op in {"tdg", "tdag"}:
            return self._call(op_name, hip_backend.apply_tdg, self._handle, self._d_state, self._num_qubits, targets[0])
        if op == "rx":
            angle = self._angle(op_name, params, "theta", "phi")
            return self._call(op_name, hip_backend.apply_rx, self._handle, self._d_state, self._num_qubits, targets[0], angle)
        if op == "ry":
            angle = self._angle(op_name, params, "theta", "phi")
            return self._call(op_name, hip_backend.apply_ry, self._handle, self._d_state, self._num_qubits, targets[0], angle)
        if op == "rz":
            angle = self._angle(op_name, params, "phi", "theta")
            return self._call(op_name, hip_backend.apply_rz, self._handle, self._d_state, self._num_qubits, targets[0], angle)
        if op in {"p", "phase"}:
            angle = self._angle(op_name, params, "phi", "theta")
            return self._call(op_name, hip_backend.apply_p, self._handle, self._d_state, self._num_qubits, targets[0], angle)
        if op == "cnot":
            return self._call(op_name, hip_backend.apply_cnot, self._handle, self._d_state, self._num_qubits, targets[0], targets[1])
        if op == "cz":
            return self._call(op_name, hip_backend.apply_cz, self._handle, self._d_state, self._num_qubits, targets[0], targets[1])
        if op == "swap":
            return self._call(op_name, hip_backend.apply_swap, self._handle, self._d_state, self._num_qubits, targets[0], targets[1])
        if op == "crx":
            angle = self._angle(op_name, params, "theta", "phi")
            return self._call(op_name, hip_backend.apply_crx, self._handle, self._d_state, self._num_qubits, targets[0], targets[1], angle)
        if op == "cry":
            angle = self._angle(op_name, params, "theta", "phi")
            return self._call(op_name, hip_backend.apply_cry, self._handle, self._d_state, self._num_qubits, targets[0], targets[1], angle)
        if op == "crz":
            angle = self._angle(op_name, params, "phi", "theta")
            return self._call(op_name, hip_backend.apply_crz, self._handle, self._d_state, self._num_qubits, targets[0], targets[1], angle)
        if op in {"cp", "cphase"}:
            angle = self._angle(op_name, params, "phi", "theta")
            return self._call(op_name, hip_backend.apply_cp, self._handle, self._d_state, self._num_qubits, targets[0], targets[1], angle)
        if op in {"mcx", "ccx", "toffoli"}:
            controls = targets[:-1]
            if not controls:
                raise ValueError(f"Gate '{op_name}' requires at least one control qubit.")
            return self._call(op_name, hip_backend.apply_mcx, self._handle, self._d_state, self._num_qubits, controls, targets[-1])
        if op == "cswap":
            if len(targets) != 3:
                raise ValueError("Gate 'cswap' requires [control, target_a, target_b].")
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
                terms = _combined_pauli_terms(operator)
            except NotImplementedError:
                return _expect_mixed_sum_operator(operator, self._num_qubits, self.expectation)
        else:
            terms = _combined_pauli_terms(operator)

        total = 0.0 + 0.0j
        for coefficient, paulis in terms:
            if not paulis:
                total += coefficient
                continue

            if len(paulis) == 1:
                pauli, qubit = paulis[0]
                if pauli == "X":
                    value = hip_backend.get_expectation_value_x(self._handle, self._d_state, self._num_qubits, qubit)
                elif pauli == "Y":
                    value = hip_backend.get_expectation_value_y(self._handle, self._d_state, self._num_qubits, qubit)
                elif pauli == "Z":
                    value = hip_backend.get_expectation_value_z(self._handle, self._d_state, self._num_qubits, qubit)
                else:
                    raise NotImplementedError(f"Unsupported Pauli observable '{pauli}'.")
                total += coefficient * value
                continue

            if all(pauli == "Z" for pauli, _ in paulis):
                value = hip_backend.get_expectation_value_pauli_product_z(
                    self._handle,
                    self._d_state,
                    self._num_qubits,
                    [qubit for _, qubit in paulis],
                )
                total += coefficient * value
                continue

            pauli_string = "".join(pauli for pauli, _ in paulis)
            qubits = [qubit for _, qubit in paulis]
            value = hip_backend.get_expectation_pauli_string(
                self._handle,
                self._d_state,
                self._num_qubits,
                pauli_string,
                qubits,
            )
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

    def sample(self, measured_qubits: Sequence[int], num_shots: int):
        return self._state.sample(list(measured_qubits), int(num_shots))

    def get_density_matrix(self):
        return self._state.get_density_matrix()


class _BaseBackend:
    """Abstract base class for a quantum simulation backend."""

    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits

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


class StateVectorBackend(_BaseBackend):
    """Simulates a quantum state vector by dispatching to hipStateVec."""

    def __init__(self, num_qubits: int, enable_fusion: Optional[bool] = None):
        super().__init__(num_qubits)
        if enable_fusion is None:
            enable_fusion = os.environ.get(_DISABLE_FUSION_ENV_VAR, "").strip().lower() not in {"1", "true", "yes", "on"}

        if hip_backend is None:
            if not _mock_backends_enabled():
                raise _native_backend_error("_rocq_hip_backend", "state_vector")
            self._state = _MockStateVectorState(num_qubits)
            self._uses_mock = True
        else:
            self._state = _HipStateVectorState(num_qubits, enable_fusion=enable_fusion)
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
            gate.params = [float(op.params.get("theta", op.params.get("phi")))]
        elif lower == "rz":
            gate.params = [float(op.params.get("phi", op.params.get("theta")))]
        return gate

    def _try_fuse(self, ops, index: int) -> int:
        if self._uses_mock:
            return 0

        fusion = self._state.fusion_engine()
        if fusion is None:
            return 0

        current = ops[index]
        queue = None

        if self._is_cnot(current):
            if index + 1 < len(ops) and self._is_fusable_neighbor(ops[index + 1], current):
                queue = [current, ops[index + 1]]
        elif index + 1 < len(ops) and self._is_fusable_neighbor(current, ops[index + 1]) and self._is_cnot(ops[index + 1]):
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
        if shots <= 0:
            raise ValueError("shots must be positive.")
        measured_qubits = list(range(self.num_qubits)) if qubits is None else [int(q) for q in qubits]
        raw_results = self._state.sample(measured_qubits, int(shots))
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
            self._state = _MockDensityMatrixState(num_qubits)
            self._uses_mock = True
        else:
            self._state = _HipDensityMatrixState(num_qubits)
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

    def _angle(self, params: Optional[Dict[str, float]]) -> Optional[float]:
        if isinstance(params, dict):
            value = params.get("theta", params.get("phi"))
        elif isinstance(params, (list, tuple)) and params:
            value = params[0]
        else:
            value = None
        return None if value is None else float(value)

    def _apply_op(self, op):
        params = op.params or {}
        name = op.name.lower()

        if name == "cnot":
            self._state.apply_cnot(op.targets[0], op.targets[1])
            return
        if name == "swap":
            control, target = op.targets
            self._state.apply_cnot(control, target)
            self._state.apply_cnot(target, control)
            self._state.apply_cnot(control, target)
            return
        if name in {"cz", "crx", "cry", "crz", "cp"}:
            controlled_name = "z" if name == "cz" else name[1:]
            matrix = self._gate_matrix(controlled_name, self._angle(params))
            self._state.apply_controlled_gate(matrix, op.targets[0], op.targets[1])
            return

        matrix = self._gate_matrix(name, self._angle(params))
        self._state.apply_gate_matrix(matrix, op.targets[0])

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
        channel_lower = channel_type.lower()
        if channel_lower == "kraus":
            normalized_targets = _normalize_channel_targets(targets, self.num_qubits)
            matrices = _probability_mixed_kraus_matrices(kraus_matrices, normalized_targets, prob)
            self._state.apply_channel(normalized_targets, matrices)
            return

        for target in targets:
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
        if shots <= 0:
            raise ValueError("shots must be positive.")
        measured_qubits = list(range(self.num_qubits)) if qubits is None else [int(q) for q in qubits]
        raw_results = self._state.sample(measured_qubits, int(shots))
        return _format_sample_counts(raw_results, len(measured_qubits))

    def expectation(self, operator):
        if isinstance(operator, HermitianOperator):
            matrix, targets = _normalize_matrix_targets(operator.matrix, operator.targets, self.num_qubits)
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
                terms = _combined_pauli_terms(operator)
            except NotImplementedError:
                return _expect_mixed_sum_operator(operator, self.num_qubits, self.expectation)
        else:
            terms = _combined_pauli_terms(operator)

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


def get_backend(backend_name: str, num_qubits: int) -> _BaseBackend:
    """Factory function to instantiate a simulation backend."""

    supported = {"state_vector": StateVectorBackend, "density_matrix": DensityMatrixBackend}
    if backend_name not in supported:
        raise ValueError(f"Unsupported backend '{backend_name}'. Supported backends are: {list(supported.keys())}")
    return supported[backend_name](num_qubits)
