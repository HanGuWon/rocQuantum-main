from __future__ import annotations

from collections import Counter
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
    return [int(target) for target in targets]


def normalize_params(params: Iterable[object] | None) -> list[float]:
    if params is None:
        return []
    normalized = []
    for param in params:
        if hasattr(param, "bind"):
            raise TypeError(f"Unbound symbolic parameter {param!r} cannot be executed.")
        normalized.append(float(param))
    return normalized


def as_complex_matrix(matrix: object) -> np.ndarray:
    out = np.asarray(matrix, dtype=np.complex128)
    if out.ndim != 2 or out.shape[0] != out.shape[1]:
        raise ValueError("Operation matrix must be square.")
    return np.ascontiguousarray(out)


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
    normalized = np.asarray(statevector, dtype=np.complex128).reshape(-1)
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
    state = np.asarray(statevector, dtype=np.complex128)
    probabilities = np.abs(state) ** 2
    total = float(np.sum(probabilities))
    if total <= 0.0:
        raise ValueError("Statevector probabilities sum to zero.")
    probabilities = probabilities / total

    if rng is None:
        rng = np.random
    counts = rng.multinomial(int(shots), probabilities)
    sample_indices = np.repeat(np.arange(len(probabilities)), counts)
    num_wires = int(np.log2(len(probabilities))) if len(probabilities) else 0
    return samples_to_binary_rows(sample_indices, num_wires)


def sample_indices_from_probabilities(probabilities: Sequence[float], shots: int, rng=None) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    if probabilities.size == 0:
        raise ValueError("Sampler probabilities cannot be empty.")
    if np.any(probabilities < -1.0e-12):
        raise ValueError("Sampler probabilities must be non-negative.")
    probabilities = np.clip(probabilities, 0.0, None)
    total = float(np.sum(probabilities))
    if total <= 0.0:
        raise ValueError("Sampler probabilities sum to zero.")
    probabilities = probabilities / total

    if rng is None:
        rng = np.random
    counts = rng.multinomial(int(shots), probabilities)
    return np.repeat(np.arange(probabilities.size, dtype=np.int64), counts)


def sample_indices_batch_from_probabilities(probabilities: Sequence[Sequence[float]], shots: int, rng=None) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2:
        raise ValueError("Batched sampler probabilities must be a two-dimensional array.")
    shots = int(shots)
    if probabilities.shape[0] == 0:
        return np.empty((0, shots), dtype=np.int64)
    return np.vstack(
        [sample_indices_from_probabilities(row, shots, rng=rng) for row in probabilities]
    ).astype(np.int64, copy=False)


def probabilities_from_statevector(
    statevector: Sequence[complex],
    qubits: Iterable[int] | None = None,
) -> np.ndarray:
    state = np.asarray(statevector, dtype=np.complex128)
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
    state = np.asarray(statevector, dtype=np.complex128)
    if len(pauli_string) != len(targets):
        raise ValueError("Pauli string length must match target qubit count.")
    if not targets:
        return 1.0

    num_qubits = int(np.log2(state.size)) if state.size else 0
    if state.size != (1 << num_qubits):
        raise ValueError("Statevector length must be a power of two.")

    normalized_paulis = [pauli.upper() for pauli in pauli_string]
    normalized_targets = normalize_targets(targets)
    if len(set(normalized_targets)) != len(normalized_targets):
        raise ValueError("Pauli expectation targets must be unique.")
    for target in normalized_targets:
        if target < 0 or target >= num_qubits:
            raise ValueError("Pauli expectation target is outside the statevector qubit range.")
    for pauli in normalized_paulis:
        if pauli not in {"I", "X", "Y", "Z"}:
            raise ValueError("Pauli string may only contain I, X, Y, or Z.")

    result = 0.0 + 0.0j
    for basis_index, amplitude in enumerate(state):
        partner_index = basis_index
        phase = 1.0 + 0.0j
        for pauli, target in zip(normalized_paulis, normalized_targets):
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
    state = np.asarray(statevector, dtype=np.complex128).reshape(-1)
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
    state = np.asarray(statevector, dtype=np.complex128).reshape(-1)
    normalized_data = np.asarray(data, dtype=np.complex128).reshape(-1)
    normalized_indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    normalized_indptr = np.asarray(indptr, dtype=np.int64).reshape(-1)
    rows, cols = (int(dim) for dim in shape)
    if rows != state.size or cols != state.size:
        raise ValueError("Sparse Hamiltonian shape must match the statevector length.")
    if normalized_indptr.size != rows + 1:
        raise ValueError("Sparse Hamiltonian CSR indptr length must equal rows + 1.")
    if normalized_data.size != normalized_indices.size:
        raise ValueError("Sparse Hamiltonian CSR data and indices lengths must match.")
    if normalized_indptr[0] != 0 or normalized_indptr[-1] != normalized_data.size:
        raise ValueError("Sparse Hamiltonian CSR indptr must start at 0 and end at nnz.")

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


def apply_sparse_matrix_to_statevector(
    statevector: Sequence[complex],
    data: object,
    indices: object,
    indptr: object,
    shape: Sequence[int],
    targets: Iterable[int],
    num_qubits: int,
) -> np.ndarray:
    state = np.asarray(statevector, dtype=np.complex128).reshape(-1)
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
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if batch_size == 1:
            return cls(simulator_cls(int(num_qubits)))
        return cls(simulator_cls(int(num_qubits), batch_size))

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
        normalized_targets = normalize_targets(targets)
        normalized_matrix = as_complex_matrix(matrix)
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
        normalized_controls = normalize_targets(controls)
        normalized_targets = normalize_targets(targets)
        normalized_matrix = as_complex_matrix(matrix)

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
        normalized_targets = normalize_targets(targets)
        normalized_data = np.ascontiguousarray(np.asarray(data, dtype=np.complex128).reshape(-1))
        normalized_indices = np.ascontiguousarray(np.asarray(indices, dtype=np.int64).reshape(-1))
        normalized_indptr = np.ascontiguousarray(np.asarray(indptr, dtype=np.int64).reshape(-1))
        normalized_shape = tuple(int(dim) for dim in shape)

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

    def set_statevector(self, statevector: object) -> None:
        normalized_state = np.ascontiguousarray(np.asarray(statevector, dtype=np.complex128).reshape(-1))
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
        normalized_states = np.ascontiguousarray(np.asarray(statevectors, dtype=np.complex128).reshape(-1))
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
                return np.asarray(getter(batch_index), dtype=np.complex128)
            except TypeError:
                return np.asarray(getter(), dtype=np.complex128)
        return np.asarray(getter(batch_index), dtype=np.complex128)

    def statevectors(self) -> np.ndarray:
        getter = getattr(self.simulator, "get_statevectors", None)
        if not callable(getter):
            getter = getattr(self.simulator, "GetStateVectors", None)
        if callable(getter):
            states = np.asarray(getter(), dtype=np.complex128)
            return states.reshape(self.batch_size(), 1 << self.num_qubits())
        if self.batch_size() == 1:
            return self.statevector().reshape(1, -1)
        raise NotImplementedError("The active rocQuantum binding does not expose batched state readback.")

    def measure(self, qubits: Iterable[int], shots: int) -> list[int]:
        measure = getattr(self.simulator, "measure", None)
        if not callable(measure):
            raise NotImplementedError("The active rocQuantum binding does not expose native sampling.")
        return [int(sample) for sample in measure(normalize_targets(qubits), int(shots))]

    def measure_batch(self, qubits: Iterable[int], shots: int) -> np.ndarray:
        normalized_qubits = normalize_targets(qubits)
        shots = int(shots)

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
                return np.asarray(native(native_qubits), dtype=float)
            except Exception as exc:
                if not _native_probabilities_unavailable(exc):
                    raise

        legacy = getattr(self.simulator, "Probabilities", None)
        if callable(legacy):
            try:
                return np.asarray(legacy(native_qubits), dtype=float)
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
                probabilities = np.asarray(native(native_qubits), dtype=float)
                return probabilities.reshape(self.batch_size(), -1)
            except Exception as exc:
                if not _native_probabilities_unavailable(exc):
                    raise

        legacy = getattr(self.simulator, "ProbabilitiesBatch", None)
        if callable(legacy):
            try:
                probabilities = np.asarray(legacy(native_qubits), dtype=float)
                return probabilities.reshape(self.batch_size(), -1)
            except Exception as exc:
                if not _native_probabilities_unavailable(exc):
                    raise

        if self.batch_size() == 1:
            return self.probabilities(normalized_qubits).reshape(1, -1)

        return np.vstack(
            [probabilities_from_statevector(state, normalized_qubits) for state in self.statevectors()]
        )

    def expectation_value(self, pauli: str, target: int) -> float:
        native = getattr(self.simulator, "expectation_value", None)
        if callable(native):
            return float(native(str(pauli), int(target)))

        legacy = getattr(self.simulator, "GetExpectationValue", None)
        if callable(legacy):
            return float(legacy(str(pauli), int(target)))

        return self.expectation_pauli_string(str(pauli), [int(target)])

    def expectation_pauli_string(self, pauli_string: str, targets: Iterable[int]) -> float:
        normalized_targets = normalize_targets(targets)

        native = getattr(self.simulator, "expectation_pauli_string", None)
        if callable(native):
            return float(native(str(pauli_string), normalized_targets))

        legacy = getattr(self.simulator, "GetExpectationPauliString", None)
        if callable(legacy):
            return float(legacy(str(pauli_string), normalized_targets))

        return expectation_from_statevector(self.statevector(), str(pauli_string), normalized_targets)

    def expectation_pauli_string_batch(self, pauli_string: str, targets: Iterable[int]) -> np.ndarray:
        normalized_targets = normalize_targets(targets)

        def _native_expectation_unavailable(exc: Exception) -> bool:
            message = str(exc)
            return isinstance(exc, NotImplementedError) or "status 5" in message

        native = getattr(self.simulator, "expectation_pauli_string_batch", None)
        if callable(native):
            try:
                return np.asarray(native(str(pauli_string), normalized_targets), dtype=float).reshape(self.batch_size())
            except Exception as exc:
                if not _native_expectation_unavailable(exc):
                    raise

        legacy = getattr(self.simulator, "GetExpectationPauliStringBatch", None)
        if callable(legacy):
            try:
                return np.asarray(legacy(str(pauli_string), normalized_targets), dtype=float).reshape(self.batch_size())
            except Exception as exc:
                if not _native_expectation_unavailable(exc):
                    raise

        if self.batch_size() == 1:
            return np.asarray([self.expectation_pauli_string(pauli_string, normalized_targets)], dtype=float)

        return np.asarray(
            [
                expectation_from_statevector(state, str(pauli_string), normalized_targets)
                for state in self.statevectors()
            ],
            dtype=float,
        )

    def expectation_matrix(self, matrix: object, targets: Iterable[int]) -> complex:
        normalized_targets = normalize_targets(targets)
        normalized_matrix = np.ascontiguousarray(np.asarray(matrix, dtype=np.complex128))

        native = getattr(self.simulator, "expectation_matrix", None)
        if callable(native):
            return complex(native(normalized_matrix, normalized_targets))

        legacy = getattr(self.simulator, "ExpectationMatrix", None)
        if callable(legacy):
            return complex(legacy(normalized_matrix, normalized_targets))

        raise NotImplementedError("The active rocQuantum binding does not expose dense matrix expectations.")

    def expectation_matrix_batch(self, matrix: object, targets: Iterable[int]) -> np.ndarray:
        normalized_targets = normalize_targets(targets)
        normalized_matrix = np.ascontiguousarray(np.asarray(matrix, dtype=np.complex128))

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

        if self.batch_size() == 1:
            return np.asarray([self.expectation_matrix(normalized_matrix, normalized_targets)], dtype=np.complex128)

        return np.asarray(
            [
                expectation_matrix_from_statevector(state, normalized_matrix, normalized_targets)
                for state in self.statevectors()
            ],
            dtype=np.complex128,
        )

    def sparse_hamiltonian_moments(
        self,
        data: object,
        indices: object,
        indptr: object,
        shape: Iterable[int],
    ) -> tuple[complex, complex]:
        normalized_data = np.ascontiguousarray(np.asarray(data, dtype=np.complex128).reshape(-1))
        normalized_indices = np.ascontiguousarray(np.asarray(indices, dtype=np.int64).reshape(-1))
        normalized_indptr = np.ascontiguousarray(np.asarray(indptr, dtype=np.int64).reshape(-1))
        normalized_shape = tuple(int(dim) for dim in shape)

        native = getattr(self.simulator, "sparse_hamiltonian_moments", None)
        if callable(native):
            mean, second_moment = native(
                normalized_data,
                normalized_indices,
                normalized_indptr,
                normalized_shape,
            )
            return complex(mean), complex(second_moment)

        legacy = getattr(self.simulator, "SparseHamiltonianMoments", None)
        if callable(legacy):
            mean, second_moment = legacy(
                normalized_data,
                normalized_indices,
                normalized_indptr,
                normalized_shape,
            )
            return complex(mean), complex(second_moment)

        raise NotImplementedError("The active rocQuantum binding does not expose sparse Hamiltonian moments.")

    def sparse_hamiltonian_moments_batch(
        self,
        data: object,
        indices: object,
        indptr: object,
        shape: Iterable[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        normalized_data = np.ascontiguousarray(np.asarray(data, dtype=np.complex128).reshape(-1))
        normalized_indices = np.ascontiguousarray(np.asarray(indices, dtype=np.int64).reshape(-1))
        normalized_indptr = np.ascontiguousarray(np.asarray(indptr, dtype=np.int64).reshape(-1))
        normalized_shape = tuple(int(dim) for dim in shape)

        native = getattr(self.simulator, "sparse_hamiltonian_moments_batch", None)
        if callable(native):
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

        legacy = getattr(self.simulator, "SparseHamiltonianMomentsBatch", None)
        if callable(legacy):
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
