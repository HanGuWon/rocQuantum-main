from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

import numpy as np


GATE_ALIASES = {
    "CNOT": "CNOT",
    "CX": "CNOT",
    "CZ": "CZ",
    "CRX": "CRX",
    "CRY": "CRY",
    "CRZ": "CRZ",
    "H": "H",
    "HADAMARD": "H",
    "I": "I",
    "ID": "I",
    "IDENTITY": "I",
    "PAULIX": "X",
    "PAULIY": "Y",
    "PAULIZ": "Z",
    "RX": "RX",
    "RY": "RY",
    "RZ": "RZ",
    "S": "S",
    "SDG": "SDG",
    "SWAP": "SWAP",
    "T": "T",
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


def qiskit_memory_from_samples(
    raw_samples: Sequence[int],
    measured_items: Sequence[tuple[int, int]],
    memory_width: int,
) -> list[str]:
    if memory_width <= 0:
        memory_width = len(measured_items)

    memory = []
    for sample in raw_samples:
        bits = ["0"] * memory_width
        for packed_bit, (classical_bit, _) in enumerate(measured_items):
            output_index = memory_width - 1 - int(classical_bit)
            if 0 <= output_index < memory_width:
                bits[output_index] = "1" if ((int(sample) >> packed_bit) & 1) else "0"
        memory.append("".join(bits))
    return memory


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


class RocQuantumRuntime:
    """Small adapter around the public rocquantum_bind simulator surface."""

    def __init__(self, simulator):
        self.simulator = simulator

    @classmethod
    def from_bindings(cls, num_qubits: int, binding_module=None):
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
        return cls(simulator_cls(int(num_qubits)))

    def reset(self) -> None:
        reset = getattr(self.simulator, "reset", None)
        if callable(reset):
            reset()
            return
        self.simulator = type(self.simulator)(int(self.num_qubits()))

    def num_qubits(self) -> int:
        getter = getattr(self.simulator, "num_qubits", None)
        if callable(getter):
            return int(getter())
        return int(getattr(self.simulator, "num_qubits", 0))

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

        if matrix is not None and normalized_name in {
            "RX", "RY", "RZ",
            "CRX", "CRY", "CRZ",
            "UNITARY", "QUBITUNITARY",
        }:
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

    def statevector(self) -> np.ndarray:
        getter = getattr(self.simulator, "get_statevector", None)
        if not callable(getter):
            getter = getattr(self.simulator, "GetStateVector", None)
        if not callable(getter):
            raise NotImplementedError("The active rocQuantum binding does not expose state readback.")
        return np.asarray(getter(), dtype=np.complex128)

    def measure(self, qubits: Iterable[int], shots: int) -> list[int]:
        measure = getattr(self.simulator, "measure", None)
        if not callable(measure):
            raise NotImplementedError("The active rocQuantum binding does not expose native sampling.")
        return [int(sample) for sample in measure(normalize_targets(qubits), int(shots))]

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
