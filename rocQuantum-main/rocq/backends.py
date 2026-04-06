from __future__ import annotations

import math
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .operator import iter_pauli_terms

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
        total = 0.0 + 0.0j
        for coefficient, paulis in iter_pauli_terms(operator):
            if not paulis:
                total += coefficient
        return _finalize_expectation(total)


class _MockDensityMatrixState:
    def __init__(self, n_qubits: int):
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

    def compute_expectation(self, pauli, target: int):
        return 0.0

    def compute_z_product_expectation(self, targets: Sequence[int]):
        return 0.0

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
        if op == "rx":
            angle = self._angle(op_name, params, "theta", "phi")
            return self._call(op_name, hip_backend.apply_rx, self._handle, self._d_state, self._num_qubits, targets[0], angle)
        if op == "ry":
            angle = self._angle(op_name, params, "theta", "phi")
            return self._call(op_name, hip_backend.apply_ry, self._handle, self._d_state, self._num_qubits, targets[0], angle)
        if op == "rz":
            angle = self._angle(op_name, params, "phi", "theta")
            return self._call(op_name, hip_backend.apply_rz, self._handle, self._d_state, self._num_qubits, targets[0], angle)
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

    def expectation(self, operator):
        total = 0.0 + 0.0j
        for coefficient, paulis in iter_pauli_terms(operator):
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

    def compute_expectation(self, pauli, target: int):
        return self._state.compute_expectation(pauli, target)

    def compute_z_product_expectation(self, targets: Sequence[int]):
        return self._state._compute_z_product_expectation(list(targets))

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

    def apply_noise(self, channel: str, targets: List[int], prob: float):
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

    def apply_noise(self, channel: str, targets: List[int], prob: float):
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
        if name in {"cz", "crx", "cry", "crz"}:
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
                self.apply_noise(channel["type"], list(targets), float(channel["prob"]))

    def apply_noise(self, channel_type: str, targets: List[int], prob: float):
        channel_lower = channel_type.lower()
        for target in targets:
            if channel_lower == "depolarizing":
                self._state.apply_depolarizing_channel(target, prob)
            elif channel_lower == "bit_flip":
                self._state.apply_bit_flip_channel(target, prob)
            else:
                raise ValueError(f"Noise channel '{channel_type}' is not supported by the DensityMatrixBackend.")

    def get_state(self):
        return self._state.get_density_matrix()

    def sample(self, shots: int, qubits: Optional[Sequence[int]] = None):
        raise NotImplementedError(
            "The density-matrix backend does not expose a native sampling path yet. "
            "Use backend='state_vector' for sampling."
        )

    def expectation(self, operator):
        total = 0.0 + 0.0j
        for coefficient, paulis in iter_pauli_terms(operator):
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
