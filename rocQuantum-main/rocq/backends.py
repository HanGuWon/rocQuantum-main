from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

try:
    import _rocq_hip_backend as hip_backend
except ImportError:
    hip_backend = None

try:
    import rocq_hip as dm_backend
except ImportError:
    dm_backend = None


class _MockStateVectorState:
    def __init__(self, n_qubits):
        print(f"MOCK C++ SV: Initializing for {n_qubits} qubits.")

    def apply_h(self, t):
        print(f"MOCK C++ SV: apply_h on qubit {t}")

    def apply_x(self, t):
        print(f"MOCK C++ SV: apply_x on qubit {t}")

    def apply_y(self, t):
        print(f"MOCK C++ SV: apply_y on qubit {t}")

    def apply_z(self, t):
        print(f"MOCK C++ SV: apply_z on qubit {t}")

    def apply_cnot(self, c, t):
        print(f"MOCK C++ SV: apply_cnot on control={c}, target={t}")

    def apply_ry(self, angle, t):
        print(f"MOCK C++ SV: apply_ry(theta={angle}) on qubit {t}")

    def apply_rz(self, angle, t):
        print(f"MOCK C++ SV: apply_rz(phi={angle}) on qubit {t}")

    def get_state_vector(self):
        print("MOCK C++ SV: get_state_vector()")
        return "mock_cpp_state_vector_data"


class _MockDensityMatrixState:
    def __init__(self, n_qubits):
        print(f"MOCK C++ DM: Initializing for {n_qubits} qubits.")

    def apply_gate(self, gate_matrix, target_qubit, adjoint=False):
        print(f"MOCK C++ DM: apply_gate on qubit {target_qubit} (adjoint={adjoint})")

    def apply_cnot(self, c, t):
        print(f"MOCK C++ DM: apply_cnot on control={c}, target={t}")

    def apply_depolarizing_channel(self, target, prob):
        print(f"MOCK C++ DM: apply_depolarizing_channel on {target} with prob={prob}")

    def apply_bit_flip_channel(self, target, prob):
        print(f"MOCK C++ DM: apply_bit_flip_channel on {target} with prob={prob}")

    def get_density_matrix(self):
        print("MOCK C++ DM: get_density_matrix()")
        return "mock_cpp_density_matrix_data"


class _HipStateVectorState:
    def __init__(self, n_qubits: int):
        if hip_backend is None:
            raise RuntimeError("rocq state-vector backend is not available.")

        self._handle = hip_backend.RocsvHandle()
        self._num_qubits = n_qubits
        self._d_state = hip_backend.allocate_state_internal(self._handle, n_qubits, 1)
        status = hip_backend.initialize_state(self._handle, self._d_state, n_qubits)
        if status != hip_backend.rocqStatus.SUCCESS:
            raise RuntimeError(f"rocsvInitializeState failed: {status}")

    def apply_named_gate(self, op_name: str, targets: List[int], params: Optional[Dict[str, float]] = None):
        params = params or {}
        op = op_name.lower()

        def _call(func, *args):
            status = func(*args)
            if status != hip_backend.rocqStatus.SUCCESS:
                raise RuntimeError(f"hipStateVec call failed for '{op_name}': {status}")
            return status

        def _angle(primary_key: str, secondary_key: str) -> Optional[float]:
            if isinstance(params, dict):
                return params.get(primary_key, params.get(secondary_key))
            if isinstance(params, (list, tuple)) and params:
                return params[0]
            return None

        if op == "h":
            return _call(hip_backend.apply_h, self._handle, self._d_state, self._num_qubits, targets[0])
        if op == "x":
            return _call(hip_backend.apply_x, self._handle, self._d_state, self._num_qubits, targets[0])
        if op == "y":
            return _call(hip_backend.apply_y, self._handle, self._d_state, self._num_qubits, targets[0])
        if op == "z":
            return _call(hip_backend.apply_z, self._handle, self._d_state, self._num_qubits, targets[0])
        if op == "s":
            return _call(hip_backend.apply_s, self._handle, self._d_state, self._num_qubits, targets[0])
        if op == "sdg":
            return _call(hip_backend.apply_sdg, self._handle, self._d_state, self._num_qubits, targets[0])
        if op == "t":
            return _call(hip_backend.apply_t, self._handle, self._d_state, self._num_qubits, targets[0])
        if op == "rx":
            angle = _angle("theta", "phi")
            if angle is None:
                raise ValueError("Gate 'rx' requires a rotation angle.")
            return _call(hip_backend.apply_rx, self._handle, self._d_state, self._num_qubits, targets[0], angle)
        if op == "ry":
            angle = _angle("theta", "phi")
            if angle is None:
                raise ValueError("Gate 'ry' requires a rotation angle.")
            return _call(hip_backend.apply_ry, self._handle, self._d_state, self._num_qubits, targets[0], angle)
        if op == "rz":
            angle = _angle("phi", "theta")
            if angle is None:
                raise ValueError("Gate 'rz' requires a rotation angle.")
            return _call(hip_backend.apply_rz, self._handle, self._d_state, self._num_qubits, targets[0], angle)
        if op == "cnot":
            return _call(hip_backend.apply_cnot, self._handle, self._d_state, self._num_qubits, targets[0], targets[1])
        if op == "cz":
            return _call(hip_backend.apply_cz, self._handle, self._d_state, self._num_qubits, targets[0], targets[1])
        if op == "swap":
            return _call(hip_backend.apply_swap, self._handle, self._d_state, self._num_qubits, targets[0], targets[1])
        if op == "crx":
            angle = _angle("theta", "phi")
            if angle is None:
                raise ValueError("Gate 'crx' requires a rotation angle.")
            return _call(hip_backend.apply_crx, self._handle, self._d_state, self._num_qubits, targets[0], targets[1], angle)
        if op == "cry":
            angle = _angle("theta", "phi")
            if angle is None:
                raise ValueError("Gate 'cry' requires a rotation angle.")
            return _call(hip_backend.apply_cry, self._handle, self._d_state, self._num_qubits, targets[0], targets[1], angle)
        if op == "crz":
            angle = _angle("phi", "theta")
            if angle is None:
                raise ValueError("Gate 'crz' requires a rotation angle.")
            return _call(hip_backend.apply_crz, self._handle, self._d_state, self._num_qubits, targets[0], targets[1], angle)

        raise ValueError(f"Gate '{op_name}' is not supported by the hipStateVec backend.")

    def get_state_vector(self):
        return hip_backend.get_state_vector_full(self._handle, self._d_state, self._num_qubits, 1)


class _HipDensityMatrixState:
    def __init__(self, n_qubits: int):
        if dm_backend is None:
            raise RuntimeError("rocq density-matrix backend is not available.")
        self._state = dm_backend.DensityMatrixState(n_qubits)

    def apply_gate_matrix(self, matrix: np.ndarray, target: int, adjoint: bool = False):
        return self._state.apply_gate(matrix, target, adjoint)

    def apply_cnot(self, control: int, target: int):
        return self._state.apply_cnot(control, target)

    def apply_depolarizing_channel(self, target: int, prob: float):
        return self._state.apply_depolarizing_channel(target, prob)

    def apply_bit_flip_channel(self, target: int, prob: float):
        return self._state.apply_bit_flip_channel(target, prob)

    def get_density_matrix(self):
        return self._state.get_density_matrix()


class _BaseBackend:
    """Abstract base class for a quantum simulation backend."""

    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits

    def apply_gate(self, op_name: str, targets: List[int], params: Optional[Dict[str, float]] = None):
        raise NotImplementedError

    def apply_noise(self, channel: str, targets: List[int], prob: float):
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError


class StateVectorBackend(_BaseBackend):
    """Simulates a quantum state vector by dispatching to hipStateVec."""

    def __init__(self, num_qubits: int):
        super().__init__(num_qubits)
        if hip_backend is None:
            self._state = _MockStateVectorState(num_qubits)
            self._uses_mock = True
        else:
            self._state = _HipStateVectorState(num_qubits)
            self._uses_mock = False

    def apply_gate(self, op_name: str, targets: List[int], params: Optional[Dict[str, float]] = None):
        if self._uses_mock:
            op = op_name.lower()
            if op == "h":
                return self._state.apply_h(targets[0])
            if op == "x":
                return self._state.apply_x(targets[0])
            if op == "y":
                return self._state.apply_y(targets[0])
            if op == "z":
                return self._state.apply_z(targets[0])
            if op == "cnot":
                return self._state.apply_cnot(targets[0], targets[1])
            if op == "ry":
                return self._state.apply_ry(params["theta"], targets[0])
            if op == "rz":
                return self._state.apply_rz(params["phi"], targets[0])
            raise ValueError(f"Gate '{op_name}' is not supported by the mock StateVectorBackend.")
        return self._state.apply_named_gate(op_name, targets, params)

    def apply_noise(self, channel: str, targets: List[int], prob: float):
        raise NotImplementedError("Noise models are only supported by the 'density_matrix' backend.")

    def get_state(self):
        return self._state.get_state_vector()


class DensityMatrixBackend(_BaseBackend):
    """Simulates a quantum system using a density matrix backend."""

    def __init__(self, num_qubits: int):
        super().__init__(num_qubits)
        if dm_backend is None:
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
        if op in ("rx", "ry", "rz") and param is not None:
            c = math.cos(param / 2.0)
            s = math.sin(param / 2.0)
            if op == "rx":
                return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex64)
            if op == "ry":
                return np.array([[c, -s], [s, c]], dtype=np.complex64)
            return np.array([[c - 1j * s, 0], [0, c + 1j * s]], dtype=np.complex64)
        raise ValueError(f"Gate '{op}' is not supported by the density matrix backend.")

    def apply_gate(self, op_name: str, targets: List[int], params: Optional[Dict[str, float]] = None):
        params = params or {}
        op = op_name.lower()
        if op == "cnot":
            return self._state.apply_cnot(targets[0], targets[1])
        if isinstance(params, dict):
            angle = params.get("theta", params.get("phi"))
        elif isinstance(params, (list, tuple)) and params:
            angle = params[0]
        else:
            angle = None
        gate = self._gate_matrix(op, angle)
        return self._state.apply_gate_matrix(gate, targets[0])

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


def get_backend(backend_name: str, num_qubits: int) -> _BaseBackend:
    """Factory function to instantiate a simulation backend."""

    supported = {"state_vector": StateVectorBackend, "density_matrix": DensityMatrixBackend}
    if backend_name not in supported:
        raise ValueError(f"Unsupported backend '{backend_name}'. Supported backends are: {list(supported.keys())}")
    return supported[backend_name](num_qubits)
