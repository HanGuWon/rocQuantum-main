from __future__ import annotations

# tests/test_backend.py
import math
import os
import sys
import types
import unittest

import numpy as np
from qiskit import QuantumCircuit

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_PROVIDER_ROOT = os.path.dirname(_TEST_DIR)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_PROVIDER_ROOT))
for _path in (_PROJECT_ROOT, _PROVIDER_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:
    from qiskit_rocquantum_provider.backend import RocQuantumBackend
    QISKIT_ROCQ_AVAILABLE = True
except ImportError:
    QISKIT_ROCQ_AVAILABLE = False


class _FakeQuantumSimulator:
    def __init__(self, num_qubits):
        self._num_qubits = int(num_qubits)
        self.state = np.zeros(1 << self._num_qubits, dtype=np.complex128)
        self.state[0] = 1.0

    def reset(self):
        self.__init__(self._num_qubits)

    def num_qubits(self):
        return self._num_qubits

    def apply_gate(self, name, targets, params=None):
        params = list(params or [])
        if name == "H":
            scale = 1.0 / math.sqrt(2.0)
            self._apply_single(np.array([[scale, scale], [scale, -scale]], dtype=np.complex128), targets[0])
        elif name == "RZ":
            angle = float(params[0])
            self._apply_single(
                np.array(
                    [[np.exp(-0.5j * angle), 0.0], [0.0, np.exp(0.5j * angle)]],
                    dtype=np.complex128,
                ),
                targets[0],
            )
        elif name in {"CNOT", "CX"}:
            self._apply_cx(targets[0], targets[1])
        else:
            raise ValueError(f"Unsupported fake gate: {name}")

    def measure(self, qubits, shots):
        if tuple(qubits) == tuple(range(self._num_qubits)) and self._num_qubits == 2:
            return [0 if shot % 2 == 0 else 3 for shot in range(int(shots))]
        return [0 for _ in range(int(shots))]

    def get_statevector(self):
        return self.state.copy()

    def _apply_single(self, matrix, target):
        step = 1 << int(target)
        new_state = self.state.copy()
        for base in range(0, len(self.state), step * 2):
            for offset in range(step):
                i0 = base + offset
                i1 = i0 + step
                a0 = self.state[i0]
                a1 = self.state[i1]
                new_state[i0] = matrix[0, 0] * a0 + matrix[0, 1] * a1
                new_state[i1] = matrix[1, 0] * a0 + matrix[1, 1] * a1
        self.state = new_state

    def _apply_cx(self, control, target):
        control_mask = 1 << int(control)
        target_mask = 1 << int(target)
        new_state = self.state.copy()
        for index in range(len(self.state)):
            if index & control_mask and not index & target_mask:
                flipped = index | target_mask
                new_state[index] = self.state[flipped]
                new_state[flipped] = self.state[index]
        self.state = new_state


def _install_fake_binding_if_needed():
    if "rocquantum_bind" in sys.modules:
        return
    try:
        __import__("rocquantum_bind")
    except ImportError:
        fake = types.ModuleType("rocquantum_bind")
        fake.QuantumSimulator = _FakeQuantumSimulator
        sys.modules["rocquantum_bind"] = fake


@unittest.skipIf(not QISKIT_ROCQ_AVAILABLE, "qiskit-rocquantum-provider is not installed")
class TestRocQuantumBackend(unittest.TestCase):
    def setUp(self):
        _install_fake_binding_if_needed()
        self.backend = RocQuantumBackend()

    def test_bell_state_statevector(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        statevector = self.backend.run(qc).result().get_statevector()
        expected = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
        self.assertTrue(np.allclose(statevector, expected))

    def test_bell_state_counts(self):
        shots = 1000
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])
        counts = self.backend.run(qc, shots=shots).result().get_counts()
        self.assertEqual(set(counts.keys()), {"00", "11"})
        self.assertAlmostEqual(counts.get("00", 0) / shots, 0.5, delta=0.1)

    def test_rz_rotation(self):
        angle = np.pi / 2
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.rz(angle, 0)
        statevector = self.backend.run(qc).result().get_statevector()
        expected = (1/np.sqrt(2)) * np.array([np.exp(-1j*angle/2), np.exp(1j*angle/2)])
        self.assertTrue(np.allclose(statevector, expected))
