"""
P3 Integration Tests - PennyLane Adapter

Mock-based runtime contract tests for the PennyLane device adapter.

    python -m unittest tests.test_pennylane_integration -v
"""

import importlib.util
import os
import sys
import types
import unittest
from unittest import mock

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_DEVICE_PATH = os.path.join(
    _PROJECT_ROOT, "integrations", "pennylane-rocq",
    "pennylane_rocq", "rocq_device.py",
)


class _FakeOperation:
    def __init__(self, name, wires, matrix=None):
        self.name = name
        self.wires = list(wires)
        self.matrix = np.asarray(matrix, dtype=np.complex128) if matrix is not None else None


class _FakeQSim:
    instances = []

    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.applied_gates = []
        self.executed = False
        _FakeQSim.instances.append(self)

    def ApplyGate(self, gate, *indices):
        self.applied_gates.append((gate, tuple(indices)))

    def Execute(self):
        self.executed = True

    def GetStateVector(self):
        size = 1 << self.num_qubits
        out = np.zeros(size, dtype=np.complex128)
        out[0] = 1.0 + 0.0j
        return out


def _build_fake_pennylane_modules():
    qml = types.ModuleType("pennylane")
    operation_mod = types.ModuleType("pennylane.operation")

    class Operation:
        pass

    class QubitDevice:
        def __init__(self, wires, shots=None):
            if isinstance(wires, int):
                self.wires = list(range(wires))
            else:
                self.wires = list(wires)
            self.shots = shots
            self.wire_map = {wire: idx for idx, wire in enumerate(self.wires)}

        def marginal_prob(self, probs, wires_to_trace):
            probs = np.asarray(probs)
            num_wires = int(np.log2(probs.size))
            reduced = probs.reshape([2] * num_wires)
            for axis in sorted(wires_to_trace, reverse=True):
                reduced = reduced.sum(axis=axis)
            return reduced.reshape(-1)

    def _matrix(op):
        return np.asarray(op.matrix, dtype=np.complex128)

    operation_mod.Operation = Operation
    qml.QubitDevice = QubitDevice
    qml.matrix = _matrix
    return qml, operation_mod


def _load_device_module():
    fake_bind = types.ModuleType("rocquantum_bind")
    fake_bind.QSim = _FakeQSim
    qml, operation_mod = _build_fake_pennylane_modules()
    module_name = "test_pennylane_adapter_module"

    with mock.patch.dict(
        sys.modules,
        {
            "pennylane": qml,
            "pennylane.operation": operation_mod,
            "rocquantum_bind": fake_bind,
        },
        clear=False,
    ):
        sys.modules.pop(module_name, None)
        spec = importlib.util.spec_from_file_location(module_name, _DEVICE_PATH)
        module = importlib.util.module_from_spec(spec)
        if spec.loader is None:
            raise RuntimeError("Unable to load PennyLane adapter module.")
        spec.loader.exec_module(module)
    return module


class TestPennyLaneAdapterRuntime(unittest.TestCase):
    def setUp(self):
        _FakeQSim.instances.clear()

    def test_apply_dispatches_named_and_matrix_ops(self):
        module = _load_device_module()
        device = module.RocQDevice(wires=[0, 1], shots=5)

        matrix = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        operations = [
            _FakeOperation("Hadamard", [0]),
            _FakeOperation("CNOT", [0, 1]),
            _FakeOperation("RX", [1], matrix=matrix),
        ]

        device.apply(operations)
        qsim = _FakeQSim.instances[-1]

        self.assertEqual(qsim.applied_gates[0], ("H", (0,)))
        self.assertEqual(qsim.applied_gates[1], ("CNOT", (0, 1)))
        matrix_call, matrix_indices = qsim.applied_gates[2]
        self.assertEqual(matrix_indices, (1,))
        np.testing.assert_allclose(matrix_call, matrix)
        self.assertTrue(qsim.executed)
        self.assertEqual(len(device.state), 4)

    def test_unsupported_operation_raises(self):
        module = _load_device_module()
        device = module.RocQDevice(wires=[0], shots=2)

        with self.assertRaises(NotImplementedError):
            device.apply([_FakeOperation("FooGate", [0])])

    def test_generate_samples_returns_binary_rows(self):
        module = _load_device_module()
        device = module.RocQDevice(wires=[0, 1], shots=4)
        device._state = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)

        with mock.patch("numpy.random.multinomial", return_value=np.array([1, 2, 1, 0], dtype=int)):
            samples = device.generate_samples()

        self.assertEqual(samples.shape, (4, 2))
        self.assertTrue(set(np.unique(samples)).issubset({0, 1}))

    def test_analytic_probability_respects_wire_subset(self):
        module = _load_device_module()
        device = module.RocQDevice(wires=[0, 1], shots=4)
        amp = 1.0 / np.sqrt(2.0)
        device._state = np.array([0.0, amp, amp, 0.0], dtype=np.complex128)

        subset_probs = device.analytic_probability(wires=[0])
        self.assertEqual(len(subset_probs), 2)
        self.assertAlmostEqual(float(np.sum(subset_probs)), 1.0)


if __name__ == "__main__":
    unittest.main()
