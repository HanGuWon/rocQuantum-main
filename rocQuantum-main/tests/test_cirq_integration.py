"""
P3 Integration Tests - Cirq Adapter

Mock-based runtime contract tests for the Cirq adapter.

    python -m unittest tests.test_cirq_integration -v
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

_ADAPTER_PATH = os.path.join(
    _PROJECT_ROOT, "integrations", "cirq-rocm", "cirq_rocm",
    "roc_quantum_simulator.py",
)


class _MeasurementGate:
    def __init__(self, key="m"):
        self.key = key


class _MatrixGate:
    def __init__(self, matrix):
        self.matrix = np.asarray(matrix, dtype=np.complex128)


class _RxGate:
    pass


class _RyGate:
    pass


class _RzGate:
    pass


class _XGate:
    pass


class _YGate:
    pass


class _ZGate:
    pass


class _HGate:
    pass


class _SGate:
    pass


class _TGate:
    pass


class _CnotGate:
    pass


class _CzGate:
    pass


class _FakeOp:
    def __init__(self, gate, *qubits):
        self.gate = gate
        self.qubits = tuple(qubits)


class _FakeCircuit:
    def __init__(self, ops):
        self._ops = list(ops)

    def all_operations(self):
        return list(self._ops)

    def all_qubits(self):
        qubits = []
        for op in self._ops:
            qubits.extend(op.qubits)
        return set(qubits)


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


def _build_fake_cirq_module():
    cirq = types.ModuleType("cirq")

    class _SimulatesFinalState:
        pass

    class _SimulatesSamples:
        pass

    class _StateVectorTrialResult:
        def __init__(self, params, measurements, final_simulator_state):
            self.params = params
            self.measurements = measurements
            self.final_simulator_state = final_simulator_state

    cirq.SimulatesFinalState = _SimulatesFinalState
    cirq.SimulatesSamples = _SimulatesSamples
    cirq.StateVectorTrialResult = _StateVectorTrialResult

    cirq.MeasurementGate = _MeasurementGate
    cirq.MatrixGate = _MatrixGate
    cirq.Rx = _RxGate
    cirq.Ry = _RyGate
    cirq.Rz = _RzGate

    cirq.X = _XGate()
    cirq.Y = _YGate()
    cirq.Z = _ZGate()
    cirq.H = _HGate()
    cirq.S = _SGate()
    cirq.T = _TGate()
    cirq.CNOT = _CnotGate()
    cirq.CZ = _CzGate()

    def _unitary(op):
        gate = op.gate
        if hasattr(gate, "matrix"):
            return np.asarray(gate.matrix, dtype=np.complex128)
        return np.eye(2, dtype=np.complex128)

    cirq.unitary = _unitary
    return cirq


def _load_adapter_module():
    fake_cirq = _build_fake_cirq_module()
    fake_bind = types.ModuleType("rocquantum_bind")
    fake_bind.QSim = _FakeQSim
    module_name = "test_cirq_adapter_module"

    with mock.patch.dict(sys.modules, {"cirq": fake_cirq, "rocquantum_bind": fake_bind}, clear=False):
        sys.modules.pop(module_name, None)
        spec = importlib.util.spec_from_file_location(module_name, _ADAPTER_PATH)
        module = importlib.util.module_from_spec(spec)
        if spec.loader is None:
            raise RuntimeError("Unable to load Cirq adapter module.")
        spec.loader.exec_module(module)
    return module, fake_cirq


class TestCirqAdapterRuntime(unittest.TestCase):
    def setUp(self):
        _FakeQSim.instances.clear()

    def test_core_and_matrix_gates_dispatch_to_qsim(self):
        adapter, cirq = _load_adapter_module()
        sim = adapter.RocQuantumSimulator()

        matrix = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        circuit = _FakeCircuit([
            _FakeOp(cirq.X, 0),
            _FakeOp(cirq.CNOT, 0, 1),
            _FakeOp(cirq.MatrixGate(matrix), 1),
        ])

        state = sim._get_final_statevector(circuit, [0, 1])
        qsim = _FakeQSim.instances[-1]

        self.assertEqual(qsim.applied_gates[0], ("X", (0,)))
        self.assertEqual(qsim.applied_gates[1], ("CNOT", (0, 1)))
        matrix_call, matrix_indices = qsim.applied_gates[2]
        self.assertEqual(matrix_indices, (1,))
        np.testing.assert_allclose(matrix_call, matrix)
        self.assertTrue(qsim.executed)
        self.assertEqual(len(state), 4)

    def test_unsupported_gate_raises(self):
        adapter, _ = _load_adapter_module()
        sim = adapter.RocQuantumSimulator()

        class _UnsupportedGate:
            pass

        circuit = _FakeCircuit([_FakeOp(_UnsupportedGate(), 0)])
        with self.assertRaises(TypeError):
            sim._get_final_statevector(circuit, [0])

    def test_run_collects_measurement_results_by_key(self):
        adapter, cirq = _load_adapter_module()
        sim = adapter.RocQuantumSimulator()

        circuit = _FakeCircuit([
            _FakeOp(cirq.MeasurementGate("m"), 0),
        ])

        with mock.patch.object(
            adapter.RocQuantumSimulator,
            "_get_final_statevector",
            return_value=np.array([0.5 + 0.0j, 0.5 + 0.0j], dtype=np.complex128),
        ):
            with mock.patch("numpy.random.choice", return_value=np.array([0, 1, 1], dtype=np.int64)):
                result = sim._run(circuit, param_resolver=None, repetitions=3)

        self.assertIn("m", result)
        self.assertEqual(result["m"].shape, (3, 1))
        np.testing.assert_array_equal(result["m"][:, 0], np.array([0, 1, 1], dtype=np.int64))


if __name__ == "__main__":
    unittest.main()
