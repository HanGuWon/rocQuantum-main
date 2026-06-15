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
_PACKAGE_DIR = os.path.dirname(_ADAPTER_PATH)
_PACKAGE_INIT_PATH = os.path.join(_PACKAGE_DIR, "__init__.py")


class _MeasurementGate:
    def __init__(self, key="m"):
        self.key = key


class _MatrixGate:
    def __init__(self, matrix):
        self.matrix = np.asarray(matrix, dtype=np.complex128)


class _RxGate:
    def __init__(self, rads):
        self._rads = rads


class _RyGate:
    def __init__(self, rads):
        self._rads = rads


class _RzGate:
    def __init__(self, rads):
        self._rads = rads


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
        self.measured = []
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

    def measure(self, qubits, shots):
        self.measured.append((tuple(qubits), int(shots)))
        return [0, 1, 1][: int(shots)]


class _FakeQuantumSimulator:
    instances = []

    def __init__(self, num_qubits):
        self._num_qubits = num_qubits
        self.applied_gates = []
        self.applied_matrices = []
        self.measured = []
        _FakeQuantumSimulator.instances.append(self)

    def num_qubits(self):
        return self._num_qubits

    def apply_gate(self, name, targets, params=None):
        self.applied_gates.append((name, tuple(targets), tuple(params or ())))

    def apply_matrix(self, matrix, targets):
        self.applied_matrices.append((np.asarray(matrix), tuple(targets)))

    def get_statevector(self):
        size = 1 << self._num_qubits
        out = np.zeros(size, dtype=np.complex128)
        out[0] = 1.0 + 0.0j
        return out

    def measure(self, qubits, shots):
        self.measured.append((tuple(qubits), int(shots)))
        return [0, 1, 1][: int(shots)]


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


def _load_adapter_module(include_modern_simulator=False):
    fake_cirq = _build_fake_cirq_module()
    fake_bind = types.ModuleType("rocquantum_bind")
    fake_bind.QSim = _FakeQSim
    if include_modern_simulator:
        fake_bind.QuantumSimulator = _FakeQuantumSimulator
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
        _FakeQuantumSimulator.instances.clear()

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

    def test_prefers_quantum_simulator_when_binding_exposes_it(self):
        adapter, cirq = _load_adapter_module(include_modern_simulator=True)
        sim = adapter.RocQuantumSimulator()

        matrix = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        circuit = _FakeCircuit([
            _FakeOp(cirq.X, 0),
            _FakeOp(cirq.MatrixGate(matrix), 0),
        ])

        state = sim._get_final_statevector(circuit, [0])
        modern = _FakeQuantumSimulator.instances[-1]

        self.assertFalse(_FakeQSim.instances)
        self.assertEqual(modern.applied_gates, [("X", (0,), ())])
        matrix_call, matrix_indices = modern.applied_matrices[0]
        self.assertEqual(matrix_indices, (0,))
        np.testing.assert_allclose(matrix_call, matrix)
        self.assertEqual(len(state), 2)

    def test_cirq_rotations_dispatch_to_native_modern_runtime(self):
        adapter, cirq = _load_adapter_module(include_modern_simulator=True)
        sim = adapter.RocQuantumSimulator()

        circuit = _FakeCircuit([
            _FakeOp(cirq.Rx(0.125), 0),
            _FakeOp(cirq.Ry(0.25), 0),
            _FakeOp(cirq.Rz(0.5), 0),
        ])

        sim._get_final_statevector(circuit, [0])
        modern = _FakeQuantumSimulator.instances[-1]

        self.assertEqual(
            modern.applied_gates,
            [
                ("RX", (0,), (0.125,)),
                ("RY", (0,), (0.25,)),
                ("RZ", (0,), (0.5,)),
            ],
        )
        self.assertFalse(modern.applied_matrices)

    def test_cirq_rotations_fall_back_to_matrix_on_legacy_qsim(self):
        adapter, cirq = _load_adapter_module()
        sim = adapter.RocQuantumSimulator()

        circuit = _FakeCircuit([_FakeOp(cirq.Rx(0.125), 0)])

        sim._get_final_statevector(circuit, [0])
        qsim = _FakeQSim.instances[-1]

        matrix_call, matrix_indices = qsim.applied_gates[0]
        self.assertEqual(matrix_indices, (0,))
        np.testing.assert_allclose(matrix_call, np.eye(2, dtype=np.complex128))

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

        result = sim._run(circuit, param_resolver=None, repetitions=3)
        qsim = _FakeQSim.instances[-1]

        self.assertIn("m", result)
        self.assertEqual(result["m"].shape, (3, 1))
        self.assertEqual(qsim.measured, [((0,), 3)])
        np.testing.assert_array_equal(result["m"][:, 0], np.array([0, 1, 1], dtype=np.int64))

    def test_package_import_defers_missing_native_binding_error_until_execution(self):
        fake_cirq = _build_fake_cirq_module()
        module_name = "cirq_rocm"
        submodule_name = "cirq_rocm.roc_quantum_simulator"

        with mock.patch.dict(sys.modules, {"cirq": fake_cirq, "rocquantum_bind": None}, clear=False):
            sys.modules.pop(module_name, None)
            sys.modules.pop(submodule_name, None)
            spec = importlib.util.spec_from_file_location(
                module_name,
                _PACKAGE_INIT_PATH,
                submodule_search_locations=[_PACKAGE_DIR],
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            if spec.loader is None:
                raise RuntimeError("Unable to load Cirq package module.")
            spec.loader.exec_module(module)

            sim = module.RocQuantumSimulator()
            with self.assertRaises(ImportError) as ctx:
                sim._get_final_statevector(_FakeCircuit([]), [])

        self.assertIn("rocquantum_bind", str(ctx.exception))
        sys.modules.pop(module_name, None)
        sys.modules.pop(submodule_name, None)


if __name__ == "__main__":
    unittest.main()
