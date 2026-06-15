"""Contract tests for the legacy ``python/rocq`` API surface."""

from __future__ import annotations

import importlib.util
import os
import sys
import types
import unittest
from unittest import mock

import numpy as np


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LEGACY_ROOT = os.path.join(_PROJECT_ROOT, "python", "rocq")
_API_PATH = os.path.join(_LEGACY_ROOT, "api.py")


def _fake_backend(with_fusion=True):
    backend = types.ModuleType("_rocq_hip_backend")
    backend.calls = []

    class _Status:
        SUCCESS = 0
        NOT_IMPLEMENTED = 5

    backend.rocqStatus = _Status
    backend.DeviceBuffer = type("DeviceBuffer", (), {})
    backend.GateOp = type("GateOp", (), {})

    class _MLIRCompiler:
        def __init__(self):
            self.module_name = None
            self.module_string = ""

        def initialize_module(self, name):
            self.module_name = name
            backend.calls.append(("initialize_module", (name,)))
            return True

        def load_module_from_string(self, mlir_string):
            self.module_string = mlir_string
            backend.calls.append(("load_module", (mlir_string,)))
            return True

        def get_module_string(self):
            return self.module_string

        def dump_module(self):
            backend.calls.append(("dump_module", (self.module_string,)))

    backend.MLIRCompiler = _MLIRCompiler

    def _record(name, value):
        def _helper(*args):
            backend.calls.append((name, args))
            return value
        return _helper

    backend.get_expectation_value_x = _record("x", 0.5)
    backend.get_expectation_value_y = _record("y", -0.25)
    backend.get_expectation_value_z = _record("z", 0.75)
    backend.get_expectation_value_pauli_product_z = _record("zz", -1.0)
    backend.get_expectation_pauli_string = _record("pauli", 0.125)
    backend.allocate_state_internal = _record("allocate_state", "device-state")
    backend.initialize_state = _record("initialize_state", _Status.SUCCESS)
    backend.allocate_distributed_state = _record("allocate_distributed_state", _Status.SUCCESS)
    backend.initialize_distributed_state = _record("initialize_distributed_state", _Status.SUCCESS)
    backend.get_state_vector_full = _record(
        "state_full",
        np.array([1, 0, 0, 0, 0, 1, 0, 0], dtype=np.complex64),
    )

    for gate_name in ("x", "y", "z", "h", "s", "t", "rx", "ry", "rz", "cnot", "cz", "swap", "crx", "cry", "crz", "mcx", "cswap"):
        setattr(backend, f"apply_{gate_name}", _record(f"apply_{gate_name}", _Status.SUCCESS))

    if with_fusion:
        class _GateFusion:
            def __init__(self, *args):
                backend.calls.append(("fusion_init", args))

            def process_queue(self, queue):
                payload = tuple(
                    (op.name, tuple(op.controls), tuple(op.targets), tuple(op.params))
                    for op in queue
                )
                backend.calls.append(("fusion", payload))
                return _Status.SUCCESS

        backend.GateFusion = _GateFusion

    def _forbidden_statevector_readback(*args):
        raise AssertionError("legacy Circuit.expval must not read back the statevector")

    backend.get_state_vector_slice = _forbidden_statevector_readback
    return backend


def _load_legacy_api(backend):
    package_name = "_test_legacy_rocq"
    for name in list(sys.modules):
        if name == package_name or name.startswith(f"{package_name}."):
            del sys.modules[name]

    package = types.ModuleType(package_name)
    package.__path__ = [_LEGACY_ROOT]
    sys.modules[package_name] = package
    sys.modules[f"{package_name}._rocq_hip_backend"] = backend

    spec = importlib.util.spec_from_file_location(f"{package_name}.api", _API_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    if spec.loader is None:
        raise RuntimeError("Unable to load legacy rocq API module.")
    spec.loader.exec_module(module)
    return module


def _make_simulator(module):
    simulator = module.Simulator.__new__(module.Simulator)
    simulator._handle_wrapper = "handle"
    simulator._active_circuits = 0
    return simulator


def _make_circuit(module):
    circuit = module.Circuit.__new__(module.Circuit)
    circuit.num_qubits = 3
    circuit._sim_handle = "handle"
    circuit._d_state_buffer = "device-state"
    circuit.is_multi_gpu = False
    circuit.batch_size = 1
    circuit._gate_queue = []
    circuit._is_dirty = False
    circuit._fusion_engine = None
    return circuit


class TestLegacyCircuitGateFusion(unittest.TestCase):
    def test_flush_uses_native_gate_fusion_for_cnot_adjacent_span(self):
        backend = _fake_backend()
        module = _load_legacy_api(backend)
        circuit = _make_circuit(module)

        circuit.h(0)
        circuit.cx(0, 1)
        circuit.rz(0.25, 1)
        circuit.flush()

        calls_by_name = [name for name, _ in backend.calls]
        self.assertIn("fusion_init", calls_by_name)
        self.assertIn("fusion", calls_by_name)
        self.assertNotIn("apply_h", calls_by_name)
        self.assertNotIn("apply_cnot", calls_by_name)
        self.assertNotIn("apply_rz", calls_by_name)
        fusion_payload = next(payload for name, payload in backend.calls if name == "fusion")
        self.assertEqual(
            fusion_payload,
            (
                ("H", (), (0,), ()),
                ("CNOT", (0,), (1,), ()),
                ("RZ", (), (1,), (0.25,)),
            ),
        )
        self.assertFalse(circuit._is_dirty)
        self.assertEqual(circuit._gate_queue, [])

    def test_flush_uses_native_gate_fusion_for_same_target_single_qubit_span(self):
        backend = _fake_backend()
        module = _load_legacy_api(backend)
        circuit = _make_circuit(module)

        circuit.h(0)
        circuit.rz(0.25, 0)
        circuit.x(1)
        circuit.flush()

        calls_by_name = [name for name, _ in backend.calls]
        self.assertIn("fusion", calls_by_name)
        self.assertNotIn("apply_h", calls_by_name)
        self.assertNotIn("apply_rz", calls_by_name)
        self.assertIn("apply_x", calls_by_name)
        fusion_payload = next(payload for name, payload in backend.calls if name == "fusion")
        self.assertEqual(
            fusion_payload,
            (
                ("H", (), (0,), ()),
                ("RZ", (), (0,), (0.25,)),
            ),
        )
        self.assertFalse(circuit._is_dirty)
        self.assertEqual(circuit._gate_queue, [])

    def test_flush_replays_gates_when_fusion_is_disabled(self):
        backend = _fake_backend()
        module = _load_legacy_api(backend)
        circuit = _make_circuit(module)

        circuit.h(0)
        circuit.cx(0, 1)
        with mock.patch.dict(os.environ, {"ROCQ_DISABLE_GATE_FUSION": "1"}):
            circuit.flush()

        calls_by_name = [name for name, _ in backend.calls]
        self.assertNotIn("fusion_init", calls_by_name)
        self.assertNotIn("fusion", calls_by_name)
        self.assertEqual(calls_by_name, ["apply_h", "apply_cnot"])


class TestLegacyCircuitBatchState(unittest.TestCase):
    def test_constructor_allocates_requested_batch_size(self):
        backend = _fake_backend()
        module = _load_legacy_api(backend)
        simulator = _make_simulator(module)

        circuit = module.Circuit(np.int64(2), simulator, batch_size=np.int64(3))

        self.assertEqual(circuit.batch_size, 3)
        self.assertEqual(circuit.num_qubits, 2)
        self.assertEqual(circuit._d_state_buffer, "device-state")
        self.assertIn(("allocate_state", ("handle", 2, 3)), backend.calls)
        self.assertIn(("initialize_state", ("handle", "device-state", 2)), backend.calls)

    def test_constructor_rejects_ambiguous_circuit_options(self):
        backend = _fake_backend()
        module = _load_legacy_api(backend)
        simulator = _make_simulator(module)

        for num_qubits in (True, 2.0, "2", -1):
            with self.subTest(num_qubits=num_qubits):
                with self.assertRaisesRegex(ValueError, "Number of qubits"):
                    module.Circuit(num_qubits, simulator)

        with self.assertRaisesRegex(ValueError, "multi_gpu must be a boolean"):
            module.Circuit(2, simulator, multi_gpu="yes")

    def test_constructor_rejects_invalid_batch_size(self):
        backend = _fake_backend()
        module = _load_legacy_api(backend)
        simulator = _make_simulator(module)

        with self.assertRaisesRegex(ValueError, "batch_size must be a positive integer"):
            module.Circuit(2, simulator, batch_size=0)

        with self.assertRaisesRegex(ValueError, "batch_size must be a positive integer"):
            module.Circuit(2, simulator, batch_size=True)

        with self.assertRaisesRegex(NotImplementedError, "multi_gpu=True"):
            module.Circuit(2, simulator, multi_gpu=True, batch_size=2)

    def test_gate_methods_validate_targets_and_angles_before_backend_dispatch(self):
        backend = _fake_backend()
        module = _load_legacy_api(backend)
        circuit = _make_circuit(module)

        for target in (True, 0.5, "0", -1, 3):
            with self.subTest(target=target):
                with self.assertRaisesRegex(ValueError, "target qubit"):
                    circuit.x(target)

        for angle in (True, "0.125", np.nan, np.inf):
            with self.subTest(angle=angle):
                with self.assertRaisesRegex(ValueError, "angle"):
                    circuit.rx(angle, 0)

        with self.assertRaisesRegex(ValueError, "Control and target"):
            circuit.cx(0, 0)
        with self.assertRaisesRegex(ValueError, "distinct"):
            circuit.ccx(0, 1, 1)
        with self.assertRaisesRegex(ValueError, "distinct"):
            circuit.cswap(0, 1, 1)

        circuit.ry(np.float64(0.125), np.int64(1))
        self.assertEqual(circuit._gate_queue[-1].targets, [1])
        self.assertEqual(circuit._gate_queue[-1].params, [0.125])

    def test_sample_validates_qubits_and_shots_before_backend_dispatch(self):
        backend = _fake_backend()
        module = _load_legacy_api(backend)
        circuit = _make_circuit(module)

        invalid_qubit_lists = ([], [True], [0.5], ["0"], [-1], [0, 0])
        for measured_qubits in invalid_qubit_lists:
            with self.subTest(measured_qubits=measured_qubits):
                with self.assertRaisesRegex(ValueError, "measured_qubits|List of measured_qubits"):
                    circuit.sample(measured_qubits, 10)

        for shots in (0, -1, True, 1.5, "10"):
            with self.subTest(shots=shots):
                with self.assertRaisesRegex(ValueError, "Number of shots"):
                    circuit.sample([0], shots)

    def test_multi_gpu_constructor_warns_about_partial_support(self):
        backend = _fake_backend()
        module = _load_legacy_api(backend)
        simulator = _make_simulator(module)

        with self.assertWarnsRegex(module.ExperimentalMultiGpuWarning, "experimental partial") as warning:
            circuit = module.Circuit(2, simulator, multi_gpu=True)

        self.assertTrue(circuit.is_multi_gpu)
        self.assertEqual(circuit.execution_notes, [str(warning.warning)])
        self.assertIn(("allocate_distributed_state", ("handle", 2)), backend.calls)
        self.assertIn(("initialize_distributed_state", ("handle",)), backend.calls)

    def test_multi_node_constructor_fails_fast(self):
        backend = _fake_backend()
        module = _load_legacy_api(backend)
        simulator = _make_simulator(module)

        with self.assertRaisesRegex(NotImplementedError, "multi-node distributed execution is not implemented"):
            module.Circuit(2, simulator, multi_node=True)

        with self.assertRaisesRegex(NotImplementedError, "multi-node distributed execution is not implemented"):
            module.Circuit(2, simulator, node_count=2)

        with self.assertRaisesRegex(ValueError, "node_count must be a positive integer"):
            module.Circuit(2, simulator, node_count=0)

    def test_batched_statevector_readback_can_return_slice_or_full_batch(self):
        backend = _fake_backend()
        backend.get_state_vector_slice = lambda *args: (
            backend.calls.append(("state_slice", args)) or np.array([0, 1, 0, 0], dtype=np.complex64)
        )
        module = _load_legacy_api(backend)
        circuit = _make_circuit(module)
        circuit.num_qubits = 2
        circuit.batch_size = 2

        state = circuit.get_statevector(batch_index=1)
        states = circuit.get_statevectors()

        np.testing.assert_array_equal(state, np.array([0, 1, 0, 0], dtype=np.complex64))
        np.testing.assert_array_equal(
            states,
            np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.complex64),
        )
        self.assertIn(("state_slice", ("handle", "device-state", 2, 2, 1)), backend.calls)
        self.assertIn(("state_full", ("handle", "device-state", 2, 2)), backend.calls)

    def test_batched_statevector_readback_validates_slice_index(self):
        backend = _fake_backend()
        module = _load_legacy_api(backend)
        circuit = _make_circuit(module)
        circuit.batch_size = 2

        with self.assertRaisesRegex(ValueError, r"batch_index must be in \[0, 2\)"):
            circuit.get_statevector(batch_index=2)

        with self.assertRaisesRegex(TypeError, "batch_index must be an integer"):
            circuit.get_statevector(batch_index=True)


class TestLegacyCircuitExpectation(unittest.TestCase):
    def test_legacy_pauli_operator_validates_coefficients(self):
        backend = _fake_backend()
        module = _load_legacy_api(backend)

        for coeff in (True, "1.0", np.nan, np.inf):
            with self.subTest(coeff=coeff):
                with self.assertRaisesRegex(ValueError, "Coefficient"):
                    module.PauliOperator({"Z0": coeff})

        operator = module.PauliOperator({"Z0": np.float64(2.0)})
        self.assertEqual(operator.terms, [([("Z", 0)], 2.0)])

        scaled = operator * np.float64(0.5)
        self.assertEqual(scaled.terms, [([("Z", 0)], 1.0)])

        for scalar in (True, np.nan, np.inf):
            with self.subTest(scalar=scalar):
                with self.assertRaisesRegex(ValueError, "scalar"):
                    operator * scalar

    def test_expval_uses_native_single_pauli_helper(self):
        backend = _fake_backend()
        module = _load_legacy_api(backend)
        circuit = _make_circuit(module)

        result = circuit.expval(module.PauliOperator({"Z0": 2.0, "I": 1.0}))

        self.assertEqual(result, 2.5)
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(backend.calls[0][0], "z")

    def test_expval_uses_native_product_helpers(self):
        backend = _fake_backend()
        module = _load_legacy_api(backend)
        circuit = _make_circuit(module)

        result = circuit.expval(
            module.PauliOperator({"Z0 Z2": 3.0, "X0 Y1": 4.0})
        )

        self.assertEqual(result, -2.5)
        self.assertEqual([call[0] for call in backend.calls], ["zz", "pauli"])
        self.assertEqual(backend.calls[0][1][-1], [0, 2])
        self.assertEqual(backend.calls[1][1][-2:], ("XY", [0, 1]))

    def test_get_expval_shares_native_expectation_path(self):
        backend = _fake_backend()
        module = _load_legacy_api(backend)
        circuit = _make_circuit(module)

        program = module.QuantumProgram.__new__(module.QuantumProgram)
        program.circuit_ref = circuit

        result = module.get_expval(program, module.PauliOperator("X0"))

        self.assertEqual(result, 0.5)
        self.assertEqual([call[0] for call in backend.calls], ["x"])


class TestLegacyBuildExecutionContract(unittest.TestCase):
    def test_build_records_python_replay_mode_and_warns(self):
        backend = _fake_backend()
        module = _load_legacy_api(backend)
        simulator = _make_simulator(module)

        @module.kernel
        def bell(circuit):
            circuit.h(0)
            circuit.cx(0, 1)

        with self.assertWarnsRegex(module.LegacyCompilerReplayWarning, "Python circuit replay") as warning:
            program = module.build(bell, 2, simulator)

        self.assertEqual(program.execution_mode, "python-replay")
        self.assertFalse(program.compiler_execution_supported)
        self.assertEqual(program.execution_notes, [str(warning.warning)])
        self.assertIn("execution_mode='python-replay'", repr(program))
        self.assertIsInstance(program.circuit_ref, module.Circuit)
        self.assertIn(("allocate_state", ("handle", 2, 1)), backend.calls)
        self.assertIn(("initialize_state", ("handle", "device-state", 2)), backend.calls)
        self.assertEqual([op.name for op in program.circuit_ref._gate_queue], ["H", "CNOT"])

    def test_build_without_simulator_is_conceptual_mlir_only(self):
        backend = _fake_backend()
        module = _load_legacy_api(backend)

        @module.kernel
        def single(circuit):
            circuit.h(0)

        program = module.build(single, 1, None)

        self.assertEqual(program.execution_mode, "conceptual-mlir")
        self.assertIsNone(program.circuit_ref)
        self.assertFalse(program.compiler_execution_supported)
        self.assertEqual(len(program.execution_notes), 1)
        self.assertIn("does not call MLIRCompiler.compile_and_execute()", program.execution_notes[0])


if __name__ == "__main__":
    unittest.main()
