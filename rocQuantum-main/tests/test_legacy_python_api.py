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
    backend.MLIRCompiler = type("MLIRCompiler", (), {})

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

        circuit = module.Circuit(2, simulator, batch_size=3)

        self.assertEqual(circuit.batch_size, 3)
        self.assertEqual(circuit._d_state_buffer, "device-state")
        self.assertIn(("allocate_state", ("handle", 2, 3)), backend.calls)
        self.assertIn(("initialize_state", ("handle", "device-state", 2)), backend.calls)

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


if __name__ == "__main__":
    unittest.main()
