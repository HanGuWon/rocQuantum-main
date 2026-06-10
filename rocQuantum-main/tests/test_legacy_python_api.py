"""Contract tests for the legacy ``python/rocq`` API surface."""

from __future__ import annotations

import importlib.util
import os
import sys
import types
import unittest


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LEGACY_ROOT = os.path.join(_PROJECT_ROOT, "python", "rocq")
_API_PATH = os.path.join(_LEGACY_ROOT, "api.py")


def _fake_backend():
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


def _make_circuit(module):
    circuit = module.Circuit.__new__(module.Circuit)
    circuit.num_qubits = 3
    circuit._sim_handle = "handle"
    circuit._d_state_buffer = "device-state"
    circuit.is_multi_gpu = False
    circuit.batch_size = 1
    circuit._gate_queue = []
    circuit._is_dirty = False
    return circuit


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
