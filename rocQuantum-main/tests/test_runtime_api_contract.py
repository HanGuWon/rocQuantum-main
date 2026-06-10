"""Contract tests for the canonical rocq runtime surface."""

from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import rocq
from rocq.kernel import kernel
from rocq.operator import HermitianOperator, PauliOperator, get_expectation_value


class _FakeBackend:
    def __init__(self):
        self.ops = None
        self.noise_model = None
        self.sample_args = None
        self.operator = None

    def run_ops(self, ops, noise_model=None):
        self.ops = list(ops)
        self.noise_model = noise_model

    def get_state(self):
        return "fake-state"

    def sample(self, shots, qubits=None):
        self.sample_args = (shots, list(qubits) if qubits is not None else None)
        return {"0": shots}

    def expectation(self, operator):
        self.operator = operator
        return 1.25


class TestCanonicalRuntimeSurface(unittest.TestCase):
    def test_observe_and_sample_exports_exist(self):
        self.assertTrue(callable(rocq.observe))
        self.assertTrue(callable(rocq.sample))

    def test_get_expectation_value_delegates_to_observe(self):
        @kernel
        def prep_state():
            q = rocq.qvec(1)
            rocq.h(q[0])

        operator = PauliOperator("Z0")

        with mock.patch("rocq.kernel.observe", return_value=1.25) as patched_observe:
            result = get_expectation_value(prep_state, operator, backend="state_vector")

        self.assertEqual(result, 1.25)
        patched_observe.assert_called_once_with(prep_state, operator, backend="state_vector")

    def test_execute_uses_program_level_backend_contract(self):
        @kernel
        def bell():
            q = rocq.qvec(2)
            rocq.h(q[0])
            rocq.cnot(q[0], q[1])

        fake_backend = _FakeBackend()
        with mock.patch("rocq.kernel.get_backend", return_value=fake_backend):
            result = rocq.execute(bell, backend="state_vector")

        self.assertEqual(result, "fake-state")
        self.assertEqual([op.name.lower() for op in fake_backend.ops], ["h", "cnot"])
        self.assertIsNone(fake_backend.noise_model)

    def test_sample_uses_backend_sample(self):
        @kernel
        def bell():
            q = rocq.qvec(2)
            rocq.h(q[0])
            rocq.cnot(q[0], q[1])

        fake_backend = _FakeBackend()
        with mock.patch("rocq.kernel.get_backend", return_value=fake_backend):
            result = rocq.sample(bell, 32, backend="state_vector", qubits=[0])

        self.assertEqual(result, {"0": 32})
        self.assertEqual(fake_backend.sample_args, (32, [0]))
        self.assertEqual([op.name.lower() for op in fake_backend.ops], ["h", "cnot"])

    def test_observe_uses_backend_expectation(self):
        @kernel
        def prep_state():
            q = rocq.qvec(1)
            rocq.h(q[0])

        operator = PauliOperator("Z0")
        fake_backend = _FakeBackend()
        with mock.patch("rocq.kernel.get_backend", return_value=fake_backend):
            result = rocq.observe(prep_state, operator, backend="state_vector")

        self.assertEqual(result, 1.25)
        self.assertIs(fake_backend.operator, operator)
        self.assertEqual([op.name.lower() for op in fake_backend.ops], ["h"])

    def test_mock_statevector_backend_evaluates_hermitian_operator(self):
        from rocq.backends import StateVectorBackend

        with mock.patch("rocq.backends.hip_backend", None):
            with mock.patch.dict(os.environ, {"ROCQ_ENABLE_MOCK_BACKENDS": "1"}):
                backend = StateVectorBackend(1)

        operator = HermitianOperator(np.diag([1.0, -1.0]), targets=[0])

        self.assertEqual(backend.expectation(operator), 1.0)

    def test_hip_statevector_backend_prefers_native_hermitian_expectation(self):
        from rocq.backends import _HipStateVectorState

        calls = []

        class _FakeHipBackend:
            def get_expectation_matrix(self, handle, d_state, num_qubits, targets, matrix):
                calls.append((handle, d_state, num_qubits, list(targets), matrix.dtype))
                return 0.5 + 0.0j

        state = _HipStateVectorState.__new__(_HipStateVectorState)
        state._handle = object()
        state._d_state = object()
        state._num_qubits = 1
        operator = HermitianOperator(np.diag([1.0, -1.0]), coefficient=2.0, targets=[0])

        with mock.patch("rocq.backends.hip_backend", _FakeHipBackend()):
            result = state.expectation(operator)

        self.assertEqual(result, 1.0)
        self.assertEqual(calls[0][2], 1)
        self.assertEqual(calls[0][3], [0])
        self.assertEqual(calls[0][4], np.dtype("complex64"))


if __name__ == "__main__":
    unittest.main()
