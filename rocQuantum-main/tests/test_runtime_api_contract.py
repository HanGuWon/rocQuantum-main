"""Contract tests for the canonical rocq runtime surface."""

from __future__ import annotations

import os
import sys
import unittest
import importlib
from unittest import mock

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import rocq
from rocq.kernel import GateOp, kernel
from rocq.operator import HermitianOperator, PauliOperator, SparseHamiltonianOperator, get_expectation_value

rocq_kernel_module = importlib.import_module("rocq.kernel")


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
    def _make_mock_statevector_backend(self, num_qubits=1):
        from rocq.backends import MockBackendWarning, StateVectorBackend

        with mock.patch("rocq.backends.hip_backend", None):
            with mock.patch.dict(os.environ, {"ROCQ_ENABLE_MOCK_BACKENDS": "1"}):
                with self.assertWarnsRegex(MockBackendWarning, "state_vector is using the Python mock fallback"):
                    return StateVectorBackend(num_qubits)

    def _make_mock_density_backend(self, num_qubits=1):
        from rocq.backends import DensityMatrixBackend, MockBackendWarning

        with mock.patch("rocq.backends.dm_backend", None):
            with mock.patch.dict(os.environ, {"ROCQ_ENABLE_MOCK_BACKENDS": "1"}):
                with self.assertWarnsRegex(MockBackendWarning, "density_matrix is using the Python mock fallback"):
                    return DensityMatrixBackend(num_qubits)

    def test_observe_and_sample_exports_exist(self):
        self.assertTrue(callable(rocq.observe))
        self.assertTrue(callable(rocq.sample))
        self.assertTrue(callable(rocq.compile_and_execute))

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

    def test_sample_requires_positive_integer_shots_before_backend_dispatch(self):
        @kernel
        def prep_state():
            q = rocq.qvec(1)
            rocq.h(q[0])

        for invalid_shots in (0, -1, 2.5, True, "4"):
            with self.subTest(shots=invalid_shots):
                with mock.patch("rocq.kernel.get_backend") as patched_get_backend:
                    with self.assertRaisesRegex(ValueError, "shots must be"):
                        rocq.sample(prep_state, invalid_shots, backend="state_vector")
                patched_get_backend.assert_not_called()

    def test_sample_validates_qubits_before_backend_dispatch(self):
        @kernel
        def prep_state():
            q = rocq.qvec(2)
            rocq.h(q[0])

        invalid_qubits = ([2], [-1], [0, 0], [True], [1.5], [], "01")
        for qubits in invalid_qubits:
            with self.subTest(qubits=qubits):
                with mock.patch("rocq.kernel.get_backend") as patched_get_backend:
                    with self.assertRaises((TypeError, ValueError)):
                        rocq.sample(prep_state, 8, backend="state_vector", qubits=qubits)
                patched_get_backend.assert_not_called()

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

    def test_stabilizer_backend_executes_clifford_bell_state(self):
        @kernel
        def bell():
            q = rocq.qvec(2)
            rocq.h(q[0])
            rocq.cnot(q[0], q[1])

        state = rocq.execute(bell, backend="stabilizer")

        np.testing.assert_allclose(
            state,
            np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)], dtype=np.complex128),
            atol=1e-12,
        )
        self.assertEqual(rocq.observe(bell, PauliOperator("Z0 Z1"), backend="stabilizer"), 1.0)
        self.assertEqual(rocq.observe(bell, PauliOperator("X0 X1"), backend="stabilizer"), 1.0)
        self.assertEqual(rocq.observe(bell, PauliOperator("Z0"), backend="stabilizer"), 0.0)

    def test_stabilizer_backend_samples_deterministic_clifford_state(self):
        @kernel
        def flipped():
            q = rocq.qvec(1)
            rocq.x(q[0])

        self.assertEqual(rocq.sample(flipped, 8, backend="stabilizer"), {"1": 8})

    def test_stabilizer_backend_tracks_phase_gate_pauli_propagation(self):
        @kernel
        def phase_state():
            q = rocq.qvec(1)
            rocq.h(q[0])
            rocq.s(q[0])

        self.assertEqual(rocq.observe(phase_state, PauliOperator("Y0"), backend="stabilizer"), 1.0)
        self.assertEqual(rocq.observe(phase_state, PauliOperator("X0"), backend="stabilizer"), 0.0)

    def test_stabilizer_backend_evaluates_pauli_sums(self):
        @kernel
        def bell():
            q = rocq.qvec(2)
            rocq.h(q[0])
            rocq.cnot(q[0], q[1])

        operator = 0.5 + 0.25 * PauliOperator("Z0 Z1") + 0.75 * PauliOperator("X0 X1")

        self.assertEqual(rocq.observe(bell, operator, backend="stabilizer"), 1.5)

    def test_stabilizer_backend_samples_marginal_probabilities(self):
        @kernel
        def bell():
            q = rocq.qvec(2)
            rocq.h(q[0])
            rocq.cnot(q[0], q[1])

        calls = []

        def fake_choice(num_outcomes, size, p):
            calls.append((num_outcomes, size, np.asarray(p)))
            return np.array([0, 1, 0, 1], dtype=np.uint64)

        with mock.patch("rocq.backends.np.random.choice", side_effect=fake_choice):
            counts = rocq.sample(bell, 4, backend="stabilizer", qubits=[0])

        self.assertEqual(counts, {"0": 2, "1": 2})
        self.assertEqual(calls[0][0], 2)
        self.assertEqual(calls[0][1], 4)
        np.testing.assert_allclose(calls[0][2], np.array([0.5, 0.5]))

    def test_stabilizer_backend_tracks_swap_tableau(self):
        @kernel
        def swapped_plus_state():
            q = rocq.qvec(2)
            rocq.h(q[0])
            rocq.swap(q[0], q[1])

        self.assertEqual(rocq.observe(swapped_plus_state, PauliOperator("X1"), backend="stabilizer"), 1.0)
        self.assertEqual(rocq.observe(swapped_plus_state, PauliOperator("X0"), backend="stabilizer"), 0.0)

    def test_stabilizer_backend_rejects_non_clifford_gates(self):
        @kernel
        def non_clifford():
            q = rocq.qvec(1)
            rocq.t(q[0])

        with self.assertRaisesRegex(NotImplementedError, "Clifford-only"):
            rocq.execute(non_clifford, backend="stabilizer")

    def test_stabilizer_backend_aliases_are_available(self):
        @kernel
        def prep_state():
            q = rocq.qvec(1)
            rocq.h(q[0])

        self.assertEqual(rocq.observe(prep_state, PauliOperator("X0"), backend="tableau"), 1.0)
        self.assertEqual(rocq.observe(prep_state, PauliOperator("X0"), backend="clifford"), 1.0)

    def test_density_matrix_backend_decomposes_toffoli(self):
        from rocq.backends import DensityMatrixBackend

        class RecordingDensityState:
            def __init__(self):
                self.calls = []

            def apply_gate_matrix(self, matrix, target, adjoint=False):
                self.calls.append(("gate", np.asarray(matrix), int(target)))

            def apply_cnot(self, control, target):
                self.calls.append(("cnot", int(control), int(target)))

        state = RecordingDensityState()
        backend = DensityMatrixBackend.__new__(DensityMatrixBackend)
        backend.num_qubits = 3
        backend._state = state

        backend._apply_op(GateOp("ccx", [0, 1, 2], {}))

        def gate_label(matrix):
            for name in ("h", "t", "tdg"):
                if np.allclose(matrix, backend._gate_matrix(name)):
                    return name
            return "unknown"

        labeled_calls = [
            (call[0], gate_label(call[1]), call[2]) if call[0] == "gate" else call
            for call in state.calls
        ]
        self.assertEqual(
            labeled_calls,
            [
                ("gate", "h", 2),
                ("cnot", 1, 2),
                ("gate", "tdg", 2),
                ("cnot", 0, 2),
                ("gate", "t", 2),
                ("cnot", 1, 2),
                ("gate", "tdg", 2),
                ("cnot", 0, 2),
                ("gate", "t", 1),
                ("gate", "t", 2),
                ("gate", "h", 2),
                ("cnot", 0, 1),
                ("gate", "t", 0),
                ("gate", "tdg", 1),
                ("cnot", 0, 1),
            ],
        )

    def test_density_matrix_backend_decomposes_cswap_and_bounds_mcx(self):
        from rocq.backends import DensityMatrixBackend

        backend = DensityMatrixBackend.__new__(DensityMatrixBackend)
        backend.num_qubits = 4
        backend._state = mock.Mock()

        decomposed = []
        backend._apply_ccx_decomposition = lambda a, b, target: decomposed.append((a, b, target))
        backend._apply_op(GateOp("cswap", [0, 1, 2], {}))

        self.assertEqual(decomposed, [(0, 2, 1), (0, 1, 2), (0, 2, 1)])

        backend._apply_op(GateOp("mcx", [0, 1], {}))
        backend._state.apply_cnot.assert_called_once_with(0, 1)
        with self.assertRaisesRegex(NotImplementedError, "at most two controls"):
            backend._apply_op(GateOp("mcx", [0, 1, 2, 3], {}))

    def test_qir_missing_binding_error_is_actionable(self):
        @kernel
        def prep_state():
            q = rocq.qvec(1)
            rocq.h(q[0])

        with mock.patch.object(rocq_kernel_module, "rocquantum_bind", None):
            with self.assertRaisesRegex(RuntimeError, "ROCQUANTUM_BUILD_BINDINGS=ON"):
                prep_state.qir()

    def test_qir_error_string_is_not_returned_as_qir(self):
        @kernel
        def prep_state():
            q = rocq.qvec(1)
            rocq.h(q[0])

        class _FakeCompiler:
            def __init__(self, num_qubits, backend):
                self.num_qubits = num_qubits
                self.backend = backend

            def emit_qir(self, mlir):
                return "Error: lowering failed"

        fake_binding = mock.Mock()
        fake_binding.MLIRCompiler = _FakeCompiler

        with mock.patch.object(rocq_kernel_module, "rocquantum_bind", fake_binding):
            with self.assertRaisesRegex(RuntimeError, "Supported canonical MLIR gates"):
                prep_state.qir()

    def test_qir_runtime_failure_is_augmented(self):
        @kernel
        def prep_state():
            q = rocq.qvec(1)
            rocq.h(q[0])

        class _FakeCompiler:
            def __init__(self, num_qubits, backend):
                pass

            def emit_qir(self, mlir):
                raise RuntimeError("MLIR compiler support is disabled")

        fake_binding = mock.Mock()
        fake_binding.MLIRCompiler = _FakeCompiler

        with mock.patch.object(rocq_kernel_module, "rocquantum_bind", fake_binding):
            with self.assertRaisesRegex(RuntimeError, "Supported canonical MLIR gates"):
                prep_state.qir()

    def test_compile_and_execute_uses_native_compiler_binding(self):
        @kernel
        def bell():
            q = rocq.qvec(2)
            rocq.h(q[0])
            rocq.cnot(q[0], q[1])

        calls = []

        class _FakeCompiler:
            def __init__(self, num_qubits, backend):
                calls.append(("init", num_qubits, backend))

            def compile_and_execute(self, mlir, options):
                calls.append(("compile_and_execute", mlir, dict(options)))
                return [1.0, 0.0, 0.0, 0.0]

        fake_binding = mock.Mock()
        fake_binding.MLIRCompiler = _FakeCompiler

        with mock.patch.object(rocq_kernel_module, "rocquantum_bind", fake_binding):
            result = rocq.compile_and_execute(bell, strict=False)

        self.assertEqual(result, [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(calls[0], ("init", 2, "hip_statevec"))
        self.assertEqual(calls[1][2], {"strict": False})
        self.assertIn('"quantum.cnot"', calls[1][1])

    def test_compile_and_execute_augments_compiler_failures(self):
        @kernel
        def prep_state():
            q = rocq.qvec(1)
            rocq.h(q[0])

        class _FakeCompiler:
            def __init__(self, num_qubits, backend):
                pass

            def compile_and_execute(self, mlir, options):
                raise RuntimeError("backend refused MLIR")

        fake_binding = mock.Mock()
        fake_binding.MLIRCompiler = _FakeCompiler

        with mock.patch.object(rocq_kernel_module, "rocquantum_bind", fake_binding):
            with self.assertRaisesRegex(RuntimeError, "Supported canonical MLIR gates"):
                rocq.compile_and_execute(prep_state)

    def test_mock_statevector_backend_evaluates_hermitian_operator(self):
        backend = self._make_mock_statevector_backend(1)

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

    def test_mock_statevector_backend_evaluates_sparse_hamiltonian_operator(self):
        backend = self._make_mock_statevector_backend(1)

        operator = SparseHamiltonianOperator(
            data=np.array([1.0, -1.0], dtype=np.complex128),
            indices=np.array([0, 1], dtype=np.int64),
            indptr=np.array([0, 1, 2], dtype=np.int64),
            shape=(2, 2),
        )

        self.assertEqual(backend.expectation(operator), 1.0)

    def test_hip_statevector_backend_prefers_native_sparse_moments(self):
        from rocq.backends import _HipStateVectorState

        calls = []

        class _FakeHipBackend:
            def get_sparse_matrix_moments(self, handle, d_state, num_qubits, data, indices, indptr, rows, cols):
                calls.append((num_qubits, data.dtype, list(indices), list(indptr), rows, cols))
                return 0.25 + 0.0j, 0.5 + 0.0j

        state = _HipStateVectorState.__new__(_HipStateVectorState)
        state._handle = object()
        state._d_state = object()
        state._num_qubits = 1
        operator = SparseHamiltonianOperator(
            data=np.array([1.0, -1.0], dtype=np.complex128),
            indices=np.array([0, 1], dtype=np.int64),
            indptr=np.array([0, 1, 2], dtype=np.int64),
            shape=(2, 2),
            coefficient=2.0,
        )

        with mock.patch("rocq.backends.hip_backend", _FakeHipBackend()):
            result = state.expectation(operator)

        self.assertEqual(result, 0.5)
        self.assertEqual(calls[0], (1, np.dtype("complex64"), [0, 1], [0, 1, 2], 2, 2))

    def test_hip_statevector_backend_combines_duplicate_pauli_terms(self):
        from rocq.backends import _HipStateVectorState

        calls = []

        class _FakeHipBackend:
            def get_expectation_value_z(self, handle, d_state, num_qubits, qubit):
                calls.append((handle, d_state, num_qubits, qubit))
                return 0.25

        state = _HipStateVectorState.__new__(_HipStateVectorState)
        state._handle = object()
        state._d_state = object()
        state._num_qubits = 1
        operator = PauliOperator("Z0") + 2 * PauliOperator("Z0") + 0.5

        with mock.patch("rocq.backends.hip_backend", _FakeHipBackend()):
            result = state.expectation(operator)

        self.assertEqual(result, 1.25)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][2:], (1, 0))

    def test_hip_statevector_backend_reuses_duplicate_hermitian_sum_terms(self):
        from rocq.backends import _HipStateVectorState

        calls = []

        class _FakeHipBackend:
            def get_expectation_matrix(self, handle, d_state, num_qubits, targets, matrix):
                calls.append((num_qubits, list(targets), matrix.copy()))
                return 0.25 + 0.0j

        state = _HipStateVectorState.__new__(_HipStateVectorState)
        state._handle = object()
        state._d_state = object()
        state._num_qubits = 1
        matrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
        operator = 2 * HermitianOperator(matrix, targets=[0]) + 3 * HermitianOperator(matrix.copy(), targets=[0]) + 0.5

        with mock.patch("rocq.backends.hip_backend", _FakeHipBackend()):
            result = state.expectation(operator)

        self.assertEqual(result, 1.75)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], 1)
        self.assertEqual(calls[0][1], [0])
        np.testing.assert_allclose(calls[0][2], matrix.astype(np.complex64))

    def test_hip_statevector_backend_combines_pauli_terms_inside_mixed_sum(self):
        from rocq.backends import _HipStateVectorState

        pauli_calls = []
        matrix_calls = []

        class _FakeHipBackend:
            def get_expectation_value_z(self, handle, d_state, num_qubits, qubit):
                pauli_calls.append((num_qubits, qubit))
                return 0.25

            def get_expectation_matrix(self, handle, d_state, num_qubits, targets, matrix):
                matrix_calls.append((num_qubits, list(targets), matrix.copy()))
                return 0.5 + 0.0j

        state = _HipStateVectorState.__new__(_HipStateVectorState)
        state._handle = object()
        state._d_state = object()
        state._num_qubits = 1
        matrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
        operator = PauliOperator("Z0") + 2 * PauliOperator("Z0") + HermitianOperator(matrix, targets=[0])

        with mock.patch("rocq.backends.hip_backend", _FakeHipBackend()):
            result = state.expectation(operator)

        self.assertEqual(result, 1.25)
        self.assertEqual(pauli_calls, [(1, 0)])
        self.assertEqual(len(matrix_calls), 1)
        self.assertEqual(matrix_calls[0][1], [0])

    def test_hip_statevector_backend_skips_zero_hermitian_sum_terms(self):
        from rocq.backends import _HipStateVectorState

        class _FakeHipBackend:
            def get_expectation_matrix(self, handle, d_state, num_qubits, targets, matrix):
                raise AssertionError("zero-coefficient Hermitian terms should not be read out")

        state = _HipStateVectorState.__new__(_HipStateVectorState)
        state._handle = object()
        state._d_state = object()
        state._num_qubits = 1
        matrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
        operator = 0 * HermitianOperator(matrix, targets=[0]) + 0.5

        with mock.patch("rocq.backends.hip_backend", _FakeHipBackend()):
            result = state.expectation(operator)

        self.assertEqual(result, 0.5)

    def test_hip_statevector_backend_reuses_duplicate_sparse_sum_terms(self):
        from rocq.backends import _HipStateVectorState

        calls = []

        class _FakeHipBackend:
            def get_sparse_matrix_moments(self, handle, d_state, num_qubits, data, indices, indptr, rows, cols):
                calls.append((num_qubits, list(indices), list(indptr), rows, cols))
                return 0.5 + 0.0j, 0.25 + 0.0j

        state = _HipStateVectorState.__new__(_HipStateVectorState)
        state._handle = object()
        state._d_state = object()
        state._num_qubits = 1
        data = np.array([1.0, -1.0], dtype=np.complex128)
        indices = np.array([0, 1], dtype=np.int64)
        indptr = np.array([0, 1, 2], dtype=np.int64)
        operator = (
            2 * SparseHamiltonianOperator(data, indices, indptr, shape=(2, 2))
            + 3 * SparseHamiltonianOperator(data.copy(), indices.copy(), indptr.copy(), shape=(2, 2))
            + 0.25
        )

        with mock.patch("rocq.backends.hip_backend", _FakeHipBackend()):
            result = state.expectation(operator)

        self.assertEqual(result, 2.75)
        self.assertEqual(calls, [(1, [0, 1], [0, 1, 2], 2, 2)])

    def test_mock_statevector_sum_keeps_matrix_fallback(self):
        backend = self._make_mock_statevector_backend(1)

        operator = HermitianOperator(np.diag([1.0, -1.0]), targets=[0]) + 0.5

        self.assertEqual(backend.expectation(operator), 1.5)

    def test_mock_statevector_observable_division_scales_matrix_fallback(self):
        backend = self._make_mock_statevector_backend(1)

        operator = HermitianOperator(np.diag([1.0, -1.0]), targets=[0]) / 2

        self.assertEqual(backend.expectation(operator), 0.5)

    def test_mock_density_backend_evaluates_hermitian_sum(self):
        backend = self._make_mock_density_backend(1)

        operator = HermitianOperator(np.diag([1.0, -1.0]), targets=[0]) + 0.5

        self.assertEqual(backend.expectation(operator), 1.5)

    def test_mock_density_backend_evaluates_sparse_hamiltonian_sum(self):
        backend = self._make_mock_density_backend(1)

        operator = SparseHamiltonianOperator(
            data=np.array([1.0, -1.0], dtype=np.complex128),
            indices=np.array([0, 1], dtype=np.int64),
            indptr=np.array([0, 1, 2], dtype=np.int64),
            shape=(2, 2),
        ) + 0.5

        self.assertEqual(backend.expectation(operator), 1.5)

    def test_mock_density_backend_evaluates_offdiagonal_hermitian(self):
        backend = self._make_mock_density_backend(1)

        backend._state._density = np.array(  # noqa: SLF001 - contract test for fallback math
            [[0.5, 0.5], [0.5, 0.5]],
            dtype=np.complex64,
        )
        operator = HermitianOperator(np.array([[0.0, 1.0], [1.0, 0.0]]), targets=[0])

        self.assertEqual(backend.expectation(operator), 1.0)

    def test_native_density_backend_prefers_dense_expectation_hook(self):
        from rocq.backends import DensityMatrixBackend

        class _FakeDensityState:
            instances = []

            def __init__(self, num_qubits):
                self.num_qubits = int(num_qubits)
                self.calls = []
                _FakeDensityState.instances.append(self)

            def compute_expectation_matrix(self, matrix, targets):
                self.calls.append((np.asarray(matrix).dtype, np.asarray(matrix).shape, list(targets)))
                return 0.25 + 0.0j

            def get_density_matrix(self):
                raise AssertionError("native dense expectation should avoid full density readback")

        class _FakeDensityModule:
            DensityMatrixState = _FakeDensityState

        with mock.patch("rocq.backends.dm_backend", _FakeDensityModule):
            backend = DensityMatrixBackend(1)
            operator = HermitianOperator(
                np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
                coefficient=2.0,
                targets=[0],
            )

            self.assertEqual(backend.expectation(operator), 0.5)

        self.assertEqual(_FakeDensityState.instances[-1].calls, [(np.dtype(np.complex64), (2, 2), [0])])

    def test_mock_density_backend_applies_multi_qubit_kraus_noise_model(self):
        from rocq.kernel import GateOp

        backend = self._make_mock_density_backend(2)

        kraus = np.zeros((1, 4, 4), dtype=np.complex64)
        kraus[0, 3, 0] = 1.0
        noise = rocq.NoiseModel()
        noise.add_channel("kraus", 0.25, on_qubits=[0, 1], kraus_matrices=kraus)

        backend.run_ops([GateOp("z", [0], {})], noise_model=noise)

        np.testing.assert_allclose(
            np.diag(backend.get_state()).real,
            np.array([0.75, 0.0, 0.0, 0.25], dtype=np.float32),
            atol=1e-7,
        )

    def test_framework_runtime_exposes_native_adjoint_jacobian_hook(self):
        from rocquantum.framework_runtime import RocQuantumRuntime

        class _FakeSimulator:
            def __init__(self):
                self.calls = []

            def adjoint_jacobian(self, operations, observables, trainable_params):
                self.calls.append((operations, observables, trainable_params))
                return np.asarray([[-0.25]], dtype=float)

        sim = _FakeSimulator()
        runtime = RocQuantumRuntime(sim)

        operations = [{"name": "RY", "wires": [0], "params": [0.5], "param_indices": [0]}]
        observables = [[{"coefficient": (1.0, 0.0), "pauli_string": "Z", "targets": [0]}]]

        self.assertTrue(runtime.supports_adjoint_jacobian())
        np.testing.assert_allclose(runtime.adjoint_jacobian(operations, observables, [0]), [[-0.25]])
        self.assertEqual(sim.calls, [(operations, observables, [0])])


if __name__ == "__main__":
    unittest.main()
