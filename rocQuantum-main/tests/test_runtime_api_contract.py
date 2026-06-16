"""Contract tests for the canonical rocq runtime surface."""

from __future__ import annotations

import os
import sys
import unittest
import importlib
from concurrent.futures import Future, ThreadPoolExecutor
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
        self.assertTrue(callable(rocq.density_matrix_capabilities))
        self.assertTrue(callable(rocq.distributed_capabilities))
        self.assertTrue(callable(rocq.runtime_capabilities))
        self.assertTrue(callable(rocq.observe))
        self.assertTrue(callable(rocq.sample))
        self.assertTrue(callable(rocq.compile_and_execute))
        self.assertTrue(callable(rocq.execute_async))
        self.assertTrue(callable(rocq.get_state))
        self.assertTrue(callable(rocq.get_state_async))
        self.assertTrue(callable(rocq.sample_async))
        self.assertTrue(callable(rocq.observe_async))
        self.assertTrue(callable(rocq.compile_and_execute_async))
        for name in (
            "execute_async",
            "get_state",
            "get_state_async",
            "sample_async",
            "observe_async",
            "compile_and_execute_async",
        ):
            self.assertIn(name, rocq.__all__)
        self.assertIn("density_matrix_capabilities", rocq.__all__)
        self.assertIn("distributed_capabilities", rocq.__all__)
        self.assertIn("runtime_capabilities", rocq.__all__)

    def test_runtime_capabilities_expose_canonical_runtime_contract(self):
        capabilities = rocq.runtime_capabilities()

        self.assertEqual(capabilities["status"], "partial")
        self.assertEqual(capabilities["primary_python_surface"], "rocq")
        self.assertIn("python/rocq", capabilities["legacy_python_surface"])
        self.assertIn("execute", capabilities["execution_entry_points"])
        self.assertIn("compile_and_execute", capabilities["execution_entry_points"])
        self.assertIn("observe_async", capabilities["execution_entry_points"])
        self.assertIn("compile_and_execute_async", capabilities["execution_entry_points"])
        self.assertIn("state_vector", capabilities["supported_backends"])
        self.assertIn(
            "host-side Future wrappers for execute/get_state/sample/observe/compile_and_execute",
            capabilities["supported_features"],
        )
        self.assertIn(
            "bool-safe state-vector-only enable_fusion execution option",
            capabilities["supported_features"],
        )
        self.assertIn(
            "canonical backend-name validation",
            capabilities["supported_features"],
        )
        self.assertIn(
            "positive-integer direct backend size validation",
            capabilities["supported_features"],
        )
        self.assertIn(
            "direct backend gate-target validation",
            capabilities["supported_features"],
        )
        self.assertIn(
            "finite direct backend gate-angle validation",
            capabilities["supported_features"],
        )
        self.assertIn(
            "GateFusion rotation-angle validation before native queue dispatch",
            capabilities["supported_features"],
        )
        self.assertIn(
            "direct backend state readback validation before returning native results",
            capabilities["supported_features"],
        )
        self.assertIn(
            "Pauli observable target validation before backend dispatch",
            capabilities["supported_features"],
        )
        self.assertIn(
            "lazy statevector fallback for legacy Pauli expectation bindings",
            capabilities["supported_features"],
        )
        self.assertIn(
            "dense Hermitian observable validation before native/backend dispatch",
            capabilities["supported_features"],
        )
        self.assertIn(
            "dense matrix operation validation before native device upload",
            capabilities["supported_features"],
        )
        self.assertIn(
            "sparse Hamiltonian observable CSR validation before native/backend dispatch",
            capabilities["supported_features"],
        )
        self.assertIn(
            "density-matrix Kraus channel payload validation before native device upload",
            capabilities["supported_features"],
        )
        self.assertIn(
            "density-matrix noise-model channel revalidation before backend dispatch",
            capabilities["supported_features"],
        )
        self.assertIn(
            "partial compiler execution entry point with compiler_capabilities() boundary metadata",
            capabilities["supported_features"],
        )
        self.assertIn(
            "native HIP-stream futures",
            capabilities["unsupported_features"],
        )
        self.assertEqual(
            capabilities["async_execution"]["future_type"],
            "concurrent.futures.Future",
        )
        self.assertEqual(
            capabilities["async_execution"]["submission"],
            "host_threadpool",
        )
        self.assertTrue(capabilities["async_execution"]["preserves_backend_validation"])
        self.assertFalse(capabilities["async_execution"]["native_hip_stream_future"])
        self.assertFalse(capabilities["async_execution"]["multi_qpu_scheduler"])
        self.assertFalse(capabilities["async_execution"]["distributed_scheduler"])
        self.assertFalse(capabilities["async_execution"]["device_overlap_proof"])
        self.assertIn("enable_fusion", capabilities["runtime_options"])
        self.assertIn("ROCQ_DISABLE_GATE_FUSION", capabilities["environment_switches"])
        self.assertIn("compatibility surface", capabilities["legacy_note"])
        self.assertIn("self-hosted ROCm CI", capabilities["performance_note"])

    def test_distributed_capabilities_expose_partial_runtime_contract(self):
        capabilities = rocq.distributed_capabilities()

        self.assertEqual(capabilities["status"], "partial")
        self.assertIn("native_binding_available", capabilities)
        self.assertIn("native_backend_query_available", capabilities)
        self.assertIn("ROCQ_REQUIRE_RCCL", capabilities["runtime_switches"])
        self.assertIn(
            "local-domain selected-qubit sampling/probabilities",
            capabilities["supported_features"],
        )
        self.assertIn(
            "swap-localized selected-qubit sampling/probabilities for covered single-node layouts",
            capabilities["supported_features"],
        )
        self.assertIn(
            "multi-node distributed allocation",
            capabilities["unsupported_features"],
        )
        self.assertEqual(
            capabilities["execution_scope"]["single_node_multi_gpu"],
            "experimental_partial",
        )
        self.assertEqual(capabilities["execution_scope"]["multi_node"], "unsupported")
        self.assertEqual(
            capabilities["execution_scope"]["host_fallback"],
            "explicit_slow_debug_only",
        )
        self.assertFalse(capabilities["hardware_evidence"]["probe_performed"])
        self.assertTrue(capabilities["hardware_evidence"]["native_rocm_device_required"])
        self.assertTrue(
            capabilities["hardware_evidence"]["multiple_gpus_required_for_runtime_proof"]
        )
        self.assertFalse(capabilities["hardware_evidence"]["capability_query_is_runtime_proof"])
        self.assertIn("MULTI_GPU_GUIDE.md", capabilities["guide"])
        self.assertIn("self-hosted ROCm CI", capabilities["performance_note"])
        self.assertIn("does not perform a hardware probe", capabilities["performance_note"])

    def test_density_matrix_capabilities_expose_cudensitymat_boundary(self):
        capabilities = rocq.density_matrix_capabilities()

        self.assertEqual(capabilities["status"], "partial")
        self.assertIn("native_binding_available", capabilities)
        self.assertIn(
            "generic single- and multi-qubit Kraus channel application up to four targets",
            capabilities["supported_features"],
        )
        self.assertIn(
            "density sampling with device-side measured-qubit marginal probability reduction",
            capabilities["supported_features"],
        )
        self.assertIn(
            "native dense Hermitian expectation for up to four target qubits",
            capabilities["supported_features"],
        )
        self.assertIn(
            "GPU-resident cuDensityMat-style channel descriptors",
            capabilities["unsupported_features"],
        )
        self.assertIn(
            "GPU-resident RNG/CDF sampling",
            capabilities["unsupported_features"],
        )
        self.assertEqual(
            capabilities["execution_scope"]["channel_application"],
            "per_kraus_kernel_correctness_path_up_to_4_targets",
        )
        self.assertEqual(
            capabilities["execution_scope"]["sampling"],
            "device_marginal_probabilities_host_shot_drawing",
        )
        self.assertEqual(
            capabilities["execution_scope"]["dense_observable_targets"],
            "native_up_to_4_target_qubits",
        )
        self.assertEqual(
            capabilities["execution_scope"]["descriptor_planning"],
            "unsupported",
        )
        self.assertFalse(capabilities["hardware_evidence"]["probe_performed"])
        self.assertFalse(capabilities["hardware_evidence"]["capability_query_is_runtime_proof"])
        self.assertEqual(capabilities["limits"]["max_qubits_before_dense_size_overflow"], 30)
        self.assertEqual(capabilities["limits"]["max_kraus_channel_targets"], 4)
        self.assertEqual(capabilities["limits"]["max_dense_observable_targets"], 4)
        self.assertEqual(capabilities["limits"]["max_sampled_qubits"], 20)
        self.assertIn("cuDensityMat-style descriptor planning", capabilities["performance_note"])

    def test_statevector_backend_fuses_same_target_single_qubit_spans(self):
        from rocq.backends import StateVectorBackend

        fusion_calls = []
        replay_calls = []

        class _Status:
            SUCCESS = 0

        class _GateOp:
            pass

        class _FakeHipBackend:
            GateOp = _GateOp
            rocqStatus = _Status

        class _FakeFusion:
            def process_queue(self, queue):
                fusion_calls.append(
                    tuple(
                        (op.name, tuple(op.controls), tuple(op.targets), tuple(op.params))
                        for op in queue
                    )
                )
                return _Status.SUCCESS

        class _FakeState:
            def fusion_engine(self):
                return _FakeFusion()

            def apply_named_gate(self, name, targets, params):
                replay_calls.append((name, list(targets), dict(params)))

        backend = StateVectorBackend.__new__(StateVectorBackend)
        backend.num_qubits = 2
        backend._uses_mock = False
        backend._state = _FakeState()

        ops = [
            GateOp("h", [0], {}),
            GateOp("rz", [0], {"phi": 0.25}),
            GateOp("x", [1], {}),
        ]
        with mock.patch("rocq.backends.hip_backend", _FakeHipBackend):
            backend.run_ops(ops)

        self.assertEqual(
            fusion_calls,
            [(("H", (), (0,), ()), ("RZ", (), (0,), (0.25,)))],
        )
        self.assertEqual(replay_calls, [("x", [1], {})])

    def test_statevector_backend_rejects_invalid_fusion_rotation_angles_before_queue_dispatch(self):
        from rocq.backends import StateVectorBackend

        class _Status:
            SUCCESS = 0

        class _GateOp:
            pass

        class _FakeHipBackend:
            GateOp = _GateOp
            rocqStatus = _Status

        class _FakeFusion:
            def process_queue(self, queue):
                raise AssertionError("GateFusion should not receive invalid rotation angles")

        class _FakeState:
            def fusion_engine(self):
                return _FakeFusion()

            def apply_named_gate(self, name, targets, params):
                raise AssertionError("invalid fusion spans should fail before replay dispatch")

        invalid_params = ({}, {"theta": None}, {"theta": np.nan}, {"theta": True}, {"theta": "0.25"})
        for params in invalid_params:
            with self.subTest(params=params):
                backend = StateVectorBackend.__new__(StateVectorBackend)
                backend.num_qubits = 1
                backend._uses_mock = False
                backend._state = _FakeState()

                ops = [
                    GateOp("rx", [0], dict(params)),
                    GateOp("ry", [0], {"theta": 0.25}),
                ]
                with mock.patch("rocq.backends.hip_backend", _FakeHipBackend):
                    with self.assertRaisesRegex(ValueError, "angle must"):
                        backend.run_ops(ops)

    def test_top_level_phase_gate_exports_record_canonical_ops(self):
        for name in ("tdg", "tdag", "p", "phase", "cp", "cphase"):
            with self.subTest(name=name):
                self.assertTrue(callable(getattr(rocq, name)))
                self.assertIn(name, rocq.__all__)

        @kernel
        def phase_gates():
            q = rocq.qvec(2)
            rocq.tdg(q[0])
            rocq.tdag(q[1])
            rocq.p(0.125, q[0])
            rocq.phase(0.25, q[1])
            rocq.cp(0.5, q[0], q[1])
            rocq.cphase(0.75, q[1], q[0])

        fake_backend = _FakeBackend()
        with mock.patch("rocq.kernel.get_backend", return_value=fake_backend):
            rocq.execute(phase_gates, backend="state_vector")

        recorded = [(op.name.lower(), op.targets, op.params) for op in fake_backend.ops]
        self.assertEqual(
            recorded,
            [
                ("tdg", [0], {}),
                ("tdg", [1], {}),
                ("p", [0], {"phi": 0.125}),
                ("p", [1], {"phi": 0.25}),
                ("cp", [0, 1], {"phi": 0.5}),
                ("cp", [1, 0], {"phi": 0.75}),
            ],
        )

    def test_qvec_requires_positive_integer_size(self):
        for invalid_size in (0, -1, 1.5, True, "2"):
            with self.subTest(size=invalid_size):
                with self.assertRaisesRegex(ValueError, "qvec size must be a positive integer"):
                    rocq.qvec(invalid_size)

        register = rocq.qvec(np.int64(2))
        self.assertEqual(len(register), 2)
        self.assertEqual(register.qubits, [0, 1])

    def test_spin_factories_require_nonnegative_integer_targets(self):
        for factory in (rocq.spin.i, rocq.spin.x, rocq.spin.y, rocq.spin.z):
            for target in (-1, 0.5, True, "0"):
                with self.subTest(factory=factory.__name__, target=target):
                    with self.assertRaisesRegex(ValueError, "target"):
                        factory(target)

        self.assertEqual(rocq.spin.x(np.int64(2)).pauli_string, "X2")
        self.assertEqual(rocq.spin.i().pauli_string, "I")

    def test_observable_targets_and_sparse_shape_require_integer_inputs(self):
        matrix = np.eye(2)
        for targets in (-1, 0.5, True, "0", [-1], [0.5], [True], ["0"], [0, 0]):
            with self.subTest(targets=targets):
                with self.assertRaisesRegex(ValueError, "HermitianOperator target"):
                    HermitianOperator(matrix, targets=targets)

        scalar_operator = HermitianOperator(matrix, targets=np.int64(0))
        self.assertEqual(scalar_operator.targets, [0])

        operator = HermitianOperator(matrix, targets=[np.int64(0)])
        self.assertEqual(operator.targets, [0])

    def test_hermitian_operator_validates_matrix_payload_inputs(self):
        invalid_shapes = (
            1.0,
            [1.0, 0.0],
            [[1.0, 0.0]],
            np.eye(3),
        )
        for matrix in invalid_shapes:
            with self.subTest(matrix=matrix):
                with self.assertRaisesRegex(ValueError, "HermitianOperator matrix"):
                    HermitianOperator(matrix, targets=[0])

        invalid_values = (
            [[1.0, 0.0], [0.0, True]],
            [[1.0, 0.0], [0.0, "1.0"]],
            [[1.0, 0.0], [0.0, np.nan]],
            [[1.0, 0.0], [0.0, np.inf]],
        )
        for matrix in invalid_values:
            with self.subTest(matrix=matrix):
                with self.assertRaisesRegex(ValueError, "HermitianOperator matrix"):
                    HermitianOperator(matrix, targets=[0])

        operator = HermitianOperator([[1, 0], [0, -1j]], targets=np.int64(0))
        np.testing.assert_array_equal(
            operator.matrix,
            np.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, -1.0j]], dtype=np.complex128),
        )
        self.assertEqual(operator.targets, [0])

    def test_sparse_operator_validates_shape_inputs(self):
        data = np.array([1.0 + 0.0j])
        indices = np.array([0])
        indptr = np.array([0, 1, 1])
        for shape in ((0, 2), (-2, 2), (2.5, 2), (True, 2), ("2", 2)):
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(ValueError, "SparseHamiltonianOperator shape dimension"):
                    SparseHamiltonianOperator(data, indices, indptr, shape=shape)
        with self.assertRaisesRegex(ValueError, "two dimensions"):
            SparseHamiltonianOperator(data, indices, indptr, shape=(2,))
        with self.assertRaisesRegex(ValueError, "two dimensions"):
            SparseHamiltonianOperator(data, indices, indptr, shape=2)
        with self.assertRaisesRegex(ValueError, "square"):
            SparseHamiltonianOperator(data, indices, indptr, shape=(2, 4))
        with self.assertRaisesRegex(ValueError, "power of two"):
            SparseHamiltonianOperator(data, indices, indptr, shape=(3, 3))

        sparse = SparseHamiltonianOperator(data, indices, indptr, shape=(np.int64(2), np.int64(2)))
        self.assertEqual(sparse.shape, (2, 2))

    def test_sparse_operator_validates_csr_payload_inputs(self):
        data = np.array([1.0 + 0.0j])
        indices = np.array([0])
        indptr = np.array([0, 1, 1])

        invalid_data = ([np.nan], [np.inf], [True], ["1.0"])
        for value in invalid_data:
            with self.subTest(data=value):
                with self.assertRaisesRegex(ValueError, "CSR data"):
                    SparseHamiltonianOperator(value, indices, indptr, shape=(2, 2))

        invalid_indices = ([0.5], [True], ["0"], [-1])
        for value in invalid_indices:
            with self.subTest(indices=value):
                with self.assertRaisesRegex(ValueError, "CSR indices"):
                    SparseHamiltonianOperator(data, value, indptr, shape=(2, 2))

        invalid_indptr = ([0, 1.5, 1], [0, True, 1], [0, "1", 1], [0, -1, 1])
        for value in invalid_indptr:
            with self.subTest(indptr=value):
                with self.assertRaisesRegex(ValueError, "CSR indptr"):
                    SparseHamiltonianOperator(data, indices, value, shape=(2, 2))

        with self.assertRaisesRegex(ValueError, "data and indices lengths"):
            SparseHamiltonianOperator([1.0, 2.0], indices, indptr, shape=(2, 2))
        with self.assertRaisesRegex(ValueError, "indptr length"):
            SparseHamiltonianOperator(data, indices, [0, 1], shape=(2, 2))
        with self.assertRaisesRegex(ValueError, "start at 0 and end"):
            SparseHamiltonianOperator(data, indices, [1, 1, 1], shape=(2, 2))
        with self.assertRaisesRegex(ValueError, "monotonic"):
            SparseHamiltonianOperator(data, indices, [0, 2, 1], shape=(2, 2))
        with self.assertRaisesRegex(ValueError, "column index"):
            SparseHamiltonianOperator(data, [2], indptr, shape=(2, 2))

        sparse = SparseHamiltonianOperator(data, indices, indptr, shape=(2, 2))
        np.testing.assert_array_equal(sparse.data, np.array([1.0 + 0.0j], dtype=np.complex128))
        np.testing.assert_array_equal(sparse.indices, np.array([0], dtype=np.int64))
        np.testing.assert_array_equal(sparse.indptr, np.array([0, 1, 1], dtype=np.int64))

    def test_observable_coefficients_must_be_finite_numeric_values(self):
        matrix = np.eye(2)
        data = np.array([1.0 + 0.0j])
        indices = np.array([0])
        indptr = np.array([0, 1, 1])

        invalid_values = (np.nan, np.inf, -np.inf, True, "1.0")
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "coefficient|scalar|divisor"):
                    PauliOperator("Z0", coefficient=value)
                with self.assertRaisesRegex(ValueError, "coefficient|scalar|divisor"):
                    HermitianOperator(matrix, coefficient=value, targets=[0])
                with self.assertRaisesRegex(ValueError, "coefficient|scalar|divisor"):
                    SparseHamiltonianOperator(data, indices, indptr, shape=(2, 2), coefficient=value)

        for value in (np.nan, np.inf, -np.inf, True):
            with self.subTest(arithmetic_value=value):
                with self.assertRaisesRegex(ValueError, "coefficient|scalar|divisor"):
                    PauliOperator("Z0") * value
                with self.assertRaisesRegex(ValueError, "coefficient|scalar|divisor"):
                    PauliOperator("Z0") + value

        with self.assertRaisesRegex(ValueError, "divisor must be non-zero"):
            PauliOperator("Z0") / 0

        operator = PauliOperator("Z0", coefficient=1.0 + 0.5j)
        self.assertEqual(operator.coefficient, 1.0 + 0.5j)

    def test_noise_model_validates_probability_targets_and_names(self):
        invalid_probabilities = (-0.1, 1.1, np.nan, np.inf, True, "0.1")
        for probability in invalid_probabilities:
            with self.subTest(probability=probability):
                noise = rocq.NoiseModel()
                with self.assertRaisesRegex(ValueError, "Probability"):
                    noise.add_channel("bit_flip", probability)

        invalid_qubits = (0.5, True, "0", [], [0.5], [True], ["0"], [0, 0], [-1])
        for on_qubits in invalid_qubits:
            with self.subTest(on_qubits=on_qubits):
                noise = rocq.NoiseModel()
                with self.assertRaises((TypeError, ValueError)):
                    noise.add_channel("bit_flip", 0.1, on_qubits=on_qubits)

        for channel_type in ("", None):
            with self.subTest(channel_type=channel_type):
                noise = rocq.NoiseModel()
                with self.assertRaisesRegex(ValueError, "channel_type"):
                    noise.add_channel(channel_type, 0.1)

        for after_op in ("", 7):
            with self.subTest(after_op=after_op):
                noise = rocq.NoiseModel()
                with self.assertRaisesRegex(ValueError, "after_op"):
                    noise.add_channel("bit_flip", 0.1, after_op=after_op)

        noise = rocq.NoiseModel()
        noise.add_channel("bit_flip", np.float64(0.25), on_qubits=np.int64(1), after_op="X")
        self.assertEqual(
            noise.get_channels()[0],
            {
                "type": "bit_flip",
                "prob": 0.25,
                "qubits": [1],
                "op": "x",
                "kraus_matrices": None,
            },
        )

        with mock.patch("builtins.print") as patched_print:
            noise = rocq.NoiseModel()
            noise.add_channel(" depolarizing ", 0.1, after_op=" H ")
        patched_print.assert_not_called()
        self.assertEqual(noise.get_channels()[0]["type"], "depolarizing")
        self.assertEqual(noise.get_channels()[0]["op"], "h")

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

    def test_execute_sample_observe_forward_explicit_fusion_option(self):
        @kernel
        def prep_state():
            q = rocq.qvec(1)
            rocq.h(q[0])

        fake_backend = _FakeBackend()
        with mock.patch("rocq.kernel.get_backend", return_value=fake_backend) as patched_get_backend:
            rocq.execute(prep_state, backend="state_vector", enable_fusion=False)
        patched_get_backend.assert_called_once_with("state_vector", 1, enable_fusion=False)

        fake_backend = _FakeBackend()
        with mock.patch("rocq.kernel.get_backend", return_value=fake_backend) as patched_get_backend:
            rocq.sample(prep_state, 8, backend="state_vector", enable_fusion=True)
        patched_get_backend.assert_called_once_with("state_vector", 1, enable_fusion=True)

        fake_backend = _FakeBackend()
        with mock.patch("rocq.kernel.get_backend", return_value=fake_backend) as patched_get_backend:
            rocq.observe(
                prep_state,
                PauliOperator("Z0"),
                backend="state_vector",
                enable_fusion=False,
            )
        patched_get_backend.assert_called_once_with("state_vector", 1, enable_fusion=False)

    def test_execute_async_forwards_explicit_fusion_option(self):
        @kernel
        def prep_state():
            q = rocq.qvec(1)
            rocq.h(q[0])

        fake_backend = _FakeBackend()
        with ThreadPoolExecutor(max_workers=1) as executor:
            with mock.patch("rocq.kernel.get_backend", return_value=fake_backend) as patched_get_backend:
                future = rocq.execute_async(
                    prep_state,
                    backend="state_vector",
                    enable_fusion=False,
                    executor=executor,
                )
                self.assertEqual(future.result(timeout=5), "fake-state")

        patched_get_backend.assert_called_once_with("state_vector", 1, enable_fusion=False)

    def test_backend_factory_validates_fusion_option_scope(self):
        from rocq.backends import StateVectorBackend, get_backend

        with self.assertRaisesRegex(ValueError, "enable_fusion must be a boolean"):
            get_backend("state_vector", 1, enable_fusion="false")

        with mock.patch("rocq.backends.hip_backend", None):
            with mock.patch.dict(os.environ, {"ROCQ_ENABLE_MOCK_BACKENDS": "1"}):
                with self.assertRaisesRegex(ValueError, "enable_fusion must be a boolean"):
                    StateVectorBackend(1, enable_fusion="false")

        with self.assertRaisesRegex(ValueError, "enable_fusion only applies"):
            get_backend("density_matrix", 1, enable_fusion=False)

    def test_backend_constructors_validate_num_qubits_before_dispatch(self):
        from rocq.backends import DensityMatrixBackend, StateVectorBackend, get_backend

        invalid_num_qubits = (0, -1, True, 1.5, "1")
        for num_qubits in invalid_num_qubits:
            with self.subTest(factory_num_qubits=num_qubits):
                with self.assertRaisesRegex(ValueError, "num_qubits must"):
                    get_backend("stabilizer", num_qubits)
            with self.subTest(statevector_num_qubits=num_qubits):
                with mock.patch("rocq.backends.hip_backend", None):
                    with mock.patch.dict(os.environ, {"ROCQ_ENABLE_MOCK_BACKENDS": "1"}):
                        with self.assertRaisesRegex(ValueError, "num_qubits must"):
                            StateVectorBackend(num_qubits)

        self.assertEqual(get_backend("stabilizer", np.int64(2)).num_qubits, 2)

        with mock.patch("rocq.backends.hip_backend", None):
            with mock.patch.dict(os.environ, {"ROCQ_ENABLE_MOCK_BACKENDS": "1"}):
                with self.assertRaisesRegex(ValueError, "state-vector backend maximum"):
                    StateVectorBackend(61)

        with mock.patch("rocq.backends.dm_backend", None):
            with mock.patch.dict(os.environ, {"ROCQ_ENABLE_MOCK_BACKENDS": "1"}):
                with self.assertRaisesRegex(ValueError, "density-matrix backend maximum"):
                    DensityMatrixBackend(31)

    def test_backend_factory_validates_backend_name_before_dispatch(self):
        from rocq.backends import get_backend

        invalid_backend_names = (None, True, ["state_vector"], b"state_vector")
        for backend_name in invalid_backend_names:
            with self.subTest(backend_name=backend_name):
                with self.assertRaisesRegex(ValueError, "backend_name must be one of"):
                    get_backend(backend_name, 1)

        with self.assertRaisesRegex(ValueError, "Unsupported backend 'gpu'"):
            get_backend("gpu", 1)

    def test_get_state_alias_uses_execute_backend_contract(self):
        @kernel
        def bell():
            q = rocq.qvec(2)
            rocq.h(q[0])
            rocq.cnot(q[0], q[1])

        fake_backend = _FakeBackend()
        with mock.patch("rocq.kernel.get_backend", return_value=fake_backend):
            result = rocq.get_state(bell, backend="state_vector")

        self.assertEqual(result, "fake-state")
        self.assertEqual([op.name.lower() for op in fake_backend.ops], ["h", "cnot"])

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

    def test_direct_backends_revalidate_native_sample_results_before_counts(self):
        from rocq.backends import DensityMatrixBackend, StateVectorBackend, _format_sample_counts

        self.assertEqual(
            _format_sample_counts([np.uint64(0), np.int64(3), 3], 2, 3),
            {"00": 1, "11": 2},
        )

        invalid_payloads = (
            [0, 1],
            [0, 4, 1],
            [0, -1, 1],
            [0, True, 1],
            [0, 1.5, 1],
            [0, "1", 1],
            np.array([[0, 1, 2]], dtype=np.int64),
        )
        for payload in invalid_payloads:
            with self.subTest(helper_payload=repr(payload)):
                with self.assertRaisesRegex(ValueError, "sample result"):
                    _format_sample_counts(payload, 2, 3)

        for backend_cls in (StateVectorBackend, DensityMatrixBackend):
            with self.subTest(backend=backend_cls.__name__, payload="valid"):
                backend = backend_cls.__new__(backend_cls)
                backend.num_qubits = 2
                backend._state = mock.Mock()
                backend._state.sample.return_value = [0, 3, 3]
                self.assertEqual(backend.sample(3, qubits=[0, 1]), {"00": 1, "11": 2})
                backend._state.sample.assert_called_once_with([0, 1], 3)

            for payload in invalid_payloads:
                with self.subTest(backend=backend_cls.__name__, payload=repr(payload)):
                    backend = backend_cls.__new__(backend_cls)
                    backend.num_qubits = 2
                    backend._state = mock.Mock()
                    backend._state.sample.return_value = payload
                    with self.assertRaisesRegex(ValueError, "sample result"):
                        backend.sample(3, qubits=[0, 1])
                    backend._state.sample.assert_called_once_with([0, 1], 3)

    def test_direct_backends_revalidate_native_state_readbacks_before_returning(self):
        from rocq.backends import (
            DensityMatrixBackend,
            StateVectorBackend,
            _normalize_density_matrix_result,
            _normalize_statevector_result,
        )

        state = _normalize_statevector_result(np.array([1, 0], dtype=np.complex64), 1)
        np.testing.assert_allclose(state, np.array([1, 0], dtype=np.complex128))
        self.assertEqual(state.dtype, np.complex128)

        density = _normalize_density_matrix_result(np.eye(2, dtype=np.complex64), 1)
        np.testing.assert_allclose(density, np.eye(2, dtype=np.complex128))
        self.assertEqual(density.dtype, np.complex128)

        invalid_statevectors = (
            [1],
            [1, 0, 0],
            [[1, 0]],
            [1, np.nan],
            [1, np.inf],
            [1, complex(0.0, float("inf"))],
            [1, True],
            ["1", 0],
        )
        for payload in invalid_statevectors:
            with self.subTest(helper_statevector=repr(payload)):
                with self.assertRaisesRegex(ValueError, "Statevector"):
                    _normalize_statevector_result(payload, 1)

        invalid_density_matrices = (
            np.eye(3, dtype=np.complex128),
            [1, 0],
            np.ones((1, 1, 1), dtype=np.complex128),
            [[1, np.nan], [0, 1]],
            [[1, complex(0.0, float("inf"))], [0, 1]],
            [[1, True], [0, 1]],
            [["1", 0], [0, 1]],
        )
        for payload in invalid_density_matrices:
            with self.subTest(helper_density=repr(payload)):
                with self.assertRaisesRegex(ValueError, "Density matrix"):
                    _normalize_density_matrix_result(payload, 1)

        state_backend = StateVectorBackend.__new__(StateVectorBackend)
        state_backend.num_qubits = 1
        state_backend._state = mock.Mock()
        state_backend._state.get_state_vector.return_value = np.array([1, 0], dtype=np.complex64)
        np.testing.assert_allclose(state_backend.get_state(), np.array([1, 0], dtype=np.complex128))
        state_backend._state.get_state_vector.assert_called_once_with()

        state_backend = StateVectorBackend.__new__(StateVectorBackend)
        state_backend.num_qubits = 1
        state_backend._state = mock.Mock()
        state_backend._state.get_state_vector.return_value = [1, np.nan]
        with self.assertRaisesRegex(ValueError, "Statevector"):
            state_backend.get_state()
        state_backend._state.get_state_vector.assert_called_once_with()

        density_backend = DensityMatrixBackend.__new__(DensityMatrixBackend)
        density_backend.num_qubits = 1
        density_backend._state = mock.Mock()
        density_backend._state.get_density_matrix.return_value = np.eye(2, dtype=np.complex64)
        np.testing.assert_allclose(density_backend.get_state(), np.eye(2, dtype=np.complex128))
        density_backend._state.get_density_matrix.assert_called_once_with()

        density_backend = DensityMatrixBackend.__new__(DensityMatrixBackend)
        density_backend.num_qubits = 1
        density_backend._state = mock.Mock()
        density_backend._state.get_density_matrix.return_value = [[1, np.nan], [0, 1]]
        with self.assertRaisesRegex(ValueError, "Density matrix"):
            density_backend.get_state()
        density_backend._state.get_density_matrix.assert_called_once_with()

    def test_kernel_rejects_invalid_gate_targets_before_backend_dispatch(self):
        invalid_targets = (2, -1, 0.5, True, "0")

        for target in invalid_targets:
            with self.subTest(target=target):
                @kernel
                def bad_target():
                    rocq.qvec(1)
                    rocq.h(target)

                with mock.patch("rocq.kernel.get_backend") as patched_get_backend:
                    with self.assertRaisesRegex(ValueError, "Gate target"):
                        rocq.execute(bad_target, backend="state_vector")
                patched_get_backend.assert_not_called()

    def test_kernel_rejects_duplicate_gate_targets_before_backend_dispatch(self):
        duplicate_cases = (
            ("cnot", lambda q: rocq.cnot(q[0], q[0])),
            ("cp", lambda q: rocq.cp(0.125, q[0], q[0])),
            ("swap", lambda q: rocq.swap(q[1], q[1])),
            ("mcx", lambda q: rocq.mcx([q[0], q[1]], q[1])),
            ("cswap", lambda q: rocq.cswap(q[0], q[1], q[1])),
        )

        for name, apply_gate in duplicate_cases:
            with self.subTest(gate=name):
                @kernel
                def bad_gate():
                    q = rocq.qvec(3)
                    apply_gate(q)

                with mock.patch("rocq.kernel.get_backend") as patched_get_backend:
                    with self.assertRaisesRegex(ValueError, "target qubits must be distinct"):
                        rocq.execute(bad_gate, backend="state_vector")
                patched_get_backend.assert_not_called()

    def test_kernel_rejects_invalid_gate_arity_before_backend_dispatch(self):
        @kernel
        def missing_mcx_control():
            q = rocq.qvec(1)
            rocq.mcx([], q[0])

        with mock.patch("rocq.kernel.get_backend") as patched_get_backend:
            with self.assertRaisesRegex(ValueError, "at least 2 target"):
                rocq.execute(missing_mcx_control, backend="state_vector")
        patched_get_backend.assert_not_called()

    def test_kernel_rejects_invalid_gate_parameters_before_backend_dispatch(self):
        invalid_angles = (np.inf, np.nan, True, "0.5")

        for angle in invalid_angles:
            with self.subTest(angle=angle):
                @kernel
                def bad_parameter():
                    q = rocq.qvec(1)
                    rocq.rx(angle, q[0])

                with mock.patch("rocq.kernel.get_backend") as patched_get_backend:
                    with self.assertRaisesRegex(ValueError, "Gate parameter"):
                        rocq.execute(bad_parameter, backend="state_vector")
                patched_get_backend.assert_not_called()

    def test_direct_backends_reject_invalid_gate_angles_before_native_dispatch(self):
        from rocq.backends import DensityMatrixBackend, _HipStateVectorState

        invalid_angles = (np.inf, np.nan, True, "0.5")
        for angle in invalid_angles:
            with self.subTest(hip_statevector_angle=angle):
                state = _HipStateVectorState.__new__(_HipStateVectorState)
                state._handle = object()
                state._d_state = object()
                state._num_qubits = 1
                fake_hip_backend = mock.Mock()
                fake_hip_backend.rocqStatus.SUCCESS = "success"
                fake_hip_backend.apply_rx.side_effect = AssertionError(
                    "native gate dispatch should not receive invalid angles"
                )
                with mock.patch("rocq.backends.hip_backend", fake_hip_backend):
                    with self.assertRaisesRegex(ValueError, "angle must"):
                        state.apply_named_gate("rx", [0], {"theta": angle})

            with self.subTest(density_matrix_angle=angle):
                backend = DensityMatrixBackend.__new__(DensityMatrixBackend)
                backend.num_qubits = 1
                backend._state = mock.Mock()
                with self.assertRaisesRegex(ValueError, "angle must"):
                    backend._apply_op(GateOp("rx", [0], {"theta": angle}))
                backend._state.apply_gate_matrix.assert_not_called()

    def test_direct_backends_reject_invalid_gate_targets_before_native_dispatch(self):
        from rocq.backends import DensityMatrixBackend, StabilizerBackend, _HipStateVectorState

        invalid_single_targets = ([True], [1.5], ["0"], [-1], [1])
        for targets in invalid_single_targets:
            with self.subTest(hip_statevector_targets=targets):
                state = _HipStateVectorState.__new__(_HipStateVectorState)
                state._handle = object()
                state._d_state = object()
                state._num_qubits = 1
                fake_hip_backend = mock.Mock()
                fake_hip_backend.rocqStatus.SUCCESS = "success"
                fake_hip_backend.apply_x.side_effect = AssertionError(
                    "native gate dispatch should not receive invalid targets"
                )
                with mock.patch("rocq.backends.hip_backend", fake_hip_backend):
                    with self.assertRaises((TypeError, ValueError)):
                        state.apply_named_gate("x", targets, {})

            with self.subTest(density_matrix_targets=targets):
                backend = DensityMatrixBackend.__new__(DensityMatrixBackend)
                backend.num_qubits = 1
                backend._state = mock.Mock()
                with self.assertRaises((TypeError, ValueError)):
                    backend._apply_op(GateOp("x", targets, {}))
                backend._state.apply_gate_matrix.assert_not_called()

            with self.subTest(stabilizer_targets=targets):
                backend = StabilizerBackend(1)
                with self.assertRaises((TypeError, ValueError)):
                    backend._apply_op(GateOp("x", targets, {}))

        state = _HipStateVectorState.__new__(_HipStateVectorState)
        state._handle = object()
        state._d_state = object()
        state._num_qubits = 2
        fake_hip_backend = mock.Mock()
        fake_hip_backend.rocqStatus.SUCCESS = "success"
        fake_hip_backend.apply_cnot.side_effect = AssertionError(
            "native two-qubit dispatch should not receive duplicate targets"
        )
        with mock.patch("rocq.backends.hip_backend", fake_hip_backend):
            with self.assertRaisesRegex(ValueError, "target qubits must be distinct"):
                state.apply_named_gate("cnot", [0, 0], {})
            with self.assertRaisesRegex(ValueError, "expects 3 target"):
                state.apply_named_gate("ccx", [0, 1], {})

        backend = DensityMatrixBackend.__new__(DensityMatrixBackend)
        backend.num_qubits = 2
        backend._state = mock.Mock()
        with self.assertRaisesRegex(ValueError, "target qubits must be distinct"):
            backend._apply_op(GateOp("cnot", [0, 0], {}))
        backend._state.apply_cnot.assert_not_called()

    def test_hip_statevector_matrix_paths_validate_inputs_before_device_upload(self):
        from rocq.backends import _HipStateVectorState

        invalid_apply_matrix_cases = (
            ([0, 0], np.eye(4, dtype=np.complex128)),
            ([True], np.eye(2, dtype=np.complex128)),
            ([0, 1], np.eye(2, dtype=np.complex128)),
            ([0], np.array([[1.0, 0.0]], dtype=np.complex128)),
            ([0], np.array([[1.0, 0.0], [0.0, np.nan]], dtype=np.complex128)),
            ([0], [[1.0, 0.0], [0.0, True]]),
            ([0], [[1.0, 0.0], [0.0, "1.0"]]),
        )

        invalid_controlled_cases = (
            ([], [1], np.eye(2, dtype=np.complex128)),
            ([0], [0], np.eye(2, dtype=np.complex128)),
            ([0, 0], [1], np.eye(2, dtype=np.complex128)),
            ([0], [1, 1], np.eye(4, dtype=np.complex128)),
            ([0], [1], np.eye(4, dtype=np.complex128)),
            ([0], [1], np.array([[1.0, 0.0], [0.0, np.inf]], dtype=np.complex128)),
        )

        for targets, matrix in invalid_apply_matrix_cases:
            with self.subTest(apply_matrix_targets=targets, matrix=np.asarray(matrix, dtype=object).shape):
                state = _HipStateVectorState.__new__(_HipStateVectorState)
                state._handle = object()
                state._d_state = object()
                state._num_qubits = 2
                fake_hip_backend = mock.Mock()
                fake_hip_backend.create_device_matrix_from_numpy.side_effect = AssertionError(
                    "invalid apply_matrix inputs should fail before device upload"
                )
                with mock.patch("rocq.backends.hip_backend", fake_hip_backend):
                    with self.assertRaises((TypeError, ValueError)):
                        state.apply_matrix(targets, matrix)
                fake_hip_backend.create_device_matrix_from_numpy.assert_not_called()

        for controls, targets, matrix in invalid_controlled_cases:
            with self.subTest(controlled_controls=controls, targets=targets, matrix=np.asarray(matrix, dtype=object).shape):
                state = _HipStateVectorState.__new__(_HipStateVectorState)
                state._handle = object()
                state._d_state = object()
                state._num_qubits = 2
                fake_hip_backend = mock.Mock()
                fake_hip_backend.create_device_matrix_from_numpy.side_effect = AssertionError(
                    "invalid apply_controlled_matrix inputs should fail before device upload"
                )
                with mock.patch("rocq.backends.hip_backend", fake_hip_backend):
                    with self.assertRaises((TypeError, ValueError)):
                        state.apply_controlled_matrix(controls, targets, matrix)
                fake_hip_backend.create_device_matrix_from_numpy.assert_not_called()

        state = _HipStateVectorState.__new__(_HipStateVectorState)
        state._handle = object()
        state._d_state = object()
        state._num_qubits = 2
        fake_hip_backend = mock.Mock()
        fake_hip_backend.rocqStatus.SUCCESS = "success"
        fake_hip_backend.create_device_matrix_from_numpy.return_value = "device-matrix"
        fake_hip_backend.apply_matrix.return_value = "success"
        with mock.patch("rocq.backends.hip_backend", fake_hip_backend):
            state.apply_matrix([0], np.eye(2, dtype=np.complex128))

        uploaded_matrix = fake_hip_backend.create_device_matrix_from_numpy.call_args.args[0]
        self.assertEqual(uploaded_matrix.dtype, np.dtype(np.complex64))
        fake_hip_backend.apply_matrix.assert_called_once_with(
            state._handle,
            state._d_state,
            2,
            [0],
            "device-matrix",
            2,
        )

    def test_backends_reject_out_of_range_pauli_observables_before_native_dispatch(self):
        from rocq.backends import DensityMatrixBackend, StabilizerBackend, _HipStateVectorState

        invalid_operator = PauliOperator("Z1")
        partially_invalid_sum = PauliOperator("Z0") + PauliOperator("X1")

        state = _HipStateVectorState.__new__(_HipStateVectorState)
        state._handle = object()
        state._d_state = object()
        state._num_qubits = 1
        fake_hip_backend = mock.Mock()
        fake_hip_backend.get_expectation_value_z.side_effect = AssertionError(
            "native Pauli expectation should not receive out-of-range observable targets"
        )
        with mock.patch("rocq.backends.hip_backend", fake_hip_backend):
            with self.assertRaisesRegex(ValueError, "Pauli observable qubit index 1"):
                state.expectation(invalid_operator)
            with self.assertRaisesRegex(ValueError, "Pauli observable qubit index 1"):
                state.expectation(partially_invalid_sum)
        fake_hip_backend.get_expectation_value_z.assert_not_called()

        density_backend = DensityMatrixBackend.__new__(DensityMatrixBackend)
        density_backend.num_qubits = 1
        density_backend._uses_mock = False
        density_backend._state = mock.Mock()
        with self.assertRaisesRegex(ValueError, "Pauli observable qubit index 1"):
            density_backend.expectation(invalid_operator)
        density_backend._state.compute_expectation.assert_not_called()

        mock_density_backend = self._make_mock_density_backend(1)
        mock_density_backend._state = mock.Mock()
        with self.assertRaisesRegex(ValueError, "Pauli observable qubit index 1"):
            mock_density_backend.expectation(invalid_operator)
        mock_density_backend._state.compute_expectation.assert_not_called()

        mock_statevector_backend = self._make_mock_statevector_backend(1)
        with self.assertRaisesRegex(ValueError, "Pauli observable qubit index 1"):
            mock_statevector_backend.expectation(invalid_operator)

        stabilizer_backend = StabilizerBackend(1)
        with self.assertRaisesRegex(ValueError, "Pauli observable qubit index 1"):
            stabilizer_backend.expectation(invalid_operator)

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

    def test_execute_async_returns_future_and_uses_backend_contract(self):
        @kernel
        def bell():
            q = rocq.qvec(2)
            rocq.h(q[0])
            rocq.cnot(q[0], q[1])

        fake_backend = _FakeBackend()
        with ThreadPoolExecutor(max_workers=1) as executor:
            with mock.patch("rocq.kernel.get_backend", return_value=fake_backend):
                future = rocq.execute_async(bell, backend="state_vector", executor=executor)
                self.assertIsInstance(future, Future)
                result = future.result(timeout=5)

        self.assertEqual(result, "fake-state")
        self.assertEqual([op.name.lower() for op in fake_backend.ops], ["h", "cnot"])

    def test_get_state_async_returns_future_and_uses_backend_contract(self):
        @kernel
        def bell():
            q = rocq.qvec(2)
            rocq.h(q[0])
            rocq.cnot(q[0], q[1])

        fake_backend = _FakeBackend()
        with ThreadPoolExecutor(max_workers=1) as executor:
            with mock.patch("rocq.kernel.get_backend", return_value=fake_backend):
                future = rocq.get_state_async(bell, backend="state_vector", executor=executor)
                self.assertIsInstance(future, Future)
                result = future.result(timeout=5)

        self.assertEqual(result, "fake-state")
        self.assertEqual([op.name.lower() for op in fake_backend.ops], ["h", "cnot"])

    def test_sample_and_observe_async_return_futures(self):
        @kernel
        def bell():
            q = rocq.qvec(2)
            rocq.h(q[0])
            rocq.cnot(q[0], q[1])

        operator = PauliOperator("Z0 Z1")
        fake_backend = _FakeBackend()
        with ThreadPoolExecutor(max_workers=1) as executor:
            with mock.patch("rocq.kernel.get_backend", return_value=fake_backend):
                sample_future = rocq.sample_async(
                    bell,
                    7,
                    backend="state_vector",
                    qubits=[1],
                    executor=executor,
                )
                self.assertEqual(sample_future.result(timeout=5), {"0": 7})
                observe_future = rocq.observe_async(
                    bell,
                    operator,
                    backend="state_vector",
                    executor=executor,
                )
                self.assertEqual(observe_future.result(timeout=5), 1.25)

        self.assertEqual(fake_backend.sample_args, (7, [1]))
        self.assertIs(fake_backend.operator, operator)

    def test_async_validation_errors_are_reported_through_future(self):
        @kernel
        def prep_state():
            q = rocq.qvec(1)
            rocq.h(q[0])

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = rocq.sample_async(prep_state, 0, backend="state_vector", executor=executor)
            with self.assertRaisesRegex(ValueError, "shots must be"):
                future.result(timeout=5)

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

    def test_compiler_capabilities_expose_partial_supported_subset(self):
        with mock.patch.object(rocq_kernel_module, "rocquantum_bind", None):
            capabilities = rocq.compiler_capabilities()

        self.assertEqual(capabilities["status"], "partial")
        self.assertFalse(capabilities["binding_available"])
        self.assertFalse(capabilities["mlir_runtime_available"])
        self.assertEqual(capabilities["mlir_runtime_kind"], "missing_binding")
        self.assertEqual(capabilities["default_backend"], "hip_statevec")
        self.assertEqual(capabilities["supported_backends"], ["hip_statevec"])
        self.assertIn("Supported canonical MLIR gates", capabilities["supported_subset"])
        self.assertEqual(
            capabilities["supported_gate_groups"]["parametric_single_qubit"],
            ["rx", "ry", "rz", "p"],
        )
        self.assertIn(
            "mid-circuit measurement",
            capabilities["unsupported_features"],
        )
        self.assertIn(
            "release-wired TableGen dialect/op generation",
            capabilities["unsupported_features"],
        )
        self.assertIn(
            "release-wired adjoint-generation pass pipeline",
            capabilities["unsupported_features"],
        )
        self.assertEqual(capabilities["dialect_definition"]["active_source_tree"], "rocqCompiler/")
        self.assertEqual(
            capabilities["dialect_definition"]["legacy_scaffold_source_tree"],
            "rocquantum/include/rocquantum/Dialect and rocquantum/src/rocqCompiler",
        )
        self.assertFalse(capabilities["dialect_definition"]["release_tablegen_ops"])
        self.assertFalse(capabilities["dialect_definition"]["release_wired"])
        self.assertFalse(capabilities["dialect_definition"]["legacy_scaffold_release_linked"])
        self.assertEqual(
            capabilities["transform_pipeline"]["adjoint_generation"]["source_tree"],
            "rocquantum/src/rocqCompiler/Transforms/AdjointGeneration.cpp",
        )
        self.assertTrue(
            capabilities["transform_pipeline"]["adjoint_generation"]["legacy_scaffold_only"]
        )
        self.assertFalse(capabilities["transform_pipeline"]["adjoint_generation"]["release_wired"])
        self.assertFalse(
            capabilities["transform_pipeline"]["adjoint_generation"]["native_runtime_entry_point"]
        )
        self.assertIn("DisabledRuntimeMLIRCompiler", capabilities["mlir_runtime_note"])

    def test_compiler_capabilities_distinguish_binding_from_linked_mlir_runtime(self):
        disabled_binding = mock.Mock()
        disabled_binding.MLIR_COMPILER_ENABLED = False
        disabled_binding.MLIR_COMPILER_RUNTIME_KIND = "disabled_runtime_guard"
        enabled_binding = mock.Mock()
        enabled_binding.MLIR_COMPILER_ENABLED = True
        enabled_binding.MLIR_COMPILER_RUNTIME_KIND = "linked_runtime"

        with mock.patch.object(rocq_kernel_module, "rocquantum_bind", disabled_binding):
            disabled = rocq.compiler_capabilities()
        with mock.patch.object(rocq_kernel_module, "rocquantum_bind", enabled_binding):
            enabled = rocq.compiler_capabilities()

        self.assertTrue(disabled["binding_available"])
        self.assertFalse(disabled["mlir_runtime_available"])
        self.assertEqual(disabled["mlir_runtime_kind"], "disabled_runtime_guard")
        self.assertTrue(enabled["binding_available"])
        self.assertTrue(enabled["mlir_runtime_available"])
        self.assertEqual(enabled["mlir_runtime_kind"], "linked_runtime")

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

    def test_compile_and_execute_async_uses_native_compiler_binding(self):
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
                return [0.0, 1.0, 0.0, 0.0]

        fake_binding = mock.Mock()
        fake_binding.MLIRCompiler = _FakeCompiler

        with ThreadPoolExecutor(max_workers=1) as executor:
            with mock.patch.object(rocq_kernel_module, "rocquantum_bind", fake_binding):
                future = rocq.compile_and_execute_async(
                    bell,
                    strict=False,
                    executor=executor,
                )
                self.assertEqual(future.result(timeout=5), [0.0, 1.0, 0.0, 0.0])

        self.assertEqual(calls[0], ("init", 2, "hip_statevec"))
        self.assertEqual(calls[1][2], {"strict": False})

    def test_compile_and_execute_rejects_non_boolean_strict_option(self):
        @kernel
        def prep_state():
            q = rocq.qvec(1)
            rocq.h(q[0])

        for strict in (0, 1, "false", None, np.bool_(False)):
            with self.subTest(strict=strict):
                with self.assertRaisesRegex(ValueError, "strict must be a boolean"):
                    rocq.compile_and_execute(prep_state, strict=strict)

    def test_compile_and_execute_rejects_unsupported_compiler_backend_before_binding(self):
        @kernel
        def prep_state():
            q = rocq.qvec(1)
            rocq.h(q[0])

        for compiler_backend in ("state_vector", "cudaq", "", 1, True, None):
            with self.subTest(compiler_backend=compiler_backend):
                with mock.patch.object(rocq_kernel_module, "rocquantum_bind", None):
                    with self.assertRaisesRegex(ValueError, "compiler_backend"):
                        rocq.compile_and_execute(
                            prep_state,
                            compiler_backend=compiler_backend,
                        )

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

    def test_hip_statevector_backend_revalidates_hermitian_observable_before_native_dispatch(self):
        from rocq.backends import _HipStateVectorState

        calls = []

        class _FakeHipBackend:
            def get_expectation_matrix(self, handle, d_state, num_qubits, targets, matrix):
                calls.append((targets, matrix))
                return 0.5 + 0.0j

        def hermitian_operator_with(**updates):
            operator = HermitianOperator(np.diag([1.0, -1.0]), targets=[0])
            for name, value in updates.items():
                setattr(operator, name, value)
            return operator

        state = _HipStateVectorState.__new__(_HipStateVectorState)
        state._handle = object()
        state._d_state = object()
        state._num_qubits = 1

        invalid_operators = (
            hermitian_operator_with(matrix=[[True, 0.0], [0.0, 1.0]]),
            hermitian_operator_with(matrix=[["1.0", 0.0], [0.0, 1.0]]),
            hermitian_operator_with(matrix=[[np.nan, 0.0], [0.0, 1.0]]),
            hermitian_operator_with(matrix=np.ones((2, 3), dtype=np.complex128)),
            hermitian_operator_with(targets=[True]),
            hermitian_operator_with(targets=["0"]),
            hermitian_operator_with(targets=[0, 0]),
            hermitian_operator_with(targets=[-1]),
            hermitian_operator_with(targets=object()),
        )

        with mock.patch("rocq.backends.hip_backend", _FakeHipBackend()):
            for operator in invalid_operators:
                with self.subTest(operator=operator):
                    with self.assertRaisesRegex(ValueError, "HermitianOperator"):
                        state.expectation(operator)

        self.assertEqual(calls, [])

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

    def test_hip_statevector_backend_revalidates_sparse_observable_before_native_dispatch(self):
        from rocq.backends import _HipStateVectorState

        calls = []

        class _FakeHipBackend:
            def get_sparse_matrix_moments(self, handle, d_state, num_qubits, data, indices, indptr, rows, cols):
                calls.append((data, indices, indptr, rows, cols))
                return 0.25 + 0.0j, 0.5 + 0.0j

        def sparse_operator_with(**updates):
            operator = SparseHamiltonianOperator(
                data=np.array([1.0], dtype=np.complex128),
                indices=np.array([0], dtype=np.int64),
                indptr=np.array([0, 1, 1], dtype=np.int64),
                shape=(2, 2),
            )
            for name, value in updates.items():
                setattr(operator, name, value)
            return operator

        state = _HipStateVectorState.__new__(_HipStateVectorState)
        state._handle = object()
        state._d_state = object()
        state._num_qubits = 1

        invalid_operators = (
            sparse_operator_with(data=[True]),
            sparse_operator_with(data=["1.0"]),
            sparse_operator_with(data=[np.nan]),
            sparse_operator_with(indices=[True]),
            sparse_operator_with(indices=["0"]),
            sparse_operator_with(indptr=[0, True, 1]),
            sparse_operator_with(shape=(True, 2)),
            sparse_operator_with(shape="22"),
        )

        with mock.patch("rocq.backends.hip_backend", _FakeHipBackend()):
            for operator in invalid_operators:
                with self.subTest(operator=operator):
                    with self.assertRaisesRegex(ValueError, "SparseHamiltonianOperator"):
                        state.expectation(operator)

        self.assertEqual(calls, [])

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

    def test_hip_statevector_backend_falls_back_for_missing_pauli_expectation_helpers(self):
        from rocq.backends import _HipStateVectorState

        statevector_reads = []

        class _FakeHipBackend:
            def get_state_vector_full(self, handle, d_state, num_qubits, batch_size):
                statevector_reads.append((handle, d_state, num_qubits, batch_size))
                inv_sqrt2 = 1.0 / np.sqrt(2.0)
                return np.array([inv_sqrt2, 0.0, 0.0, inv_sqrt2], dtype=np.complex128)

        state = _HipStateVectorState.__new__(_HipStateVectorState)
        state._handle = object()
        state._d_state = object()
        state._num_qubits = 2
        operator = PauliOperator("X0 X1") + PauliOperator("Z0 Z1") + PauliOperator("Y0 Y1")

        with mock.patch("rocq.backends.hip_backend", _FakeHipBackend()):
            result = state.expectation(operator)

        self.assertAlmostEqual(result, 1.0)
        self.assertEqual(len(statevector_reads), 1)
        self.assertEqual(statevector_reads[0][2:], (2, 1))

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

    def test_density_backend_revalidates_mutable_noise_model_channels_before_dispatch(self):
        from rocq.backends import DensityMatrixBackend
        from rocq.kernel import GateOp

        class _FakeDensityState:
            def __init__(self):
                self.noise_calls = []

            def apply_gate_matrix(self, matrix, target, adjoint=False):
                return None

            def apply_bit_flip_channel(self, target, prob):
                self.noise_calls.append((target, prob))

        backend = DensityMatrixBackend.__new__(DensityMatrixBackend)
        backend.num_qubits = 1
        backend._uses_mock = False
        backend._state = _FakeDensityState()
        op = GateOp("x", [0], {})

        def mutated_noise(**updates):
            noise = rocq.NoiseModel()
            noise.add_channel("bit_flip", 0.25, on_qubits=[0])
            noise.get_channels()[0].update(updates)
            return noise

        invalid_channel_updates = (
            {"prob": True},
            {"prob": "0.25"},
            {"qubits": []},
            {"qubits": ["0"]},
            {"op": True},
            {"op": ""},
            {"type": True},
        )

        for updates in invalid_channel_updates:
            with self.subTest(updates=updates):
                backend._state.noise_calls.clear()
                with self.assertRaises((TypeError, ValueError)):
                    backend.run_ops([op], noise_model=mutated_noise(**updates))
                self.assertEqual(backend._state.noise_calls, [])

    def test_mock_density_backend_rejects_invalid_direct_noise_targets(self):
        backend = self._make_mock_density_backend(1)
        kraus = np.eye(2, dtype=np.complex64).reshape(1, 2, 2)

        invalid_targets = (0.5, True, "0", [], [0.5], [True], ["0"], [0, 0], [-1], [1])
        for targets in invalid_targets:
            with self.subTest(targets=targets):
                with self.assertRaises((TypeError, ValueError)):
                    backend.apply_noise("kraus", targets, 0.25, kraus_matrices=kraus)

        for targets in invalid_targets:
            with self.subTest(named_targets=targets):
                with self.assertRaises((TypeError, ValueError)):
                    backend.apply_noise("bit_flip", targets, 0.25)

        backend.apply_noise("bit_flip", np.int64(0), np.float64(0.25))

    def test_mock_density_backend_rejects_invalid_direct_noise_probabilities(self):
        backend = self._make_mock_density_backend(1)
        kraus = np.eye(2, dtype=np.complex64).reshape(1, 2, 2)

        invalid_probabilities = (-0.1, 1.1, np.nan, np.inf, True, "0.1")
        for probability in invalid_probabilities:
            with self.subTest(kraus_probability=probability):
                with self.assertRaisesRegex(ValueError, "between 0 and 1"):
                    backend.apply_noise("kraus", [0], probability, kraus_matrices=kraus)
            with self.subTest(named_probability=probability):
                with self.assertRaisesRegex(ValueError, "between 0 and 1"):
                    backend.apply_noise("bit_flip", [0], probability)

        with self.assertRaisesRegex(ValueError, "non-empty string"):
            backend.apply_noise("", [0], 0.25)
        with self.assertRaisesRegex(ValueError, "kraus_matrices"):
            backend.apply_noise("bit_flip", [0], 0.25, kraus_matrices=kraus)

    def test_density_backend_rejects_invalid_kraus_payloads_before_native_dispatch(self):
        from rocq.backends import DensityMatrixBackend

        class _FakeDensityState:
            instances = []

            def __init__(self, num_qubits):
                self.num_qubits = int(num_qubits)
                self.channels = []
                _FakeDensityState.instances.append(self)

            def apply_channel(self, targets, kraus_matrices):
                self.channels.append(
                    (
                        list(targets),
                        np.asarray(kraus_matrices).dtype,
                        np.asarray(kraus_matrices).shape,
                    )
                )

        class _FakeDensityModule:
            DensityMatrixState = _FakeDensityState

        with mock.patch("rocq.backends.dm_backend", _FakeDensityModule):
            backend = DensityMatrixBackend(1)
            valid_kraus = np.eye(2, dtype=np.complex128).reshape(1, 2, 2)
            backend.apply_noise("kraus", [0], 1.0, kraus_matrices=valid_kraus)

            invalid_kraus_payloads = (
                [[[True, 0.0], [0.0, 1.0]]],
                [[["1.0", 0.0], [0.0, 1.0]]],
                [[[np.nan, 0.0], [0.0, 1.0]]],
                [[[np.inf, 0.0], [0.0, 1.0]]],
            )
            for kraus in invalid_kraus_payloads:
                with self.subTest(kraus=kraus):
                    with self.assertRaisesRegex(ValueError, "Kraus matrices"):
                        backend.apply_noise("kraus", [0], 1.0, kraus_matrices=kraus)

        self.assertEqual(
            _FakeDensityState.instances[-1].channels,
            [([0], np.dtype(np.complex64), (1, 2, 2))],
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

        operations = [
            {
                "name": "RY",
                "rocq_name": "RY",
                "wires": [0],
                "params": [0.5],
                "param_indices": [0],
                "trainable_param_indices": [0],
                "trainable_param_positions": [0],
            }
        ]
        observables = [[{"coefficient": (1.0, 0.0), "pauli_string": "Z", "targets": [0]}]]

        self.assertTrue(runtime.supports_adjoint_jacobian())
        np.testing.assert_allclose(runtime.adjoint_jacobian(operations, observables, [0]), [[-0.25]])
        self.assertEqual(sim.calls, [(operations, observables, [0])])


if __name__ == "__main__":
    unittest.main()
