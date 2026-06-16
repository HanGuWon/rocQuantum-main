"""Source contracts for density-matrix channel, sampling, and observable paths."""

from __future__ import annotations

import os
import unittest


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DENSITY_SOURCE = os.path.join(
    _PROJECT_ROOT,
    "rocquantum",
    "src",
    "hipDensityMat",
    "hipDensityMat.cpp",
)
_DENSITY_HEADER = os.path.join(
    _PROJECT_ROOT,
    "rocquantum",
    "src",
    "hipDensityMat",
    "hipDensityMat.hpp",
)
_PUBLIC_HEADER = os.path.join(
    _PROJECT_ROOT,
    "rocquantum",
    "include",
    "rocquantum",
    "hipDensityMat.h",
)
_COMPAT_SOURCE = os.path.join(
    _PROJECT_ROOT,
    "rocquantum",
    "src",
    "hipDensityMat",
    "rocDensityMatCompat.cpp",
)
_PY_BINDINGS = os.path.join(
    _PROJECT_ROOT,
    "rocquantum",
    "src",
    "python",
    "py_hip_density_mat.cpp",
)
_CANONICAL_BACKEND = os.path.join(_PROJECT_ROOT, "rocq", "backends.py")
_ROCQ_INIT = os.path.join(_PROJECT_ROOT, "rocq", "__init__.py")
_README = os.path.join(_PROJECT_ROOT, "README.md")
_FEATURE_TRUTH_MATRIX = os.path.join(_PROJECT_ROOT, "FEATURE_TRUTH_MATRIX.md")


class TestDensityMatContract(unittest.TestCase):
    def test_generic_channel_and_sampling_are_public_api(self):
        with open(_DENSITY_HEADER, "r", encoding="utf-8") as f:
            header = f.read()
        with open(_PUBLIC_HEADER, "r", encoding="utf-8") as f:
            public_header = f.read()
        with open(_COMPAT_SOURCE, "r", encoding="utf-8") as f:
            compat = f.read()

        self.assertIn("hipDensityMatChannel_t", header)
        self.assertIn("HIPDENSITYMAT_MAX_QUBITS", header)
        self.assertIn("HIPDENSITYMAT_MAX_DENSE_OBSERVABLE_TARGETS", header)
        self.assertIn("HIPDENSITYMAT_MAX_KRAUS_TARGETS", header)
        self.assertIn("HIPDENSITYMAT_MAX_SAMPLE_QUBITS", header)
        self.assertIn("ROCDM_MAX_QUBITS", public_header)
        self.assertIn("ROCDM_MAX_KRAUS_TARGETS", public_header)
        self.assertIn("kraus_matrices_host", header)
        self.assertIn("num_targets", header)
        self.assertIn("target_qubits_host", header)
        self.assertIn("hipDensityMatSample", header)
        self.assertIn("rocdmChannel_t", public_header)
        self.assertIn("target_qubits_host", public_header)
        self.assertIn("rocdmSample", public_header)
        self.assertIn("rocdmComputeExpectationMatrix", public_header)
        self.assertIn("hipDensityMatSample", compat)
        self.assertIn("hipDensityMatComputeExpectationMatrix", compat)

    def test_channels_share_kraus_helper_instead_of_placeholder(self):
        with open(_DENSITY_SOURCE, "r", encoding="utf-8") as f:
            source = f.read()

        self.assertIn("apply_kraus_channel", source)
        self.assertIn("apply_multi_qubit_kraus_kernel", source)
        self.assertIn("apply_multi_qubit_kraus_channel", source)
        self.assertIn("hipDensityMatApplyChannel", source)
        self.assertIn("const hipDensityMatChannel_t* channel", source)
        self.assertIn("channel->target_qubits_host", source)
        self.assertIn("num_targets > HIPDENSITYMAT_MAX_KRAUS_TARGETS", source)
        self.assertNotIn("return HIPDENSITYMAT_STATUS_NOT_IMPLEMENTED;\n}", source.split("hipDensityMatApplyChannel", 1)[1])
        self.assertGreaterEqual(source.count("return apply_kraus_channel"), 5)

    def test_density_sampling_is_native_correctness_path(self):
        with open(_DENSITY_SOURCE, "r", encoding="utf-8") as f:
            source = f.read()

        self.assertIn("hipDensityMatSample", source)
        self.assertIn("extract_density_diagonal_kernel", source)
        self.assertIn("copy_density_diagonal_to_host", source)
        self.assertIn("accumulate_density_marginal_probabilities_kernel", source)
        self.assertIn("compute_density_marginal_probabilities", source)
        self.assertIn("atomicAdd(&outcome_probs[outcome], prob)", source)
        self.assertIn("std::discrete_distribution<uint64_t>", source)
        sample_block = source.split("hipDensityMatStatus_t hipDensityMatSample", 1)[1]
        self.assertIn("compute_density_marginal_probabilities", sample_block)
        self.assertNotIn("copy_density_diagonal_to_host", sample_block)
        self.assertNotIn("diagonal_probs", sample_block)
        self.assertNotIn("density_host[basis * dim + basis]", source)
        self.assertIn("validate_measured_qubits", source)
        self.assertIn("num_measured_qubits > HIPDENSITYMAT_MAX_SAMPLE_QUBITS", source)

    def test_dense_observable_expectation_is_native_for_supported_targets(self):
        with open(_DENSITY_SOURCE, "r", encoding="utf-8") as f:
            source = f.read()
        with open(_DENSITY_HEADER, "r", encoding="utf-8") as f:
            header = f.read()
        with open(_PY_BINDINGS, "r", encoding="utf-8") as f:
            bindings = f.read()
        with open(_CANONICAL_BACKEND, "r", encoding="utf-8") as f:
            backend = f.read()

        self.assertIn("hipDensityMatComputeExpectationMatrix", header)
        self.assertIn("density_matrix_expectation_matrix_kernel", source)
        self.assertIn("Tr(M rho)", header)
        self.assertIn("num_target_qubits > HIPDENSITYMAT_MAX_DENSE_OBSERVABLE_TARGETS", source)
        self.assertIn("HIPDENSITYMAT_STATUS_NOT_IMPLEMENTED", source)
        self.assertIn('.def("compute_expectation_matrix"', bindings)
        self.assertIn("hipDensityMatComputeExpectationMatrix", bindings)
        self.assertIn("def compute_expectation_matrix(self, matrix: np.ndarray, targets: Sequence[int])", backend)
        self.assertIn("native(matrix, targets)", backend)
        density_expectation_block = backend.split("class DensityMatrixBackend", 1)[1].split(
            "if isinstance(operator, SparseHamiltonianOperator)", 1
        )[0]
        self.assertLess(
            density_expectation_block.find("native(matrix, targets)"),
            density_expectation_block.find("_density_matrix_expectation_matrix"),
        )

    def test_python_binding_exposes_channel_and_sampling_surface(self):
        with open(_PY_BINDINGS, "r", encoding="utf-8") as f:
            bindings = f.read()

        self.assertIn('.def("apply_channel"', bindings)
        self.assertIn("2**len(target_qubits)", bindings)
        self.assertIn("target_qubit.cast<std::vector<int>>()", bindings)
        self.assertIn("targets.size() > HIPDENSITYMAT_MAX_KRAUS_TARGETS", bindings)
        self.assertIn('.def("apply_phase_flip_channel"', bindings)
        self.assertIn('.def("apply_amplitude_damping_channel"', bindings)
        self.assertIn('.def("sample"', bindings)
        self.assertIn("hipDensityMatApplyChannel", bindings)
        self.assertIn("hipDensityMatSample", bindings)

    def test_canonical_density_backend_uses_native_sample(self):
        with open(_CANONICAL_BACKEND, "r", encoding="utf-8") as f:
            backend = f.read()

        self.assertIn("def sample(self, measured_qubits: Sequence[int], num_shots: int)", backend)
        self.assertIn("return self._state.sample(list(measured_qubits), int(num_shots))", backend)
        self.assertIn('shots = _validate_positive_integer(shots, "shots")', backend)
        self.assertIn("measured_qubits = _normalize_sample_qubits(qubits, self.num_qubits)", backend)
        self.assertIn("raw_results = self._state.sample(measured_qubits, shots)", backend)
        self.assertIn("apply_phase_flip_channel", backend)
        self.assertIn("apply_amplitude_damping_channel", backend)
        self.assertNotIn("does not expose a native sampling path yet", backend)

    def test_canonical_density_backend_decomposes_multi_control_gates(self):
        with open(_CANONICAL_BACKEND, "r", encoding="utf-8") as f:
            backend = f.read()

        self.assertIn("def _apply_ccx_decomposition", backend)
        self.assertIn("def _apply_cswap_decomposition", backend)
        self.assertIn("def _apply_mcx_decomposition", backend)
        self.assertIn('if name in {"ccx", "toffoli"}', backend)
        self.assertIn('if name in {"cswap", "fredkin"}', backend)
        self.assertIn("larger MCX needs an explicit ancilla/decomposition policy", backend)

    def test_density_matrix_capability_api_documents_cudensitymat_boundary(self):
        with open(_CANONICAL_BACKEND, "r", encoding="utf-8") as f:
            backend = f.read()
        with open(_ROCQ_INIT, "r", encoding="utf-8") as f:
            rocq_init = f.read()
        with open(_README, "r", encoding="utf-8") as f:
            readme = f.read()
        with open(_FEATURE_TRUTH_MATRIX, "r", encoding="utf-8") as f:
            matrix = f.read()

        self.assertIn("def density_matrix_capabilities", backend)
        self.assertIn("GPU-resident cuDensityMat-style channel descriptors", backend)
        self.assertIn("device_marginal_probabilities_host_shot_drawing", backend)
        self.assertIn("native_up_to_4_target_qubits", backend)
        self.assertIn("max_kraus_channel_targets", backend)
        self.assertIn("max_qubits_before_dense_size_overflow", backend)
        self.assertIn("capability_query_is_runtime_proof", backend)
        self.assertIn("density_matrix_capabilities", rocq_init)
        self.assertIn("rocq.density_matrix_capabilities()", readme)
        self.assertIn("cuDensityMat descriptor/sampling boundaries", readme)
        self.assertIn("per_kraus_kernel_correctness_path", matrix)
        self.assertIn("GPU-resident RNG/CDF sampling unsupported", matrix)


if __name__ == "__main__":
    unittest.main()
