"""Source contracts for hipTensorNet optimizer, dtype, and slicing behavior."""

from __future__ import annotations

import os
import re
import unittest


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TENSORNET_SOURCE = os.path.join(
    _PROJECT_ROOT,
    "rocquantum",
    "src",
    "hipTensorNet",
    "hipTensorNet.cpp",
)
_TENSORNET_HEADER = os.path.join(
    _PROJECT_ROOT,
    "rocquantum",
    "include",
    "rocquantum",
    "hipTensorNet.h",
)
_TENSORNET_API = os.path.join(
    _PROJECT_ROOT,
    "rocquantum",
    "include",
    "rocquantum",
    "hipTensorNet_api.h",
)
_TENSOR_UTIL_SOURCE = os.path.join(
    _PROJECT_ROOT,
    "rocquantum",
    "src",
    "hipTensorNet",
    "rocTensorUtil.cpp",
)
_TENSOR_UTIL_KERNELS = os.path.join(
    _PROJECT_ROOT,
    "rocquantum",
    "src",
    "hipTensorNet",
    "rocTensorUtil_kernels.hip",
)
_PERMUTATION_KERNELS = os.path.join(
    _PROJECT_ROOT,
    "rocquantum",
    "src",
    "hipTensorNet",
    "PermutationKernels.hip",
)
_HIP_STATEVEC_HEADER = os.path.join(
    _PROJECT_ROOT,
    "rocquantum",
    "include",
    "rocquantum",
    "hipStateVec.h",
)
_TENSORNET_CMAKE = os.path.join(
    _PROJECT_ROOT,
    "rocquantum",
    "src",
    "hipTensorNet",
    "CMakeLists.txt",
)
_PACKAGE_CONFIG_TEMPLATE = os.path.join(
    _PROJECT_ROOT,
    "cmake",
    "rocQuantumConfig.cmake.in",
)
_TENSOR_UTIL_HEADER = os.path.join(
    _PROJECT_ROOT,
    "rocquantum",
    "include",
    "rocquantum",
    "rocTensorUtil.h",
)
_WORKSPACE_SOURCE = os.path.join(
    _PROJECT_ROOT,
    "rocquantum",
    "src",
    "hipTensorNet",
    "rocWorkspaceManager.cpp",
)
_TENSOR_UTIL_TEST = os.path.join(
    _PROJECT_ROOT,
    "rocquantum",
    "src",
    "hipTensorNet",
    "test_hipTensorNet_rocTensorUtil.cpp",
)
_SLICING_TEST = os.path.join(
    _PROJECT_ROOT,
    "rocquantum",
    "src",
    "hipTensorNet",
    "test_hipTensorNet_slicing.cpp",
)
_STALE_PATHFINDER_SOURCE = os.path.join(_PROJECT_ROOT, "rocquantum", "src", "Pathfinder.cpp")
_STALE_PATHFINDER_HEADER = os.path.join(
    _PROJECT_ROOT,
    "rocquantum",
    "include",
    "rocquantum",
    "Pathfinder.h",
)
_BINDINGS_SOURCE = os.path.join(_PROJECT_ROOT, "python", "rocq", "bindings.cpp")
_README = os.path.join(_PROJECT_ROOT, "README.md")
_FEATURE_TRUTH_MATRIX = os.path.join(_PROJECT_ROOT, "FEATURE_TRUTH_MATRIX.md")


class TestTensorNetContract(unittest.TestCase):
    def test_capability_api_reports_dtype_optimizer_and_slicing(self):
        with open(_TENSORNET_API, "r", encoding="utf-8") as f:
            api = f.read()
        with open(_TENSORNET_HEADER, "r", encoding="utf-8") as f:
            header = f.read()
        with open(_TENSORNET_SOURCE, "r", encoding="utf-8") as f:
            source = f.read()

        self.assertIn("hipTensorNetCapabilities_t", api)
        self.assertIn("supports_c64", api)
        self.assertIn("supports_c128", api)
        self.assertIn("supports_pathfinder_greedy", api)
        self.assertIn("supports_runtime_slicing", api)
        self.assertIn("runtime_slicing_is_limited_pair_gemm", api)
        self.assertIn("supports_open_index_slicing", api)
        self.assertIn("supports_mixed_precision", api)
        self.assertIn("supports_simultaneous_c64_c128", api)
        self.assertIn("ROC_TENSORNET_MAX_PERMUTATION_MODES", api)
        self.assertIn("max_tensor_permutation_modes", api)
        self.assertIn("permutation_modes_are_hard_limited", api)
        self.assertIn("ROC_TENSORNET_COMPILED_COMPLEX_DTYPE", header)
        self.assertIn("rocTensorNetworkGetCapabilities", header)
        self.assertIn("rocTensorNetworkGetCapabilities", source)
        self.assertIn("runtime_slicing_is_limited_pair_gemm = 1", source)
        self.assertIn("supports_open_index_slicing = 0", source)
        self.assertIn("supports_mixed_precision = 0", source)
        self.assertIn("supports_simultaneous_c64_c128 = 0", source)
        self.assertIn(
            "max_tensor_permutation_modes = ROC_TENSORNET_MAX_PERMUTATION_MODES",
            source,
        )
        self.assertIn("permutation_modes_are_hard_limited = 1", source)

    def test_permutation_rank_limit_is_explicit_and_guarded(self):
        with open(_TENSOR_UTIL_SOURCE, "r", encoding="utf-8") as f:
            source = f.read()
        with open(_TENSOR_UTIL_KERNELS, "r", encoding="utf-8") as f:
            kernels = f.read()
        with open(_TENSOR_UTIL_HEADER, "r", encoding="utf-8") as f:
            header = f.read()
        with open(_BINDINGS_SOURCE, "r", encoding="utf-8") as f:
            bindings = f.read()

        self.assertIn("#include \"rocquantum/hipTensorNet_api.h\"", source)
        self.assertIn("input_tensor->rank() > ROC_TENSORNET_MAX_PERMUTATION_MODES", source)
        self.assertIn("return ROCQ_STATUS_NOT_IMPLEMENTED", source)
        self.assertIn("ROC_TENSORNET_MAX_PERMUTATION_MODES", kernels)
        self.assertNotIn("input_multi_indices[16]", kernels)
        self.assertNotIn("output_multi_indices[16]", kernels)
        self.assertNotIn("output_coords[16]", kernels)
        self.assertIn("ROC_TENSORNET_MAX_PERMUTATION_MODES", header)
        self.assertIn("max_tensor_permutation_modes", bindings)
        self.assertIn("permutation_modes_are_hard_limited", bindings)

    def test_permutation_template_instantiations_use_concrete_complex_aliases(self):
        with open(_PERMUTATION_KERNELS, "r", encoding="utf-8") as f:
            source = f.read()
        with open(_HIP_STATEVEC_HEADER, "r", encoding="utf-8") as f:
            header = f.read()

        self.assertIn("typedef hipFloatComplex rocFloatComplex", header)
        self.assertIn("typedef hipDoubleComplex rocDoubleComplex", header)
        self.assertIn("launch_permute_tensor<rocFloatComplex>", source)
        self.assertIn("rocFloatComplex* output_tensor", source)
        self.assertIn("launch_permute_tensor<rocDoubleComplex>", source)
        self.assertIn("rocDoubleComplex* output_tensor", source)
        self.assertNotIn("launch_permute_tensor<rocComplex>(T*", source)
        self.assertNotIn("launch_permute_tensor<rocDoubleComplex>(T*", source)

    def test_permutation_cleanup_gotos_do_not_cross_block_initialization(self):
        with open(_TENSOR_UTIL_SOURCE, "r", encoding="utf-8") as f:
            source = f.read()

        first_cleanup_jump = source.index("goto perm_cleanup")
        self.assertLess(source.index("const unsigned int threads_per_block"), first_cleanup_jump)
        self.assertLess(source.index("unsigned int num_blocks = 0"), first_cleanup_jump)

    def test_public_tensor_view_implementation_is_linked_into_tensornet(self):
        with open(_TENSORNET_CMAKE, "r", encoding="utf-8") as f:
            cmake = f.read()

        self.assertIn("TensorView.cpp", cmake)
        self.assertNotIn("AccumulationKernels.hip.cpp", cmake)

    def test_dtype_support_is_build_precision_gated(self):
        with open(_TENSORNET_SOURCE, "r", encoding="utf-8") as f:
            tensornet_source = f.read()
        with open(_TENSOR_UTIL_SOURCE, "r", encoding="utf-8") as f:
            tensor_util_source = f.read()

        self.assertIn("tensor_dtype_supported_by_build", tensornet_source)
        self.assertIn("ROC_TENSORNET_COMPILED_COMPLEX_DTYPE", tensornet_source)
        self.assertIn("return ROCQ_STATUS_NOT_IMPLEMENTED", tensornet_source)
        self.assertIn("handle->dtype != ROC_TENSORNET_COMPILED_COMPLEX_DTYPE", tensornet_source)
        self.assertIn("rocsolver_cgesvd", tensornet_source)
        self.assertIn("rocsolver_zgesvd", tensornet_source)
        self.assertIn("rocblas_cgemm", tensor_util_source)
        self.assertIn("rocblas_zgemm", tensor_util_source)
        self.assertNotIn("if (dtype != ROC_DATATYPE_C64)", tensornet_source)
        self.assertNotIn("handle->dtype != ROC_DATATYPE_C64", tensornet_source)

    def test_native_tensornet_uses_public_hip_and_rocblas_status_symbols(self):
        with open(_TENSORNET_SOURCE, "r", encoding="utf-8") as f:
            tensornet_source = f.read()
        with open(_TENSOR_UTIL_SOURCE, "r", encoding="utf-8") as f:
            tensor_util_source = f.read()

        self.assertIn(
            "const hipError_t copy_status = hipMemcpyAsync(",
            tensornet_source,
        )
        self.assertIn(
            "return copy_status == hipSuccess ? ROCQ_STATUS_SUCCESS : ROCQ_STATUS_HIP_ERROR;",
            tensornet_source,
        )
        self.assertNotIn("status_from_hip(", tensornet_source)
        self.assertEqual(tensor_util_source.count("rocblas_operation_none"), 4)
        self.assertNotIn("ROCBLAS_OPERATION_NONE", tensor_util_source)

    def test_svd_binding_does_not_allocate_unused_workspace(self):
        with open(_BINDINGS_SOURCE, "r", encoding="utf-8") as f:
            bindings = f.read()
        with open(_TENSORNET_HEADER, "r", encoding="utf-8") as f:
            header = f.read()
        with open(_TENSORNET_SOURCE, "r", encoding="utf-8") as f:
            source = f.read()

        self.assertIn("rocTensorSVD(handle.get(), &U, &S, &Vh, &A, nullptr)", bindings)
        self.assertIn("returning (U, S, Vh)", bindings)
        self.assertNotIn("DeviceBuffer workspace(1, 1)", bindings)
        self.assertIn("Reserved for future rocSOLVER workspace control", header)
        self.assertIn("(void)workspace", source)

    def test_tensor_util_contract_header_matches_real_pair_contraction(self):
        with open(_TENSOR_UTIL_HEADER, "r", encoding="utf-8") as f:
            header = f.read()
        with open(_TENSOR_UTIL_SOURCE, "r", encoding="utf-8") as f:
            source = f.read()

        self.assertIn("parse_simple_einsum_spec", source)
        self.assertIn("rocTensorContractPair_internal", source)
        self.assertIn("simplified einsum parser plus rocBLAS GEMM", header)
        self.assertIn("rocTensorContractPair_internal", header)
        self.assertNotIn("current implementation is a STUB", header)
        self.assertNotIn("Currently a placeholder", header)
        self.assertNotIn("ROCQ_STATUS_NOT_IMPLEMENTED for actual contraction logic", header)

    def test_tensornet_build_includes_workspace_implementation(self):
        with open(_TENSORNET_CMAKE, "r", encoding="utf-8") as f:
            cmake = f.read()

        target_sources = cmake.split("add_library(rocqsim_tensornet", 1)[1].split(")", 1)[0]
        active_sources = {
            line.split("#", 1)[0].strip()
            for line in target_sources.splitlines()
            if line.split("#", 1)[0].strip()
        }
        self.assertIn("rocWorkspaceManager.cpp", active_sources)

    def test_tensor_stride_and_workspace_alignment_helpers_are_reused(self):
        with open(_TENSOR_UTIL_HEADER, "r", encoding="utf-8") as f:
            header = f.read()
        with open(_WORKSPACE_SOURCE, "r", encoding="utf-8") as f:
            workspace = f.read()

        constructor_match = re.search(
            r"rocTensor\s*\(\s*rocComplex\s*\*\s*data\b.*?(?=\s*~rocTensor\s*\()",
            header,
            re.DOTALL,
        )
        self.assertIsNotNone(constructor_match)
        constructor = constructor_match.group(0)
        self.assertIn("calculate_strides();", constructor)
        self.assertNotIn("strides_[0] = 1;", constructor)
        self.assertRegex(
            workspace,
            r"size_t\s+align_up\s*\(\s*size_t\s+value\s*,\s*size_t\s+alignment\s*\)\s*noexcept",
        )
        self.assertEqual(
            len(
                re.findall(
                    r"\balign_up\s*\(\s*current_offset_bytes_\s*,\s*alignment_\s*\)",
                    workspace,
                )
            ),
            2,
        )
        self.assertIn("#include <limits>", workspace)
        self.assertIn("#include <string>", workspace)
        self.assertIn(
            "num_elements > std::numeric_limits<size_t>::max() / sizeof(rocComplex)",
            workspace,
        )
        self.assertIn(
            "value > std::numeric_limits<size_t>::max() - padding",
            workspace,
        )
        self.assertIn("return std::numeric_limits<size_t>::max();", workspace)
        self.assertIn("aligned_offset <= total_size_bytes_", workspace)
        self.assertIn("requested_bytes <= total_size_bytes_ - aligned_offset", workspace)

    def test_standalone_tensornet_tests_propagate_failure_counts(self):
        for path in (_TENSOR_UTIL_TEST, _SLICING_TEST):
            with self.subTest(path=path):
                with open(path, "r", encoding="utf-8") as f:
                    source = f.read()

                self.assertIn("int RUN_ALL_TESTS()", source)
                main_match = re.search(
                    r"int\s+main\s*\(\s*\)\s*\{(?P<body>.*?)^\}",
                    source,
                    re.DOTALL | re.MULTILINE,
                )
                self.assertIsNotNone(main_match)
                main_body = main_match.group("body")
                self.assertRegex(
                    main_body,
                    r"const\s+int\s+failed_count\s*=\s*RUN_ALL_TESTS\s*\(\s*\)\s*;",
                )
                self.assertRegex(
                    main_body,
                    r"teardown_global_test_resources\s*\(\s*\)\s*;\s*return\s+failed_count\s*;",
                )
                self.assertNotRegex(main_body, r"return\s+0\s*;")

    def test_optimizer_and_memory_limit_are_not_silent(self):
        with open(_TENSORNET_SOURCE, "r", encoding="utf-8") as f:
            source = f.read()
        with open(_TENSOR_UTIL_SOURCE, "r", encoding="utf-8") as f:
            tensor_util_source = f.read()

        self.assertIn("validate_optimizer_config", source)
        self.assertIn("pathfinder_algorithm_available", source)
        self.assertIn("effective_pathfinder_algorithm", source)
        self.assertIn(": ROCTN_PATHFINDER_ALGO_GREEDY", source)
        self.assertIn("ROCTN_PATHFINDER_ALGO_KAHYPAR", source)
        self.assertIn("ROCTN_PATHFINDER_ALGO_METIS", source)
        self.assertIn("selection_cost_for_algorithm", source)
        self.assertIn("memory_limit_bytes", source)
        self.assertIn("num_slices", source)
        self.assertIn("supports_memory_limit_planning = 1", source)
        self.assertIn("supports_runtime_slicing = 1", source)
        self.assertIn("config->memory_limit_bytes", source)
        self.assertIn("config->num_slices", source)
        self.assertIn("runtime_slices", tensor_util_source)
        self.assertIn("beta_accumulate", tensor_util_source)
        self.assertIn("gemm_K_slice", tensor_util_source)
        self.assertIn("pair_bytes", tensor_util_source)
        self.assertNotIn("if (!pathfinder_algorithm_available(config.pathfinder_algorithm))", source)

    def test_optional_metis_build_guard_and_runtime_partitioning(self):
        with open(_TENSORNET_CMAKE, "r", encoding="utf-8") as f:
            cmake = f.read()
        with open(_PACKAGE_CONFIG_TEMPLATE, "r", encoding="utf-8") as f:
            package_config = f.read()
        with open(_TENSORNET_SOURCE, "r", encoding="utf-8") as f:
            source = f.read()

        self.assertIn("ROCQUANTUM_TENSORNET_ENABLE_METIS", cmake)
        self.assertIn("find_path(ROCQUANTUM_METIS_INCLUDE_DIR NAMES metis.h)", cmake)
        self.assertIn("find_library(ROCQUANTUM_METIS_LIBRARY NAMES metis)", cmake)
        self.assertIn("message(FATAL_ERROR", cmake)
        self.assertIn("HAS_METIS=1", cmake)
        self.assertIn("target_link_libraries(rocqsim_tensornet PUBLIC METIS::METIS)", cmake)
        self.assertNotIn("PUBLIC ${METIS_LIBRARY}", cmake)
        self.assertIn("ROCQUANTUM_BUILT_WITH_METIS ON PARENT_SCOPE", cmake)
        self.assertIn(
            "set(rocQuantum_BUILT_WITH_METIS @ROCQUANTUM_BUILT_WITH_METIS@)",
            package_config,
        )
        self.assertIn("if(rocQuantum_BUILT_WITH_METIS", package_config)
        self.assertIn("add_library(METIS::METIS UNKNOWN IMPORTED)", package_config)
        self.assertIn("rocQuantum_NOT_FOUND_MESSAGE", package_config)

        self.assertIn("#include <metis.h>", source)
        self.assertIn("metis_partition_active_tensors", source)
        self.assertIn("METIS_SetDefaultOptions", source)
        self.assertIn("METIS_OPTION_NITER", source)
        self.assertIn("METIS_PartGraphKway", source)
        self.assertIn("ROCTN_PATHFINDER_ALGO_METIS", source)
        self.assertIn("using_metis_partition", source)
        self.assertIn("add_metis_partition_penalty", source)

    def test_stale_unwired_pathfinder_sources_are_not_shipped(self):
        with open(_TENSORNET_SOURCE, "r", encoding="utf-8") as f:
            source = f.read()

        self.assertFalse(os.path.exists(_STALE_PATHFINDER_SOURCE))
        self.assertFalse(os.path.exists(_STALE_PATHFINDER_HEADER))
        self.assertIn("pathfinder_algorithm_available", source)
        self.assertIn("effective_pathfinder_algorithm", source)
        self.assertIn("metis_partition_active_tensors", source)
        self.assertNotIn("findMetisPath", source)

    def test_kahypar_build_option_fails_fast_until_release_wired(self):
        with open(_TENSORNET_CMAKE, "r", encoding="utf-8") as f:
            cmake = f.read()
        with open(_TENSORNET_SOURCE, "r", encoding="utf-8") as f:
            source = f.read()

        self.assertIn("ROCQUANTUM_TENSORNET_ENABLE_KAHYPAR", cmake)
        self.assertIn("is not release-wired yet", cmake)
        self.assertIn("ROCTN_PATHFINDER_ALGO_GREEDY", cmake)
        self.assertNotIn("target_compile_definitions(rocqsim_tensornet PRIVATE HAS_KAHYPAR=1)", cmake)
        self.assertIn("supports_pathfinder_kahypar = 0", source)

    def test_python_binding_passes_optimizer_algorithm(self):
        with open(_BINDINGS_SOURCE, "r", encoding="utf-8") as f:
            source = f.read()

        self.assertIn("pathfinder_algorithm", source)
        self.assertIn("ROCTN_PATHFINDER_ALGO_GREEDY", source)
        self.assertIn("ROCTN_PATHFINDER_ALGO_KAHYPAR", source)
        self.assertIn("ROCTN_PATHFINDER_ALGO_METIS", source)
        self.assertIn("get_tensornet_capabilities", source)
        self.assertIn("warn_tensornet_pathfinder_fallback(config)", source)

    def test_python_binding_reports_actionable_tensornet_status_errors(self):
        with open(_BINDINGS_SOURCE, "r", encoding="utf-8") as f:
            source = f.read()

        self.assertIn("tensornet_status_message", source)
        self.assertIn("tensornet_contract_status_message", source)
        self.assertIn("get_tensornet_capabilities()", source)
        self.assertIn("pathfinder_algorithm=", source)
        self.assertIn("tensornet_pathfinder_supported", source)
        self.assertIn("falling back to greedy", source)
        self.assertIn("METIS/KAHYPAR parity", source)
        self.assertIn("limited runtime K-sliced GEMM execution", source)
        self.assertIn("cuTensorNet-style open-index slicing", source)
        self.assertIn("num_slices must be non-negative", source)
        self.assertIn("runtime_slicing_kind", source)
        self.assertIn("limited_pair_contraction_k_sliced_gemm", source)
        self.assertIn("supports_open_index_slicing", source)
        self.assertIn("supports_mixed_precision", source)
        self.assertIn("supports_simultaneous_c64_c128", source)
        self.assertNotIn(
            'rocTensorNetworkContract failed: " + std::to_string(status)',
            source,
        )

    def test_python_binding_warns_when_slicing_knobs_use_limited_runtime_slicing(self):
        with open(_BINDINGS_SOURCE, "r", encoding="utf-8") as f:
            source = f.read()

        self.assertIn("warn_tensornet_limited_runtime_slicing", source)
        self.assertIn("PyErr_WarnEx(PyExc_RuntimeWarning", source)
        self.assertIn("limited runtime", source)
        self.assertIn("K-sliced GEMM execution", source)
        self.assertIn("supports_runtime_slicing", source)
        self.assertIn("warn_tensornet_limited_runtime_slicing(config)", source)

    def test_python_binding_uses_simulator_stream_and_real_rocblas_handle(self):
        with open(_BINDINGS_SOURCE, "r", encoding="utf-8") as f:
            source = f.read()

        self.assertIn("rocsvGetStream(self.get_sim_handle().get(), &stream)", source)
        self.assertIn("tensornet_status_message(\"rocsvGetStream\", stream_status)", source)
        self.assertIn("rocblas_handle blas_handle_ = nullptr", source)
        self.assertIn("rocblas_create_handle(&blas_handle_)", source)
        self.assertIn("rocblas_destroy_handle(blas_handle_)", source)
        self.assertIn("self.get_blas_handle()", source)
        self.assertIn("rocTensorNetworkContract", source)
        self.assertNotIn("Using placeholders for now", source)
        self.assertNotIn("rocblas_handle blas_h = nullptr; // Placeholder", source)
        self.assertNotIn("hipStream_t stream = 0; // Placeholder", source)
        self.assertNotIn("rocblas_create_handle(&blas_h)", source)
        self.assertNotIn("rocblas_destroy_handle(blas_h)", source)

    def test_docs_describe_limited_slicing_and_mixed_precision_boundary(self):
        with open(_README, "r", encoding="utf-8") as f:
            readme = f.read()
        with open(_FEATURE_TRUTH_MATRIX, "r", encoding="utf-8") as f:
            matrix = f.read()

        self.assertIn("deterministic runtime K-sliced GEMM accumulation", readme)
        self.assertIn("limited_pair_contraction_k_sliced_gemm", readme)
        self.assertIn("open-index slicing, mixed precision", readme)
        self.assertIn("active simulator stream", readme)
        self.assertIn("reuse a rocBLAS handle", readme)
        self.assertIn("stale unwired Pathfinder scaffold is not shipped", readme)
        self.assertIn("mixed precision remains a documented future lane", readme)
        self.assertIn("placeholder BLAS/stream handles", matrix)
        self.assertIn("stale unwired `Pathfinder.cpp` / public `Pathfinder.h` scaffold is not shipped", matrix)
        self.assertIn("not full cuTensorNet-style open-index slicing", matrix)
        self.assertIn("supports_open_index_slicing=False", matrix)
        self.assertIn("supports_mixed_precision=False", matrix)
        self.assertIn("mixed precision remains a documented future lane", matrix)
        self.assertIn("No simultaneous runtime C64/C128 support", matrix)


if __name__ == "__main__":
    unittest.main()
