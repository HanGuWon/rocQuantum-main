"""Source contracts for RCCL-backed distributed reduction paths."""

from __future__ import annotations

import os
import unittest


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STATEVEC_SOURCE = os.path.join(
    _PROJECT_ROOT,
    "rocquantum",
    "src",
    "hipStateVec",
    "hipStateVec.cpp",
)
_STATEVEC_HEADER = os.path.join(
    _PROJECT_ROOT,
    "rocquantum",
    "include",
    "rocquantum",
    "hipStateVec.h",
)
_STATEVEC_CMAKE = os.path.join(
    _PROJECT_ROOT,
    "rocquantum",
    "src",
    "hipStateVec",
    "CMakeLists.txt",
)
_BINDINGS_SOURCE = os.path.join(_PROJECT_ROOT, "python", "rocq", "bindings.cpp")


class TestRcclDistributedContract(unittest.TestCase):
    def test_cmake_exposes_rccl_compile_define(self):
        with open(_STATEVEC_CMAKE, "r", encoding="utf-8") as f:
            cmake = f.read()

        self.assertIn("find_package(rccl QUIET)", cmake)
        self.assertIn("ROCQ_HAVE_RCCL=1", cmake)
        self.assertIn("TARGET rccl", cmake)
        self.assertIn("target_link_libraries(hipStateVec PUBLIC rccl)", cmake)
        self.assertIn("rccl::rccl", cmake)
        self.assertIn("benchmark_hipStateVec_distributed_reductions", cmake)

    def test_rccl_lifecycle_is_explicit(self):
        with open(_STATEVEC_SOURCE, "r", encoding="utf-8") as f:
            source = f.read()

        self.assertIn("#ifdef ROCQ_HAVE_RCCL", source)
        self.assertIn("distributedComms", source)
        self.assertIn("distributedRcclReady", source)
        self.assertIn("ncclCommInitAll", source)
        self.assertIn("ncclCommDestroy", source)
        self.assertIn("ROCQ_DISTRIBUTED_COMM", source)
        self.assertIn("ROCQ_REQUIRE_RCCL", source)
        self.assertIn("ROCQ_DISABLE_RCCL", source)

    def test_runtime_exposes_active_distributed_backend(self):
        with open(_STATEVEC_HEADER, "r", encoding="utf-8") as f:
            header = f.read()
        with open(_STATEVEC_SOURCE, "r", encoding="utf-8") as f:
            source = f.read()
        with open(_BINDINGS_SOURCE, "r", encoding="utf-8") as f:
            bindings = f.read()

        self.assertIn("rocsvDistributedBackend_t", header)
        self.assertIn("ROCSV_DISTRIBUTED_BACKEND_NONE", header)
        self.assertIn("ROCSV_DISTRIBUTED_BACKEND_HOST_FALLBACK", header)
        self.assertIn("ROCSV_DISTRIBUTED_BACKEND_RCCL", header)
        self.assertIn("rocsvGetDistributedBackend", header)
        self.assertIn("rocsvDistributedBackendName", header)
        self.assertIn("distributed_active_backend", source)
        self.assertIn("distributed_rccl_ready(handle)", source)
        self.assertIn("return ROCSV_DISTRIBUTED_BACKEND_RCCL", source)
        self.assertIn("distributed_host_fallback_enabled()", source)
        self.assertIn("return ROCSV_DISTRIBUTED_BACKEND_HOST_FALLBACK", source)
        self.assertIn("rocsvGetDistributedBackend", bindings)
        self.assertIn("get_distributed_backend", bindings)
        self.assertIn("DistributedBackend", bindings)

    def test_expectation_and_sampling_use_rccl_allreduce(self):
        with open(_STATEVEC_SOURCE, "r", encoding="utf-8") as f:
            source = f.read()

        self.assertIn("rccl_allreduce_double_sum_inplace", source)
        self.assertIn("ncclGroupStart", source)
        self.assertIn("ncclAllReduce", source)
        self.assertIn("ncclGroupEnd", source)
        self.assertIn("distributed_expectation_rccl", source)
        self.assertIn("distributed_expectation_host_fallback", source)
        self.assertIn("distributed_expectation_with_fallback", source)
        self.assertIn("distributed_sample_rccl", source)
        self.assertIn("distributed_sample_host_fallback", source)
        self.assertIn("distributed_sample_with_fallback", source)
        self.assertGreaterEqual(source.count("return distributed_expectation_with_fallback"), 5)
        self.assertIn("return distributed_sample_with_fallback", source)
        self.assertIn("distributed_all_qubits_local(handle, targets)", source)
        self.assertIn("distributed_all_qubits_local(handle, measured)", source)

    def test_sparse_apply_supports_local_distributed_slices(self):
        with open(_STATEVEC_SOURCE, "r", encoding="utf-8") as f:
            source = f.read()

        self.assertIn("apply_sparse_matrix_distributed_local", source)
        self.assertIn("rocsvApplySparseMatrix", source)
        self.assertIn("distributed_all_qubits_local(handle, targets)", source)
        self.assertIn("handle->distributedSwapBuffers", source)
        self.assertIn("handle->distributedStreams", source)
        self.assertIn("apply_sparse_matrix_kernel", source)
        self.assertIn("apply_sparse_matrix_distributed_host_fallback", source)
        self.assertIn("apply_sparse_matrix_host_state_impl", source)
        self.assertIn("sparse_matrix_moments_distributed_host_fallback", source)
        self.assertIn("compute_sparse_matrix_moments_host_state", source)
        self.assertIn("expectation_matrix_distributed_host_fallback", source)
        self.assertIn("compute_expectation_matrix_host_state", source)


if __name__ == "__main__":
    unittest.main()
