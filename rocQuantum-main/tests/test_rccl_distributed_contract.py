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
_SWAP_KERNELS_SOURCE = os.path.join(
    _PROJECT_ROOT,
    "rocquantum",
    "src",
    "hipStateVec",
    "swap_kernels.hip",
)
_BINDINGS_SOURCE = os.path.join(_PROJECT_ROOT, "python", "rocq", "bindings.cpp")
_LEGACY_API = os.path.join(_PROJECT_ROOT, "python", "rocq", "api.py")
_MULTI_GPU_GUIDE = os.path.join(
    _PROJECT_ROOT,
    "rocquantum",
    "src",
    "hipStateVec",
    "MULTI_GPU_GUIDE.md",
)


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
        self.assertIn("distributed_expectation_matrix_rccl", source)
        self.assertIn("distributed_sparse_matrix_moments_rccl", source)
        self.assertIn("build_local_csr_slice", source)
        self.assertIn("reduce_complex_block_sums_to_double_pair_kernel", source)
        self.assertIn("reduce_sparse_matrix_moments_kernel", source)
        self.assertIn("distributed_expectation_host_fallback", source)
        self.assertIn("distributed_expectation_with_fallback", source)
        self.assertIn("distributed_sample_rccl", source)
        self.assertIn("distributed_sample_host_fallback", source)
        self.assertIn("distributed_sample_with_fallback", source)
        self.assertGreaterEqual(source.count("return distributed_expectation_with_fallback"), 5)
        self.assertIn("return distributed_sample_with_fallback", source)
        self.assertIn("distributed_all_qubits_local(handle, targets)", source)
        self.assertIn("distributed_all_qubits_local(handle, measured)", source)
        self.assertIn("rccl_allreduce_double_sum_inplace(handle, rank_pairs, 2)", source)
        self.assertIn("rccl_allreduce_double_sum_inplace(handle, rank_mean_pairs, 2)", source)
        self.assertIn("rccl_allreduce_double_sum_inplace(handle, rank_second_pairs, 2)", source)

        matrix_block = source.split("rocqStatus_t rocsvGetExpectationMatrix", 1)[1].split(
            "rocqStatus_t rocsvGetExpectationMatrixBatch", 1
        )[0]
        self.assertIn("status = distributed_expectation_matrix_rccl", matrix_block)
        self.assertIn("if (status != ROCQ_STATUS_NOT_IMPLEMENTED)", matrix_block)
        self.assertIn("return expectation_matrix_distributed_host_fallback", matrix_block)

        sparse_block = source.split("rocqStatus_t rocsvGetSparseMatrixMoments(", 1)[1].split(
            "rocqStatus_t rocsvGetSparseMatrixMomentsBatch", 1
        )[0]
        self.assertIn("rocqStatus_t rccl_status = distributed_sparse_matrix_moments_rccl", sparse_block)
        self.assertIn("if (rccl_status != ROCQ_STATUS_NOT_IMPLEMENTED)", sparse_block)
        self.assertIn("return sparse_matrix_moments_distributed_host_fallback", sparse_block)

    def test_nonlocal_swap_remap_uses_rccl_send_recv_before_host_fallback(self):
        with open(_STATEVEC_SOURCE, "r", encoding="utf-8") as f:
            source = f.read()
        with open(_SWAP_KERNELS_SOURCE, "r", encoding="utf-8") as f:
            kernels = f.read()

        self.assertIn("distributed_swap_bits_rccl_remap", source)
        self.assertIn("distributed_swap_bits_rccl_rank_remap", source)
        self.assertIn("distributed_swap_bits_rccl_local_global", source)
        self.assertIn("ncclSend", source)
        self.assertIn("ncclRecv", source)
        self.assertIn("pack_local_global_swap_kernel", source)
        self.assertIn("unpack_local_global_swap_kernel", source)
        self.assertIn("static_cast<size_t>(rank) ^ rank_mask", source)
        self.assertIn("handle->distributedSwapBuffers[idx] + packed_elements", source)
        self.assertIn("pack_local_global_swap_kernel", kernels)
        self.assertIn("unpack_local_global_swap_kernel", kernels)
        self.assertIn("rocquant_kernel_insert_bit", kernels)

        swap_block = source.split("rocqStatus_t rocsvSwapIndexBits", 1)[1].split(
            "// --- Single-qubit named gates", 1
        )[0]
        self.assertIn("rocqStatus_t rccl_status = distributed_swap_bits_rccl_remap", swap_block)
        self.assertIn("if (!distributed_host_fallback_enabled())", swap_block)
        self.assertIn("return distributed_swap_bits_host_remap(handle, qubit_idx1, qubit_idx2)", swap_block)

    def test_sampling_and_expectation_host_fallbacks_are_explicit(self):
        with open(_STATEVEC_SOURCE, "r", encoding="utf-8") as f:
            source = f.read()

        self.assertIn("distributed_expectation_host_fallback", source)
        self.assertIn("host_pauli_expectation", source)
        self.assertIn("return host_pauli_expectation(host_full, numQubits, targets, pauli, result)", source)
        self.assertIn("distributed_sample_host_fallback", source)
        self.assertIn("host_sample_state", source)
        self.assertIn("return host_sample_state(handle, host_full, numQubits, measured, numShots, h_results)", source)
        self.assertGreaterEqual(
            source.count("if (!distributed_host_fallback_enabled())"),
            6,
            "Every distributed host fallback should require an explicit fallback switch.",
        )
        self.assertIn("rocqStatus_t status = distributed_expectation_rccl", source)
        self.assertIn("return distributed_expectation_host_fallback", source)
        self.assertIn("rocqStatus_t status = distributed_sample_rccl", source)
        self.assertIn("return distributed_sample_host_fallback", source)
        self.assertGreaterEqual(source.count("gather_distributed_state_to_host(handle, &host_full)"), 6)

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

    def test_multi_node_requests_are_explicitly_unsupported(self):
        with open(_STATEVEC_HEADER, "r", encoding="utf-8") as f:
            header = f.read()
        with open(_STATEVEC_SOURCE, "r", encoding="utf-8") as f:
            source = f.read()
        with open(_BINDINGS_SOURCE, "r", encoding="utf-8") as f:
            bindings = f.read()
        with open(_LEGACY_API, "r", encoding="utf-8") as f:
            legacy_api = f.read()
        with open(_MULTI_GPU_GUIDE, "r", encoding="utf-8") as f:
            guide = f.read()

        self.assertIn("rocsvAllocateMultiNodeDistributedState", header)
        self.assertIn("rocsvAllocateMultiNodeDistributedState", source)
        multi_node_block = source.split("rocsvAllocateMultiNodeDistributedState", 1)[1].split(
            "rocsvAllocateDistributedState", 1
        )[0]
        self.assertIn("nodeCount < 2", multi_node_block)
        self.assertIn("return ROCQ_STATUS_NOT_IMPLEMENTED", multi_node_block)
        self.assertIn("allocate_multi_node_distributed_state", bindings)
        self.assertIn("_MULTI_NODE_UNSUPPORTED_NOTE", legacy_api)
        self.assertIn("multi_node: bool = False", legacy_api)
        self.assertIn("node_count: int = 1", legacy_api)
        self.assertIn("raise NotImplementedError(_MULTI_NODE_UNSUPPORTED_NOTE)", legacy_api)
        self.assertIn("Multi-node execution", guide)
        self.assertIn("Not implemented", guide)


if __name__ == "__main__":
    unittest.main()
