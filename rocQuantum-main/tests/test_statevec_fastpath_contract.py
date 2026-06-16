"""Source contracts for hipStateVec fast-path and fusion behavior."""

from __future__ import annotations

import os
import re
import unittest


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STATEVEC_SOURCE = os.path.join(
    _PROJECT_ROOT,
    "rocquantum",
    "src",
    "hipStateVec",
    "hipStateVec.cpp",
)
_GATE_FUSION_SOURCE = os.path.join(
    _PROJECT_ROOT,
    "rocquantum",
    "src",
    "hipStateVec",
    "GateFusion.cpp",
)
_MULTI_GPU_GUIDE = os.path.join(
    _PROJECT_ROOT,
    "rocquantum",
    "src",
    "hipStateVec",
    "MULTI_GPU_GUIDE.md",
)
_HIPSTATEVEC_CMAKE = os.path.join(
    _PROJECT_ROOT,
    "rocquantum",
    "src",
    "hipStateVec",
    "CMakeLists.txt",
)
_STALE_MEASUREMENT_KERNELS = os.path.join(
    _PROJECT_ROOT,
    "rocquantum",
    "src",
    "hipStateVec",
    "measurement_kernels.hip",
)
_README = os.path.join(_PROJECT_ROOT, "README.md")


class TestStateVecFastPathContract(unittest.TestCase):
    def test_matrix_host_fallback_requires_explicit_opt_in(self):
        with open(_STATEVEC_SOURCE, "r", encoding="utf-8") as f:
            source = f.read()

        self.assertIn("ROCQ_ALLOW_HOST_MATRIX_FALLBACK", source)
        self.assertEqual(
            source.count("if (!allow_host_matrix_fallback())"),
            4,
            "Matrix apply/control and dense expectation host fallbacks should all require opt-in.",
        )
        self.assertEqual(source.count("return apply_matrix_host_impl"), 2)
        self.assertIn("return expectation_matrix_host_fallback", source)
        self.assertIn("return expectation_matrix_batch_host_fallback", source)
        self.assertRegex(
            source,
            re.compile(
                r"if \(!allow_host_matrix_fallback\(\)\) \{\s*"
                r"return ROCQ_STATUS_NOT_IMPLEMENTED;\s*"
                r"\}\s*std::vector<rocComplex> matrix_host;",
                re.MULTILINE,
            ),
        )

    def test_state_allocation_counts_reject_shift_and_batch_overflow(self):
        with open(_STATEVEC_SOURCE, "r", encoding="utf-8") as f:
            source = f.read()
        bindings_path = os.path.join(_PROJECT_ROOT, "python", "rocq", "bindings.cpp")
        with open(bindings_path, "r", encoding="utf-8") as f:
            bindings = f.read()

        self.assertIn("compute_state_element_count", source)
        self.assertIn("compute_state_byte_count", source)
        self.assertIn("normalized_batch_size", source)
        self.assertIn("device_malloc(handle, &allocated_ptr, total_bytes)", source)
        self.assertIn("hipMemsetAsync(target_state, 0, total_bytes", source)
        self.assertNotIn("handle->batchSize * num_elements_per_state", source)
        self.assertIn("checked_state_element_count", bindings)
        self.assertIn("checked_state_byte_count", bindings)
        self.assertIn("checked_power_of_two", bindings)
        self.assertIn("checked_square_size", bindings)
        self.assertIn("state batch is too large", bindings)
        self.assertIn("checked_power_of_two(targetQubits_vec.size(), \"get_expectation_matrix\")", bindings)
        self.assertIn("checked_power_of_two(numQubits, \"get_sparse_matrix_moments\")", bindings)
        self.assertIn("checked_power_of_two(measuredQubits_vec.size(), \"probabilities\")", bindings)
        self.assertEqual(
            bindings.count("return size_t{1} << exponent;"),
            1,
            "Raw shifts in the legacy Python binding should be isolated to checked_power_of_two().",
        )
        self.assertEqual(
            bindings.count("return dimension * dimension;"),
            1,
            "Raw matrix squaring in the legacy Python binding should be isolated to checked_square_size().",
        )
        self.assertNotIn("batch_size * (1ULL << numQubits)", bindings)
        self.assertNotIn("batch_size * (1ULL << num_qubits)", bindings)
        self.assertNotIn("size_t{1} << targetQubits_vec.size()", bindings)
        self.assertNotIn("size_t{1} << measuredQubits_vec.size()", bindings)
        self.assertNotIn("matrix_dim * matrix_dim", bindings)

    def test_gate_fusion_rejects_unsupported_queue_entries(self):
        with open(_GATE_FUSION_SOURCE, "r", encoding="utf-8") as f:
            source = f.read()

        self.assertIn("return ROCQ_STATUS_NOT_IMPLEMENTED;", source)
        self.assertNotIn("Fallback for non-fused gates", source)
        self.assertIn("if (op.params.empty())", source)
        self.assertIn("processed[i-1] = true;", source)
        self.assertIn("processed[i+1] = true;", source)
        self.assertIn("if (queue[i].name != \"CNOT\")", source)
        self.assertIn("fuseAndApplySingleQubitGates(queue)", source)
        self.assertIn("matmul_2x2(updated, op_matrix, fused_matrix)", source)
        self.assertIn("return rocsvApplyFusedSingleQubitMatrix", source)
        self.assertIn("queue[i + 1].name == \"CNOT\"", source)
        self.assertIn("get_gate_matrix_2x2(queue[i], single_matrix)", source)

    def test_stale_single_thread_measurement_kernels_are_not_built(self):
        with open(_STATEVEC_SOURCE, "r", encoding="utf-8") as f:
            source = f.read()
        with open(_HIPSTATEVEC_CMAKE, "r", encoding="utf-8") as f:
            cmake = f.read()

        self.assertFalse(os.path.exists(_STALE_MEASUREMENT_KERNELS))
        self.assertNotIn("measurement_kernels.hip", cmake)
        self.assertNotIn("calculate_prob0_kernel", source)
        self.assertNotIn("sum_sq_magnitudes_kernel", source)
        self.assertIn("reduce_measure_prob0_kernel", source)
        self.assertIn("collapse_and_renorm_measure_kernel", source)
        self.assertIn("renormalize_state_kernel", source)

    def test_docs_describe_active_measurement_kernels_without_stale_scaffold(self):
        with open(_README, "r", encoding="utf-8") as f:
            readme = f.read()

        self.assertIn("active parallel measurement/probability kernels", readme)
        self.assertIn("stale single-thread measurement scaffolding is not built", readme)

    def test_distributed_host_fallback_is_explicit_mode(self):
        with open(_STATEVEC_SOURCE, "r", encoding="utf-8") as f:
            source = f.read()

        self.assertIn("ROCQ_DISTRIBUTED_FALLBACK_MODE", source)
        self.assertIn("ROCQ_ENABLE_DISTRIBUTED_HOST_FALLBACK", source)
        self.assertIn("apply_matrix_distributed_host_fallback", source)
        self.assertIn("apply_sparse_matrix_distributed_host_fallback", source)
        self.assertIn("apply_sparse_matrix_host_state_impl", source)
        self.assertIn("expectation_matrix_host_fallback", source)
        self.assertIn("expectation_matrix_batch_host_fallback", source)
        self.assertIn("expectation_matrix_distributed_host_fallback", source)
        self.assertIn("sparse_matrix_moments_distributed_host_fallback", source)
        self.assertGreaterEqual(
            source.count("return apply_matrix_distributed_host_fallback"),
            3,
            "Distributed matrix fallbacks should keep an explicit host path for unsupported layouts.",
        )
        self.assertIn("return expectation_matrix_distributed_host_fallback", source)
        self.assertIn("return apply_sparse_matrix_distributed_host_fallback", source)
        self.assertIn("return sparse_matrix_moments_distributed_host_fallback", source)
        self.assertGreaterEqual(source.count("distributed_host_fallback_enabled()"), 3)

    def test_nonlocal_distributed_named_gates_localize_with_swaps(self):
        with open(_STATEVEC_SOURCE, "r", encoding="utf-8") as f:
            source = f.read()

        self.assertIn("launch_single_qubit_matrix_distributed_localized", source)
        self.assertIn("launch_controlled_single_qubit_matrix_distributed_localized", source)
        self.assertIn("restore_distributed_qubit_swaps", source)
        self.assertIn("choose_distributed_local_slot", source)
        self.assertIn("rocsvSwapIndexBits(handle, targetQubit, local_target)", source)
        self.assertIn("rocsvSwapIndexBits(handle, swap.first, swap.second)", source)
        self.assertIn("restore_distributed_qubit_swaps(handle, applied_swaps)", source)
        self.assertIn("return launch_single_qubit_matrix_distributed_localized", source)
        self.assertIn("return launch_controlled_single_qubit_matrix_distributed_localized", source)
        self.assertIn("return launch_controlled_single_qubit_matrix(handle", source)

    def test_common_multi_control_distributed_gates_localize_with_swaps(self):
        with open(_STATEVEC_SOURCE, "r", encoding="utf-8") as f:
            source = f.read()

        self.assertIn("localize_distributed_qubits_for_operation", source)
        self.assertIn("std::vector<unsigned> touched = controls;", source)
        self.assertIn("touched.push_back(targetQubit);", source)
        self.assertIn("apply_multi_controlled_x_kernel", source)
        self.assertIn("std::vector<unsigned> touched = {controlQubit, targetQubit1, targetQubit2};", source)
        self.assertIn("apply_CSWAP_kernel", source)
        self.assertIn("handle->numLocalQubitsPerGpu", source)
        mcx_block = source.split("rocqStatus_t rocsvApplyMultiControlledX", 1)[1].split(
            "rocqStatus_t rocsvApplyCSWAP", 1
        )[0]
        cswap_block = source.split("rocqStatus_t rocsvApplyCSWAP", 1)[1].split(
            "// --- State-vector readback helpers", 1
        )[0]
        self.assertNotRegex(
            mcx_block,
            re.compile(
                r"if \(uses_distributed_state\(handle, d_state\)\) \{\s*"
                r"return ROCQ_STATUS_NOT_IMPLEMENTED;\s*"
                r"\}",
                re.MULTILINE,
            ),
        )
        self.assertNotRegex(
            cswap_block,
            re.compile(
                r"if \(uses_distributed_state\(handle, d_state\)\) \{\s*"
                r"return ROCQ_STATUS_NOT_IMPLEMENTED;\s*"
                r"\}",
                re.MULTILINE,
            ),
        )

    def test_multi_controlled_single_target_matrix_has_native_fast_path(self):
        with open(_STATEVEC_SOURCE, "r", encoding="utf-8") as f:
            source = f.read()

        self.assertIn("apply_multi_controlled_single_qubit_matrix_kernel", source)
        self.assertIn("launch_multi_controlled_single_qubit_matrix", source)
        self.assertIn("controlQubits.size() > 63", source)
        controlled_block = source.split("rocqStatus_t rocsvApplyControlledMatrix", 1)[1].split(
            "rocqStatus_t rocsvApplyMatrixAndMeasure", 1
        )[0]
        self.assertIn("numControls > 1 && numTargets == 1", controlled_block)
        self.assertIn("return launch_multi_controlled_single_qubit_matrix", controlled_block)
        self.assertLess(
            controlled_block.find("numControls > 1 && numTargets == 1"),
            controlled_block.find("if (!allow_host_matrix_fallback())"),
            "Multi-control single-target matrices should try the native fast path before host fallback.",
        )
        self.assertIn("localize_distributed_qubits_for_operation(handle, touched", source)
        self.assertIn("apply_multi_controlled_single_qubit_matrix_kernel", source)

    def test_dense_matrix_moments_have_fused_reduction(self):
        with open(_STATEVEC_SOURCE, "r", encoding="utf-8") as f:
            source = f.read()

        self.assertIn("reduce_expectation_matrix_moments_kernel", source)
        self.assertIn("rocsvGetExpectationMatrixMoments", source)
        self.assertIn("rocsvGetExpectationMatrixMomentsBatch", source)
        self.assertIn("4 * threads_per_block * sizeof(double)", source)
        single_block = source.split("rocqStatus_t rocsvGetExpectationMatrixMoments", 1)[1].split(
            "rocqStatus_t rocsvGetExpectationMatrixMomentsBatch", 1
        )[0]
        batch_block = source.split("rocqStatus_t rocsvGetExpectationMatrixMomentsBatch", 1)[1].split(
            "rocqStatus_t rocsvGetSparseMatrixMoments", 1
        )[0]
        self.assertIn("reduce_expectation_matrix_moments_kernel", single_block)
        self.assertIn("expectation_matrix_host_fallback", single_block)
        self.assertIn("reduce_expectation_matrix_moments_kernel", batch_block)
        self.assertIn("expectation_matrix_batch_host_fallback", batch_block)
        self.assertIn("dim3(blocks, static_cast<unsigned>(batch_size))", batch_block)

    def test_nonlocal_distributed_generic_matrix_localizes_with_swaps(self):
        with open(_STATEVEC_SOURCE, "r", encoding="utf-8") as f:
            source = f.read()

        self.assertIn("apply_matrix_distributed_local_targets", source)
        self.assertIn("localize_distributed_qubits_for_operation(", source)
        self.assertIn("localized_targets", source)
        self.assertIn("restore_distributed_qubit_swaps(handle, swaps)", source)
        self.assertIn("return apply_matrix_distributed_local_targets(handle, numQubits, targets, matrix_host)", source)
        matrix_block = source.split("rocqStatus_t rocsvApplyMatrix", 1)[1].split(
            "rocqStatus_t rocsvApplySWAP", 1
        )[0]
        self.assertIn("status = apply_matrix_distributed_local_targets", matrix_block)
        self.assertIn("return apply_matrix_distributed_host_fallback", matrix_block)
        self.assertLess(
            matrix_block.find("localize_distributed_qubits_for_operation("),
            matrix_block.find("return apply_matrix_distributed_host_fallback"),
            "Distributed generic matrix paths should attempt swap-localization before host fallback.",
        )

    def test_multi_gpu_guide_matches_distributed_contract(self):
        with open(_MULTI_GPU_GUIDE, "r", encoding="utf-8") as f:
            guide = f.read()
        with open(_README, "r", encoding="utf-8") as f:
            readme = f.read()

        self.assertIn("## Behavior Matrix", guide)
        self.assertIn("## Runtime Switches", guide)
        self.assertIn("## Known Limitations", guide)
        for expected in [
            "ROCQ_DISTRIBUTED_FALLBACK_MODE",
            "ROCQ_ENABLE_DISTRIBUTED_HOST_FALLBACK",
            "ROCQ_DISTRIBUTED_COMM",
            "ROCQ_REQUIRE_RCCL",
            "ROCQ_DISABLE_RCCL",
            "distributed_expectation_rccl",
            "distributed_sample_rccl",
            "ROCQ_STATUS_NOT_IMPLEMENTED",
            "Correctness/debug only; no performance parity claim",
            "multi_gpu=True",
        ]:
            self.assertIn(expected, guide)
        self.assertIn("MULTI_GPU_GUIDE.md", readme)
        self.assertNotIn("complete distributed sampling story", guide)
        self.assertNotIn("production-grade multi-gpu", guide.lower())


if __name__ == "__main__":
    unittest.main()
