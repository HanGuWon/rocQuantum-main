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

    def test_gate_fusion_rejects_unsupported_queue_entries(self):
        with open(_GATE_FUSION_SOURCE, "r", encoding="utf-8") as f:
            source = f.read()

        self.assertIn("return ROCQ_STATUS_NOT_IMPLEMENTED;", source)
        self.assertNotIn("Fallback for non-fused gates", source)
        self.assertIn("if (op.params.empty())", source)
        self.assertIn("processed[i-1] = true;", source)
        self.assertIn("processed[i+1] = true;", source)
        self.assertIn("if (queue[i].name != \"CNOT\")", source)
        self.assertIn("queue[i + 1].name == \"CNOT\"", source)
        self.assertIn("get_gate_matrix_2x2(queue[i], single_matrix)", source)

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
