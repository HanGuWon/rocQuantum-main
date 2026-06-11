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
            7,
            "Distributed single/control/matrix fallbacks should share the same explicit host path.",
        )
        self.assertIn("return expectation_matrix_distributed_host_fallback", source)
        self.assertIn("return apply_sparse_matrix_distributed_host_fallback", source)
        self.assertIn("return sparse_matrix_moments_distributed_host_fallback", source)
        self.assertGreaterEqual(source.count("distributed_host_fallback_enabled()"), 3)


if __name__ == "__main__":
    unittest.main()
