"""Source contracts for hipTensorNet optimizer, dtype, and slicing behavior."""

from __future__ import annotations

import os
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
_BINDINGS_SOURCE = os.path.join(_PROJECT_ROOT, "python", "rocq", "bindings.cpp")


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
        self.assertIn("ROC_TENSORNET_COMPILED_COMPLEX_DTYPE", header)
        self.assertIn("rocTensorNetworkGetCapabilities", header)
        self.assertIn("rocTensorNetworkGetCapabilities", source)

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

    def test_optimizer_and_memory_limit_are_not_silent(self):
        with open(_TENSORNET_SOURCE, "r", encoding="utf-8") as f:
            source = f.read()

        self.assertIn("validate_optimizer_config", source)
        self.assertIn("pathfinder_algorithm_available", source)
        self.assertIn("ROCTN_PATHFINDER_ALGO_KAHYPAR", source)
        self.assertIn("ROCTN_PATHFINDER_ALGO_METIS", source)
        self.assertIn("selection_cost_for_algorithm", source)
        self.assertIn("memory_limit_bytes", source)
        self.assertIn("num_slices", source)
        self.assertIn("supports_memory_limit_planning = 1", source)
        self.assertIn("supports_runtime_slicing = 0", source)

    def test_python_binding_passes_optimizer_algorithm(self):
        with open(_BINDINGS_SOURCE, "r", encoding="utf-8") as f:
            source = f.read()

        self.assertIn("pathfinder_algorithm", source)
        self.assertIn("ROCTN_PATHFINDER_ALGO_GREEDY", source)
        self.assertIn("ROCTN_PATHFINDER_ALGO_KAHYPAR", source)
        self.assertIn("ROCTN_PATHFINDER_ALGO_METIS", source)
        self.assertIn("get_tensornet_capabilities", source)

    def test_python_binding_reports_actionable_tensornet_status_errors(self):
        with open(_BINDINGS_SOURCE, "r", encoding="utf-8") as f:
            source = f.read()

        self.assertIn("tensornet_status_message", source)
        self.assertIn("tensornet_contract_status_message", source)
        self.assertIn("get_tensornet_capabilities()", source)
        self.assertIn("pathfinder_algorithm=", source)
        self.assertIn("METIS/KAHYPAR pathfinders require", source)
        self.assertIn("Runtime slicing is not implemented", source)
        self.assertIn("num_slices must be non-negative", source)
        self.assertNotIn(
            'rocTensorNetworkContract failed: " + std::to_string(status)',
            source,
        )

    def test_python_binding_warns_when_slicing_knobs_are_planning_only(self):
        with open(_BINDINGS_SOURCE, "r", encoding="utf-8") as f:
            source = f.read()

        self.assertIn("warn_tensornet_planning_only_slicing", source)
        self.assertIn("PyErr_WarnEx(PyExc_RuntimeWarning", source)
        self.assertIn("contraction planning cost only", source)
        self.assertIn("supports_runtime_slicing", source)
        self.assertIn("warn_tensornet_planning_only_slicing(config)", source)


if __name__ == "__main__":
    unittest.main()
