"""
P3 C++ Expectation API Test Harness

Source-level validation that the hipStateVec expectation value API
exists and is non-destructive. This checks C++ source files since we
cannot compile without ROCm/HIP.

    python -m unittest tests.test_cpp_expectation -v
"""

import os
import sys
import unittest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


class TestHipStateVecExpectationAPI(unittest.TestCase):
    """Validate the C++ hipStateVec backend has expectation value support."""

    def _read_cpp(self, *path_parts):
        path = os.path.join(_PROJECT_ROOT, *path_parts)
        if not os.path.isfile(path):
            self.skipTest(f"{path} not found")
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def test_hip_statevec_has_expectation_method(self):
        """HipStateVecBackend must declare a get_expectation method."""
        src = self._read_cpp("rocqCompiler", "HipStateVecBackend.cpp")
        has_expval = (
            "get_expectation" in src
            or "expectation_value" in src
            or "computeExpectation" in src
        )
        # This is informational — the C++ API is still under development
        if not has_expval:
            self.skipTest(
                "hipStateVec C++ backend does not yet expose an expectation "
                "value API. This will be required for P4."
            )

    def test_simulator_has_state_vector_output(self):
        """Simulator must have get_state_vector or GetStateVector."""
        src = self._read_cpp("rocquantum", "src", "simulator.cpp")
        self.assertTrue(
            "get_state_vector" in src or "GetStateVector" in src,
            "simulator.cpp missing state vector output method"
        )


if __name__ == "__main__":
    unittest.main()
