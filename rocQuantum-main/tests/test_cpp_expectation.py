"""
P3 C++ Expectation Contract Tests

Source-level contract checks for hipStateVec expectation APIs.
No ROCm toolchain required.

    python -m unittest tests.test_cpp_expectation -v
"""

import os
import sys
import unittest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


class TestHipStateVecExpectationContract(unittest.TestCase):
    def _read(self, *parts):
        path = os.path.join(_PROJECT_ROOT, *parts)
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def test_header_declares_expectation_entrypoints(self):
        header = self._read("rocquantum", "include", "rocquantum", "hipStateVec.h")
        symbols = [
            "rocsvGetExpectationValueSinglePauliZ",
            "rocsvGetExpectationValueSinglePauliX",
            "rocsvGetExpectationValueSinglePauliY",
            "rocsvGetExpectationValuePauliProductZ",
            "rocsvGetExpectationPauliString",
            "rocsvGetExpectationWorkspaceSize",
        ]
        for symbol in symbols:
            self.assertIn(symbol, header, f"Missing declaration for {symbol}")

    def test_source_defines_expectation_entrypoints(self):
        source = self._read("rocquantum", "src", "hipStateVec", "hipStateVec.cpp")
        symbols = [
            "rocsvGetExpectationValueSinglePauliZ",
            "rocsvGetExpectationValueSinglePauliX",
            "rocsvGetExpectationValueSinglePauliY",
            "rocsvGetExpectationValuePauliProductZ",
            "rocsvGetExpectationPauliString",
            "rocsvGetExpectationWorkspaceSize",
        ]
        for symbol in symbols:
            self.assertRegex(source, rf"rocqStatus_t\s+{symbol}\s*\(")

    def test_simulator_exposes_state_vector_output(self):
        src = self._read("rocquantum", "src", "simulator.cpp")
        self.assertTrue(
            "get_statevector" in src or "GetStateVector" in src,
            "simulator.cpp missing state vector output method",
        )


if __name__ == "__main__":
    unittest.main()
