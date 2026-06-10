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
            "rocsvGetExpectationMatrix",
            "rocsvGetSparseMatrixMoments",
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
            "rocsvGetExpectationMatrix",
            "rocsvGetSparseMatrixMoments",
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

    def test_state_vector_probability_entrypoint_is_public(self):
        header = self._read("rocquantum", "include", "rocquantum", "hipStateVec.h")
        source = self._read("rocquantum", "src", "hipStateVec", "hipStateVec.cpp")
        simulator_header = self._read("include", "rocquantum", "QuantumSimulator.h")
        bindings = self._read("bindings.cpp")

        self.assertIn("rocsvProbabilities", header)
        self.assertRegex(source, r"rocqStatus_t\s+rocsvProbabilities\s*\(")
        self.assertIn("accumulate_local_sample_probabilities", source)
        self.assertIn("accumulate_distributed_sample_probabilities_rccl", source)
        self.assertIn("compute_distributed_sample_probabilities", source)
        self.assertIn("reduce_expectation_matrix_kernel", source)
        self.assertIn("reduce_sparse_matrix_moments_kernel", source)
        self.assertIn("std::vector<double> probabilities", simulator_header)
        self.assertIn("expectation_matrix", simulator_header)
        self.assertIn(".def(\"probabilities\"", bindings)
        self.assertIn(".def(\"expectation_matrix\"", bindings)


if __name__ == "__main__":
    unittest.main()
