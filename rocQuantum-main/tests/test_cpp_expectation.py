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
            "rocsvGetExpectationMatrixMoments",
            "rocsvGetExpectationMatrixMomentsBatch",
            "rocsvGetSparseMatrixMoments",
            "rocsvApplySparseMatrix",
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
            "rocsvGetExpectationMatrixMoments",
            "rocsvGetExpectationMatrixMomentsBatch",
            "rocsvGetSparseMatrixMoments",
            "rocsvApplySparseMatrix",
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
        legacy_bindings = self._read("python", "rocq", "bindings.cpp")

        self.assertIn("rocsvProbabilities", header)
        self.assertRegex(source, r"rocqStatus_t\s+rocsvProbabilities\s*\(")
        self.assertIn("accumulate_local_sample_probabilities", source)
        self.assertIn("accumulate_distributed_sample_probabilities_rccl", source)
        self.assertIn("compute_distributed_sample_probabilities", source)
        self.assertIn("reduce_expectation_matrix_kernel", source)
        self.assertIn("reduce_sparse_matrix_moments_kernel", source)
        self.assertIn("apply_sparse_matrix_kernel", source)
        self.assertIn("hipMemcpyDeviceToDevice", source)
        self.assertIn("std::vector<double> probabilities", simulator_header)
        self.assertIn("expectation_matrix", simulator_header)
        self.assertIn("apply_sparse_matrix", simulator_header)
        self.assertIn(".def(\"probabilities\"", bindings)
        self.assertIn(".def(\"expectation_matrix\"", bindings)
        self.assertIn(".def(\"apply_sparse_matrix\"", bindings)
        self.assertIn("get_expectation_matrix", legacy_bindings)
        self.assertIn("rocsvGetExpectationMatrix", legacy_bindings)
        self.assertIn("get_expectation_matrix_moments", legacy_bindings)
        self.assertIn("rocsvGetExpectationMatrixMoments", legacy_bindings)
        self.assertIn("get_expectation_matrix_moments_batch", legacy_bindings)
        self.assertIn("rocsvGetExpectationMatrixMomentsBatch", legacy_bindings)
        self.assertIn("get_sparse_matrix_moments", legacy_bindings)
        self.assertIn("rocsvGetSparseMatrixMoments", legacy_bindings)

    def test_root_binding_size_validation_uses_checked_helpers(self):
        bindings = self._read("bindings.cpp")

        self.assertIn("checked_power_of_two", bindings)
        self.assertIn("checked_bit_mask", bindings)
        self.assertIn("checked_square_size", bindings)
        self.assertIn("std::numeric_limits<std::size_t>::digits", bindings)
        self.assertIn("Statevector size does not match the simulator qubit count.", bindings)
        self.assertIn("checked_power_of_two(num_qubits, \"simulator qubit count\")", bindings)
        self.assertIn("checked_square_size(dim, \"dense observable matrix\")", bindings)
        self.assertIn("checked_square_size(dim, \"matrix operation\")", bindings)
        self.assertIn("checked_bit_mask(target, \"Pauli observable target\")", bindings)
        self.assertIn("checked_bit_mask(control, \"Controlled Pauli generator control\")", bindings)
        self.assertEqual(
            bindings.count("return std::size_t{1} << exponent;"),
            1,
            "Raw host-size shifts in the root binding should be isolated to checked_power_of_two().",
        )
        self.assertNotIn("std::size_t{1} << num_qubits", bindings)
        self.assertNotIn("std::size_t{1} << targets.size()", bindings)
        self.assertNotIn("std::size_t{1} << operation.wires.size()", bindings)
        self.assertNotIn("dim * dim", bindings)

    def test_quantum_simulator_size_validation_uses_checked_helpers(self):
        simulator = self._read("rocquantum", "src", "simulator.cpp")

        self.assertIn("checked_power_of_two", simulator)
        self.assertIn("checked_bit_mask", simulator)
        self.assertIn("checked_product", simulator)
        self.assertIn("checked_square_size", simulator)
        self.assertIn("std::numeric_limits<std::size_t>::digits", simulator)
        self.assertIn(
            "state_vec_size_ = checked_power_of_two(num_qubits_, \"QuantumSimulator qubit count\")",
            simulator,
        )
        self.assertIn("checked_square_size(matrix_dim, \"matrix payload\")", simulator)
        self.assertIn("checked_product(batch_size_, state_vec_size_, \"statevector batch\")", simulator)
        self.assertIn("checked_product(batch_size_, num_outcomes, \"probability batch\")", simulator)
        self.assertEqual(
            simulator.count("return std::size_t{1} << exponent;"),
            1,
            "Raw host-size shifts in QuantumSimulator should be isolated to checked_power_of_two().",
        )
        self.assertNotIn("std::size_t{1} << num_qubits_", simulator)
        self.assertNotIn("std::size_t{1} << targets.size()", simulator)
        self.assertNotIn("matrix_dim * matrix_dim", simulator)
        self.assertNotIn("batch_size_ * state_vec_size_", simulator)
        self.assertNotIn("batch_size_ * num_outcomes", simulator)


if __name__ == "__main__":
    unittest.main()
