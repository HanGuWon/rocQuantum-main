"""Source contracts for framework adapters using native sampling surfaces."""

from __future__ import annotations

import os
import unittest


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PENNYLANE_ADAPTER = os.path.join(
    _PROJECT_ROOT,
    "integrations",
    "pennylane-rocq",
    "pennylane_rocq",
    "rocq_device.py",
)
_CIRQ_ADAPTER = os.path.join(
    _PROJECT_ROOT,
    "integrations",
    "cirq-rocm",
    "cirq_rocm",
    "roc_quantum_simulator.py",
)
_QISKIT_BACKEND = os.path.join(
    _PROJECT_ROOT,
    "integrations",
    "qiskit-rocquantum-provider",
    "qiskit_rocquantum_provider",
    "backend.py",
)
_QISKIT_ESTIMATOR = os.path.join(
    _PROJECT_ROOT,
    "integrations",
    "qiskit-rocquantum-provider",
    "qiskit_rocquantum_provider",
    "estimator.py",
)


class TestFrameworkIntegrationContract(unittest.TestCase):
    def test_pennylane_sampling_prefers_native_measure(self):
        with open(_PENNYLANE_ADAPTER, "r", encoding="utf-8") as f:
            source = f.read()

        self.assertIn("measure = getattr(self.sim, \"measure\", None)", source)
        self.assertIn("raw_samples = self._runtime.measure(all_wires, int(self.shots))", source)
        self.assertIn("samples_to_binary_rows", source)
        self.assertIn("sample_rows_from_statevector", source, "legacy fallback should remain explicit")
        self.assertIn("def analytic_probability", source)
        self.assertIn("if not wires_to_trace: return all_probs", source)
        self.assertIn("def expval", source)
        self.assertIn("def var", source)
        self.assertIn("_pauli_square_terms", source)
        self.assertIn("expectation_pauli_string", source)

    def test_cirq_sampling_prefers_native_measure(self):
        with open(_CIRQ_ADAPTER, "r", encoding="utf-8") as f:
            source = f.read()

        self.assertIn("def _execute_circuit", source)
        self.assertIn("measure = getattr(sim, \"measure\", None)", source)
        self.assertIn("raw_samples = measure(indices, repetitions)", source)
        self.assertIn("_samples_to_bits", source)
        self.assertIn("np.random.choice", source, "legacy fallback should remain explicit")

    def test_qiskit_counts_are_fixed_width_bitstrings(self):
        with open(_QISKIT_BACKEND, "r", encoding="utf-8") as f:
            source = f.read()

        self.assertIn("qiskit_memory_from_samples", source)
        self.assertIn("raw_samples = self._runtime.measure(qubits_to_measure, shots)", source)
        self.assertIn("formatted_counts = counts_from_memory(memory)", source)
        self.assertIn("return RocQuantumJob(self, job_id, result)", source)
        self.assertNotIn("{bin(k): v for k, v in counts.items()}", source)

    def test_qiskit_estimator_uses_native_expectation(self):
        with open(_QISKIT_ESTIMATOR, "r", encoding="utf-8") as f:
            source = f.read()

        self.assertIn("class RocQuantumEstimator", source)
        self.assertIn("EstimatorPub.coerce", source)
        self.assertIn("estimate_expectation", source)
        self.assertIn("shots\": 0", source)


if __name__ == "__main__":
    unittest.main()
