"""Source contracts for framework adapters using native sampling surfaces."""

from __future__ import annotations

import os
import unittest


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FRAMEWORK_RUNTIME = os.path.join(_PROJECT_ROOT, "rocquantum", "framework_runtime.py")
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
_QISKIT_SAMPLER = os.path.join(
    _PROJECT_ROOT,
    "integrations",
    "qiskit-rocquantum-provider",
    "qiskit_rocquantum_provider",
    "sampler.py",
)


class TestFrameworkIntegrationContract(unittest.TestCase):
    def test_shared_runtime_exposes_controlled_rotation_aliases(self):
        with open(_FRAMEWORK_RUNTIME, "r", encoding="utf-8") as f:
            source = f.read()

        self.assertIn("\"CRX\": \"CRX\"", source)
        self.assertIn("\"CRY\": \"CRY\"", source)
        self.assertIn("\"CRZ\": \"CRZ\"", source)

    def test_pennylane_sampling_prefers_native_measure(self):
        with open(_PENNYLANE_ADAPTER, "r", encoding="utf-8") as f:
            source = f.read()

        self.assertIn("measure = getattr(self.sim, \"measure\", None)", source)
        self.assertIn("NATIVE_PARAMETRIC_OPS = {\"RX\", \"RY\", \"RZ\", \"CRX\", \"CRY\", \"CRZ\"}", source)
        self.assertIn("\"PhaseShift\", \"ControlledPhaseShift\"", source)
        self.assertIn("\"CH\", \"CY\", \"CCZ\", \"CRot\"", source)
        self.assertIn("\"MultiControlledX\", \"MultiRZ\"", source)
        self.assertIn("\"IsingXX\", \"IsingYY\", \"IsingZZ\", \"IsingXY\"", source)
        self.assertIn("\"SingleExcitation\", \"SingleExcitationPlus\", \"SingleExcitationMinus\"", source)
        self.assertIn("def _shot_count", source)
        self.assertIn("shots = _shot_count(self.shots)", source)
        self.assertIn("raw_samples = self._runtime.measure(all_wires, shots)", source)
        self.assertIn("def _ensure_state", source)
        self.assertIn("self._state = self._runtime.statevector()", source)
        self.assertIn("return self._ensure_state()", source)
        self.assertIn("samples_to_binary_rows", source)
        self.assertIn("sample_rows_from_statevector", source, "legacy fallback should remain explicit")
        self.assertIn("def analytic_probability", source)
        self.assertIn("if not wires_to_trace: return all_probs", source)
        self.assertIn("def expval", source)
        self.assertIn("def var", source)
        self.assertIn("def execute", source)
        self.assertIn("_analytic_measurements_use_native_pauli", source)
        self.assertIn("_skip_diagonalizing_rotations", source)
        self.assertIn("_diagonalizing_rotations_applied", source)
        self.assertIn("\"Hadamard\"", source)
        self.assertIn("observable.name == \"Hadamard\"", source)
        self.assertIn("rotation_ops = list(rotations or [])", source)
        self.assertIn("list(operations) + rotation_ops", source)
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
        self.assertIn("MATRIX_FALLBACK_OPS", source)
        self.assertIn("statevector=False", source)
        self.assertIn("Target(num_qubits=int(num_qubits))", source)
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

    def test_qiskit_sampler_uses_native_sampling(self):
        with open(_QISKIT_SAMPLER, "r", encoding="utf-8") as f:
            source = f.read()

        self.assertIn("class RocQuantumSampler", source)
        self.assertIn("SamplerPub.coerce", source)
        self.assertIn("self._backend._apply_circuit", source)
        self.assertIn("self._backend._runtime.measure", source)
        self.assertIn("BitArray.from_bool_array", source)


if __name__ == "__main__":
    unittest.main()
