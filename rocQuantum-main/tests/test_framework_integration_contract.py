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
_QUANTUM_SIMULATOR_CPP = os.path.join(
    _PROJECT_ROOT,
    "rocquantum",
    "src",
    "simulator.cpp",
)


class TestFrameworkIntegrationContract(unittest.TestCase):
    def test_shared_runtime_exposes_controlled_rotation_aliases(self):
        with open(_FRAMEWORK_RUNTIME, "r", encoding="utf-8") as f:
            source = f.read()

        self.assertIn("\"CRX\": \"CRX\"", source)
        self.assertIn("\"CRY\": \"CRY\"", source)
        self.assertIn("\"CRZ\": \"CRZ\"", source)
        self.assertIn("\"CCX\": \"MCX\"", source)
        self.assertIn("\"TOFFOLI\": \"MCX\"", source)
        self.assertIn("\"CSWAP\": \"CSWAP\"", source)

    def test_public_simulator_dispatches_native_multi_control_gates(self):
        with open(_QUANTUM_SIMULATOR_CPP, "r", encoding="utf-8") as f:
            source = f.read()

        self.assertIn("normalized == \"MCX\"", source)
        self.assertIn("normalized == \"CSWAP\"", source)
        self.assertIn("rocsvApplyMultiControlledX", source)
        self.assertIn("rocsvApplyCSWAP", source)

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
        self.assertIn("\"BasisState\", \"StatePrep\", \"Rot\"", source)
        self.assertIn("StatePrep is only supported as an initial state preparation", source)
        self.assertIn("_native_mcx_wire_indices", source)
        self.assertIn("control_values", source)
        self.assertIn("_apply_native_or_matrix(self._runtime, \"MCX\"", source)
        self.assertIn("def _apply_multirz", source)
        self.assertIn("_apply_multirz(self._runtime, wire_indices, theta)", source)
        self.assertIn("def _supports_native_phase_decomposition", source)
        self.assertIn("def _supports_native_parametric_decomposition", source)
        self.assertIn("def _supports_native_gate_decomposition", source)
        self.assertIn("def _apply_phase_shift", source)
        self.assertIn("def _apply_sx", source)
        self.assertIn("def _apply_controlled_phase_shift", source)
        self.assertIn("def _apply_controlled_phase_variant", source)
        self.assertIn("_apply_controlled_phase_shift(self._runtime, wire_indices, theta)", source)
        self.assertIn("def _apply_cy", source)
        self.assertIn("def _apply_ccz", source)
        self.assertIn("def _apply_ch", source)
        self.assertIn("def _apply_iswap", source)
        self.assertIn("def _apply_pswap", source)
        self.assertIn("def _apply_siswap", source)
        self.assertIn("def _apply_ecr", source)
        self.assertIn("def _apply_single_excitation", source)
        self.assertIn("gate_name == \"CH\"", source)
        self.assertIn("gate_name == \"CY\"", source)
        self.assertIn("gate_name == \"CCZ\"", source)
        self.assertIn("gate_name == \"ISWAP\"", source)
        self.assertIn("gate_name == \"PSWAP\"", source)
        self.assertIn("gate_name in {\"SISWAP\", \"SQISW\"}", source)
        self.assertIn("gate_name == \"ECR\"", source)
        self.assertIn("gate_name == \"SingleExcitation\"", source)
        self.assertIn("def _apply_isingxx", source)
        self.assertIn("def _apply_isingyy", source)
        self.assertIn("def _apply_isingxy", source)
        self.assertIn("gate_name == \"IsingXX\"", source)
        self.assertIn("gate_name == \"IsingYY\"", source)
        self.assertIn("gate_name == \"IsingXY\"", source)
        self.assertIn("gate_name == \"IsingZZ\"", source)
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
        self.assertIn("\"SparseHamiltonian\"", source)
        self.assertIn("observable.name == \"Hadamard\"", source)
        self.assertIn("_sparse_hamiltonian_moments", source)
        self.assertIn("sparse_matrix(wire_order=wire_order, format=\"csr\")", source)
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
        self.assertIn("MAX_AUTOMATIC_MATRIX_FALLBACK_QUBITS = 4", source)
        self.assertIn("_automatic_operation_matrix", source)
        self.assertIn("def _supports_native_phase_decomposition", source)
        self.assertIn("def _supports_native_parametric_decomposition", source)
        self.assertIn("def _apply_phase_gate", source)
        self.assertIn("def _apply_controlled_phase_gate", source)
        self.assertIn("def _apply_rxx_gate", source)
        self.assertIn("def _apply_ryy_gate", source)
        self.assertIn("def _apply_rzz_gate", source)
        self.assertIn("def _apply_sx_gate", source)
        self.assertIn("def _apply_u_gate", source)
        self.assertIn("def _apply_cy_gate", source)
        self.assertIn("def _apply_ccz_gate", source)
        self.assertIn("def _apply_ch_gate", source)
        self.assertIn("def _apply_dcx_gate", source)
        self.assertIn("def _apply_iswap_gate", source)
        self.assertIn("def _apply_ecr_gate", source)
        self.assertIn("Operator(op).data", source)
        self.assertIn("statevector=False", source)
        self.assertIn("Target(num_qubits=int(num_qubits))", source)
        self.assertIn("MCXGate(3)", source)
        self.assertIn("raw_samples = self._runtime.measure(qubits_to_measure, shots)", source)
        self.assertIn("formatted_counts = counts_from_memory(memory)", source)
        self.assertIn("return RocQuantumJob(self, job_id, result)", source)
        self.assertNotIn("{bin(k): v for k, v in counts.items()}", source)

    def test_qiskit_estimator_uses_native_expectation(self):
        with open(_QISKIT_ESTIMATOR, "r", encoding="utf-8") as f:
            source = f.read()

        self.assertIn("class RocQuantumEstimator", source)
        self.assertIn("EstimatorPub.coerce", source)
        self.assertIn("_canonical_observable_label", source)
        self.assertIn("_combine_observable_terms", source)
        self.assertIn("indices_by_parameter", source)
        self.assertIn("self._backend._apply_circuit", source)
        self.assertIn("estimate_pauli_observable", source)
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
