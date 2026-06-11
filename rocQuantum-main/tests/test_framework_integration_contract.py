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
_QUANTUM_SIMULATOR_HEADER = os.path.join(
    _PROJECT_ROOT,
    "include",
    "rocquantum",
    "QuantumSimulator.h",
)
_HIPSTATEVEC_HEADER = os.path.join(
    _PROJECT_ROOT,
    "rocquantum",
    "include",
    "rocquantum",
    "hipStateVec.h",
)
_HIPSTATEVEC_SOURCE = os.path.join(
    _PROJECT_ROOT,
    "rocquantum",
    "src",
    "hipStateVec",
    "hipStateVec.cpp",
)
_HIPSTATEVEC_SINGLE_QUBIT_KERNELS = os.path.join(
    _PROJECT_ROOT,
    "rocquantum",
    "src",
    "hipStateVec",
    "single_qubit_kernels.hip",
)
_BINDINGS_CPP = os.path.join(_PROJECT_ROOT, "bindings.cpp")
_LOW_LEVEL_BINDINGS_CPP = os.path.join(_PROJECT_ROOT, "python", "rocq", "bindings.cpp")


class TestFrameworkIntegrationContract(unittest.TestCase):
    def test_shared_runtime_exposes_controlled_rotation_aliases(self):
        with open(_FRAMEWORK_RUNTIME, "r", encoding="utf-8") as f:
            source = f.read()

        self.assertIn("\"CRX\": \"CRX\"", source)
        self.assertIn("\"CRY\": \"CRY\"", source)
        self.assertIn("\"CRZ\": \"CRZ\"", source)
        self.assertIn("\"CP\": \"CP\"", source)
        self.assertIn("\"P\": \"P\"", source)
        self.assertIn("\"PHASE\": \"P\"", source)
        self.assertIn("\"TDG\": \"TDG\"", source)
        self.assertIn("\"CCX\": \"MCX\"", source)
        self.assertIn("\"TOFFOLI\": \"MCX\"", source)
        self.assertIn("\"CSWAP\": \"CSWAP\"", source)
        self.assertIn("def statevector_to_little_endian_wires", source)
        self.assertIn("def probabilities_from_statevector", source)
        self.assertIn("def probabilities(self, qubits", source)
        self.assertIn("def probabilities_batch(self, qubits", source)
        self.assertIn("def measure_batch(self, qubits", source)
        self.assertIn("def sample_indices_from_probabilities", source)
        self.assertIn("def sample_indices_batch_from_probabilities", source)
        self.assertIn("def expectation_pauli_string_batch", source)
        self.assertIn("def apply_operation_batch", source)
        self.assertIn("def expectation_matrix(self, matrix", source)
        self.assertIn("def set_statevector", source)

    def test_public_simulator_dispatches_native_multi_control_gates(self):
        with open(_QUANTUM_SIMULATOR_CPP, "r", encoding="utf-8") as f:
            source = f.read()

        self.assertIn("normalized == \"MCX\"", source)
        self.assertIn("normalized == \"CSWAP\"", source)
        self.assertIn("rocsvApplyMultiControlledX", source)
        self.assertIn("rocsvApplyCSWAP", source)

    def test_public_simulator_exposes_native_phase_gates(self):
        with open(_QUANTUM_SIMULATOR_CPP, "r", encoding="utf-8") as f:
            simulator = f.read()
        with open(_HIPSTATEVEC_HEADER, "r", encoding="utf-8") as f:
            hip_header = f.read()
        with open(_HIPSTATEVEC_SOURCE, "r", encoding="utf-8") as f:
            hip_source = f.read()
        with open(_LOW_LEVEL_BINDINGS_CPP, "r", encoding="utf-8") as f:
            low_level_bindings = f.read()
        with open(_QISKIT_BACKEND, "r", encoding="utf-8") as f:
            qiskit_backend = f.read()
        with open(_PENNYLANE_ADAPTER, "r", encoding="utf-8") as f:
            pennylane_adapter = f.read()

        self.assertIn("normalized == \"P\"", simulator)
        self.assertIn("normalized == \"CP\"", simulator)
        self.assertIn("rocsvApplyP", simulator)
        self.assertIn("rocsvApplyCP", simulator)
        self.assertIn("rocqStatus_t rocsvApplyP", hip_header)
        self.assertIn("rocqStatus_t rocsvApplyCP", hip_header)
        self.assertIn("rocqStatus_t rocsvApplyP", hip_source)
        self.assertIn("rocqStatus_t rocsvApplyCP", hip_source)
        self.assertIn("make_complex(std::cos(theta), std::sin(theta))", hip_source)
        self.assertIn("m.def(\"apply_p\"", low_level_bindings)
        self.assertIn("m.def(\"apply_cp\"", low_level_bindings)
        self.assertIn("self._runtime.apply_operation(\"p\"", qiskit_backend)
        self.assertIn("self._runtime.apply_operation(\"cp\"", qiskit_backend)
        self.assertIn("runtime.apply_operation(\"P\"", pennylane_adapter)
        self.assertIn("runtime.apply_operation(\"CP\"", pennylane_adapter)

    def test_public_simulator_exposes_statevector_upload(self):
        with open(_QUANTUM_SIMULATOR_HEADER, "r", encoding="utf-8") as f:
            header = f.read()
        with open(_QUANTUM_SIMULATOR_CPP, "r", encoding="utf-8") as f:
            implementation = f.read()
        with open(_BINDINGS_CPP, "r", encoding="utf-8") as f:
            bindings = f.read()

        self.assertIn("void set_statevector", header)
        self.assertIn("void set_statevectors", header)
        self.assertIn("std::vector<std::complex<double>> get_statevectors", header)
        self.assertIn("std::size_t batch_size", header)
        self.assertIn("QuantumSimulator::set_statevector", implementation)
        self.assertIn("QuantumSimulator::set_statevectors", implementation)
        self.assertIn("QuantumSimulator::get_statevector(std::size_t batch_index)", implementation)
        self.assertIn("QuantumSimulator::get_statevectors", implementation)
        self.assertIn("rocsvAllocateState(sim_handle_, num_qubits_, &device_state_vector_, batch_size_)", implementation)
        self.assertIn("rocsvGetStateVectorSlice", implementation)
        self.assertIn("rocsvGetStateVectorFull", implementation)
        self.assertIn("hipMemcpyHostToDevice", implementation)
        self.assertIn(".def(\"set_statevector\"", bindings)
        self.assertIn(".def(\"set_statevectors\"", bindings)
        self.assertIn(".def(\"get_statevectors\"", bindings)
        self.assertIn(".def(\"batch_size\"", bindings)

    def test_public_simulator_exposes_batched_parametric_gates(self):
        with open(_QUANTUM_SIMULATOR_HEADER, "r", encoding="utf-8") as f:
            header = f.read()
        with open(_QUANTUM_SIMULATOR_CPP, "r", encoding="utf-8") as f:
            implementation = f.read()
        with open(_HIPSTATEVEC_HEADER, "r", encoding="utf-8") as f:
            hip_header = f.read()
        with open(_HIPSTATEVEC_SOURCE, "r", encoding="utf-8") as f:
            hip_source = f.read()
        with open(_HIPSTATEVEC_SINGLE_QUBIT_KERNELS, "r", encoding="utf-8") as f:
            single_qubit_kernels = f.read()
        with open(_BINDINGS_CPP, "r", encoding="utf-8") as f:
            bindings = f.read()
        with open(_FRAMEWORK_RUNTIME, "r", encoding="utf-8") as f:
            runtime = f.read()

        self.assertIn("void apply_gate_batch", header)
        self.assertIn("void ApplyGateBatch", header)
        self.assertIn("QuantumSimulator::apply_gate_batch", implementation)
        self.assertIn("rocsvApplyRxBatch", implementation)
        self.assertIn("rocsvApplyRyBatch", implementation)
        self.assertIn("rocsvApplyRzBatch", implementation)
        self.assertIn("rocsvApplyCRXBatch", implementation)
        self.assertIn("rocsvApplyCRYBatch", implementation)
        self.assertIn("rocsvApplyCRZBatch", implementation)
        self.assertIn("rocsvApplyRxBatch", hip_header)
        self.assertIn("rocsvApplyRyBatch", hip_header)
        self.assertIn("rocsvApplyRzBatch", hip_header)
        self.assertIn("rocsvApplyCRXBatch", hip_header)
        self.assertIn("rocsvApplyCRYBatch", hip_header)
        self.assertIn("rocsvApplyCRZBatch", hip_header)
        self.assertIn("launch_single_qubit_matrix_batch", hip_source)
        self.assertIn("launch_controlled_single_qubit_matrix_batch", hip_source)
        self.assertIn("apply_single_qubit_matrix_batch_kernel", single_qubit_kernels)
        self.assertIn("apply_controlled_single_qubit_matrix_batch_kernel", single_qubit_kernels)
        self.assertIn(".def(\"apply_gate_batch\"", bindings)
        self.assertIn(".def(\"ApplyGateBatch\"", bindings)
        self.assertIn("def apply_operation_batch", runtime)

    def test_public_simulator_exposes_native_probabilities(self):
        with open(_QUANTUM_SIMULATOR_HEADER, "r", encoding="utf-8") as f:
            header = f.read()
        with open(_QUANTUM_SIMULATOR_CPP, "r", encoding="utf-8") as f:
            implementation = f.read()
        with open(_HIPSTATEVEC_HEADER, "r", encoding="utf-8") as f:
            hip_header = f.read()
        with open(_HIPSTATEVEC_SOURCE, "r", encoding="utf-8") as f:
            hip_source = f.read()
        with open(_BINDINGS_CPP, "r", encoding="utf-8") as f:
            bindings = f.read()
        with open(_LOW_LEVEL_BINDINGS_CPP, "r", encoding="utf-8") as f:
            low_level_bindings = f.read()
        with open(_FRAMEWORK_RUNTIME, "r", encoding="utf-8") as f:
            runtime = f.read()

        self.assertIn("std::vector<double> probabilities", header)
        self.assertIn("std::vector<double> probabilities_batch", header)
        self.assertIn("std::vector<long long> measure_batch", header)
        self.assertIn("std::vector<double> Probabilities", header)
        self.assertIn("std::vector<double> ProbabilitiesBatch", header)
        self.assertIn("std::vector<long long> MeasureBatch", header)
        self.assertIn("QuantumSimulator::probabilities", implementation)
        self.assertIn("QuantumSimulator::probabilities_batch", implementation)
        self.assertIn("QuantumSimulator::measure_batch", implementation)
        self.assertIn("device_state_vector_ + batch_index * state_vec_size_", implementation)
        self.assertIn("rocsvSample(sim_handle_", implementation)
        self.assertIn("QuantumSimulator::MeasureBatch", implementation)
        self.assertIn("rocsvProbabilities", implementation)
        self.assertIn("rocsvProbabilitiesBatch", implementation)
        self.assertIn("rocsvProbabilities", hip_header)
        self.assertIn("rocsvProbabilitiesBatch", hip_header)
        self.assertRegex(hip_source, r"rocqStatus_t\s+rocsvProbabilities\s*\(")
        self.assertRegex(hip_source, r"rocqStatus_t\s+rocsvProbabilitiesBatch\s*\(")
        self.assertIn("accumulate_local_sample_probabilities", hip_source)
        self.assertIn("accumulate_sample_probabilities_batch_kernel", hip_source)
        self.assertIn("compute_local_sample_probabilities_batch", hip_source)
        self.assertIn("accumulate_distributed_sample_probabilities_rccl", hip_source)
        self.assertIn("compute_distributed_sample_probabilities", hip_source)
        self.assertIn("return compute_distributed_sample_probabilities", hip_source)
        self.assertIn(".def(\"probabilities\"", bindings)
        self.assertIn(".def(\"probabilities_batch\"", bindings)
        self.assertIn(".def(\"measure_batch\"", bindings)
        self.assertIn(".def(\"Probabilities\"", bindings)
        self.assertIn(".def(\"ProbabilitiesBatch\"", bindings)
        self.assertIn(".def(\"MeasureBatch\"", bindings)
        self.assertIn("m.def(\"probabilities\"", low_level_bindings)
        self.assertIn("_native_probabilities_unavailable", runtime)
        self.assertIn("def probabilities_batch", runtime)
        self.assertIn("def measure_batch", runtime)

    def test_public_simulator_exposes_controlled_matrix_application(self):
        with open(_QUANTUM_SIMULATOR_HEADER, "r", encoding="utf-8") as f:
            header = f.read()
        with open(_QUANTUM_SIMULATOR_CPP, "r", encoding="utf-8") as f:
            implementation = f.read()
        with open(_BINDINGS_CPP, "r", encoding="utf-8") as f:
            bindings = f.read()
        with open(_FRAMEWORK_RUNTIME, "r", encoding="utf-8") as f:
            runtime = f.read()

        self.assertIn("void apply_controlled_matrix", header)
        self.assertIn("void ApplyControlledGate", header)
        self.assertIn("QuantumSimulator::apply_controlled_matrix", implementation)
        self.assertIn("rocsvApplyControlledMatrix", implementation)
        self.assertIn(".def(\"apply_controlled_matrix\"", bindings)
        self.assertIn(".def(\"ApplyControlledGate\"", bindings)
        self.assertIn("def apply_controlled_matrix", runtime)

    def test_public_simulator_exposes_qubit_measure_and_reset(self):
        with open(_QUANTUM_SIMULATOR_HEADER, "r", encoding="utf-8") as f:
            header = f.read()
        with open(_QUANTUM_SIMULATOR_CPP, "r", encoding="utf-8") as f:
            implementation = f.read()
        with open(_BINDINGS_CPP, "r", encoding="utf-8") as f:
            bindings = f.read()
        with open(_FRAMEWORK_RUNTIME, "r", encoding="utf-8") as f:
            runtime = f.read()

        self.assertIn("int measure_qubit", header)
        self.assertIn("void reset_qubit", header)
        self.assertIn("int MeasureQubit", header)
        self.assertIn("void ResetQubit", header)
        self.assertIn("QuantumSimulator::measure_qubit", implementation)
        self.assertIn("QuantumSimulator::reset_qubit", implementation)
        self.assertIn("rocsvMeasure", implementation)
        self.assertIn("rocsvApplyX", implementation)
        self.assertIn(".def(\"measure_qubit\"", bindings)
        self.assertIn(".def(\"reset_qubit\"", bindings)
        self.assertIn(".def(\"MeasureQubit\"", bindings)
        self.assertIn(".def(\"ResetQubit\"", bindings)
        self.assertIn("def measure_qubit", runtime)
        self.assertIn("def reset_qubit", runtime)

    def test_public_simulator_exposes_sparse_hamiltonian_moments(self):
        with open(_QUANTUM_SIMULATOR_HEADER, "r", encoding="utf-8") as f:
            header = f.read()
        with open(_QUANTUM_SIMULATOR_CPP, "r", encoding="utf-8") as f:
            implementation = f.read()
        with open(_HIPSTATEVEC_HEADER, "r", encoding="utf-8") as f:
            hip_header = f.read()
        with open(_HIPSTATEVEC_SOURCE, "r", encoding="utf-8") as f:
            hipstatevec = f.read()
        with open(_BINDINGS_CPP, "r", encoding="utf-8") as f:
            bindings = f.read()
        with open(_LOW_LEVEL_BINDINGS_CPP, "r", encoding="utf-8") as f:
            low_level_bindings = f.read()
        with open(_FRAMEWORK_RUNTIME, "r", encoding="utf-8") as f:
            runtime = f.read()

        self.assertIn("sparse_hamiltonian_moments", header)
        self.assertIn("SparseHamiltonianMoments", header)
        self.assertIn("sparse_hamiltonian_moments_batch", header)
        self.assertIn("SparseHamiltonianMomentsBatch", header)
        self.assertIn("expectation_matrix", header)
        self.assertIn("ExpectationMatrix", header)
        self.assertIn("expectation_matrix_batch", header)
        self.assertIn("ExpectationMatrixBatch", header)
        self.assertIn("expectation_pauli_string_batch", header)
        self.assertIn("GetExpectationPauliStringBatch", header)
        self.assertIn("QuantumSimulator::sparse_hamiltonian_moments", implementation)
        self.assertIn("QuantumSimulator::sparse_hamiltonian_moments_batch", implementation)
        self.assertIn("QuantumSimulator::expectation_matrix", implementation)
        self.assertIn("QuantumSimulator::expectation_matrix_batch", implementation)
        self.assertIn("QuantumSimulator::expectation_pauli_string_batch", implementation)
        self.assertIn("rocsvGetExpectationPauliStringBatch", implementation)
        self.assertIn("rocsvGetExpectationPauliStringBatch", hip_header)
        self.assertIn("reduce_expectation_pauli_string_batch_kernel", hipstatevec)
        self.assertIn("compute_local_expectation_pauli_string_batch", hipstatevec)
        self.assertIn("rocsvGetExpectationMatrix", implementation)
        self.assertIn("rocsvGetExpectationMatrixBatch", implementation)
        self.assertIn("rocsvGetExpectationMatrixBatch", hip_header)
        self.assertIn("dim3(blocks, static_cast<unsigned>(batch_size))", hipstatevec)
        self.assertIn("rocsvGetSparseMatrixMoments", implementation)
        self.assertIn("rocsvGetSparseMatrixMomentsBatch", implementation)
        self.assertIn("rocsvGetSparseMatrixMoments", hipstatevec)
        self.assertIn("rocsvGetSparseMatrixMomentsBatch", hipstatevec)
        self.assertIn("reduce_sparse_matrix_moments_kernel", hipstatevec)
        self.assertIn("reduce_sparse_matrix_moments_batch_kernel", hipstatevec)
        self.assertIn("Sparse Hamiltonian CSR indptr", implementation)
        sparse_body = implementation.split("QuantumSimulator::sparse_hamiltonian_moments", 1)[1].split(
            "unsigned QuantumSimulator::num_qubits", 1
        )[0]
        self.assertIn("hipMemcpyHostToDevice", sparse_body)
        self.assertNotIn("get_statevector()", sparse_body)
        self.assertNotIn("h_state", sparse_body)
        self.assertIn(".def(\"expectation_matrix\"", bindings)
        self.assertIn(".def(\"ExpectationMatrix\"", bindings)
        self.assertIn(".def(\"expectation_matrix_batch\"", bindings)
        self.assertIn(".def(\"ExpectationMatrixBatch\"", bindings)
        self.assertIn("infer_batch_size_from_state_buffer", low_level_bindings)
        self.assertIn("m.def(\"get_expectation_pauli_string_batch\"", low_level_bindings)
        self.assertIn("rocsvGetExpectationPauliStringBatch", low_level_bindings)
        self.assertIn("m.def(\"get_expectation_matrix_batch\"", low_level_bindings)
        self.assertIn("rocsvGetExpectationMatrixBatch", low_level_bindings)
        self.assertIn("m.def(\"get_sparse_matrix_moments_batch\"", low_level_bindings)
        self.assertIn("rocsvGetSparseMatrixMomentsBatch", low_level_bindings)
        self.assertIn(".def(\"expectation_pauli_string_batch\"", bindings)
        self.assertIn(".def(\"GetExpectationPauliStringBatch\"", bindings)
        self.assertIn(".def(\"sparse_hamiltonian_moments\"", bindings)
        self.assertIn(".def(\"SparseHamiltonianMoments\"", bindings)
        self.assertIn(".def(\"sparse_hamiltonian_moments_batch\"", bindings)
        self.assertIn(".def(\"SparseHamiltonianMomentsBatch\"", bindings)
        self.assertIn("def expectation_matrix_batch", runtime)
        self.assertIn("def sparse_hamiltonian_moments_batch", runtime)

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
        self.assertIn("statevector_to_little_endian_wires", source)
        self.assertIn("self._runtime.set_statevector", source)
        self.assertIn("_native_mcx_wire_indices", source)
        self.assertIn("def _apply_mcx_with_control_values", source)
        self.assertIn("control_values", source)
        self.assertIn("def _controlled_qubit_unitary_components", source)
        self.assertIn("def _apply_controlled_qubit_unitary", source)
        self.assertIn("runtime.apply_controlled_matrix", source)
        self.assertIn("_apply_native_or_matrix(self._runtime, \"MCX\"", source)
        self.assertIn("def _apply_multirz", source)
        self.assertIn("_apply_multirz(self._runtime, wire_indices, theta)", source)
        self.assertIn("def _supports_native_phase_decomposition", source)
        self.assertIn("def _supports_native_parametric_decomposition", source)
        self.assertIn("def _supports_native_gate_decomposition", source)
        self.assertIn("def _apply_phase_shift", source)
        self.assertIn("def _apply_sx", source)
        self.assertIn("preserve_global_phase", source)
        self.assertIn("def _circuit_preserves_global_phase", source)
        self.assertIn("def _apply_controlled_phase_shift", source)
        self.assertIn("def _apply_controlled_phase_variant", source)
        self.assertIn("_apply_controlled_phase_shift(self._runtime, wire_indices, theta)", source)
        self.assertIn("def _apply_phase_shift_batch", source)
        self.assertIn("def _apply_controlled_phase_shift_batch", source)
        self.assertIn("def _apply_cy", source)
        self.assertIn("def _apply_ccz", source)
        self.assertIn("def _apply_ch", source)
        self.assertIn("def _apply_iswap", source)
        self.assertIn("def _apply_pswap", source)
        self.assertIn("def _apply_siswap", source)
        self.assertIn("def _apply_ecr", source)
        self.assertIn("def _apply_single_excitation", source)
        self.assertIn("def _apply_single_excitation_phase_variant", source)
        self.assertIn("def _apply_crot", source)
        self.assertIn("def _apply_double_excitation", source)
        self.assertIn("def _apply_double_excitation_phase_variant", source)
        self.assertIn("def _apply_fermionic_swap", source)
        self.assertIn("def _apply_orbital_rotation", source)
        self.assertIn("gate_name == \"CH\"", source)
        self.assertIn("gate_name == \"CY\"", source)
        self.assertIn("gate_name == \"CCZ\"", source)
        self.assertIn("gate_name == \"ISWAP\"", source)
        self.assertIn("gate_name == \"PSWAP\"", source)
        self.assertIn("gate_name in {\"SISWAP\", \"SQISW\"}", source)
        self.assertIn("gate_name == \"ECR\"", source)
        self.assertIn("gate_name == \"SingleExcitation\"", source)
        self.assertIn("gate_name in {\"SingleExcitationPlus\", \"SingleExcitationMinus\"}", source)
        self.assertIn("gate_name == \"CRot\"", source)
        self.assertIn("gate_name == \"DoubleExcitation\"", source)
        self.assertIn("gate_name in {\"DoubleExcitationPlus\", \"DoubleExcitationMinus\"}", source)
        self.assertIn("gate_name == \"FermionicSWAP\"", source)
        self.assertIn("gate_name == \"OrbitalRotation\"", source)
        self.assertIn("def _apply_isingxx", source)
        self.assertIn("def _apply_isingyy", source)
        self.assertIn("def _apply_isingxx_batch", source)
        self.assertIn("def _apply_isingyy_batch", source)
        self.assertIn("def _apply_isingxy", source)
        self.assertIn("gate_name == \"IsingXX\"", source)
        self.assertIn("gate_name == \"IsingYY\"", source)
        self.assertIn("gate_name == \"IsingXY\"", source)
        self.assertIn("gate_name == \"IsingZZ\"", source)
        self.assertIn("def _shot_count", source)
        self.assertIn("shots = _shot_count(self.shots)", source)
        self.assertIn("raw_samples = self._runtime.measure(all_wires, shots)", source)
        self.assertIn("self._runtime.measure_batch", source)
        self.assertIn("measurement_names[0] not in {\"SampleMP\", \"CountsMP\"}", source)
        self.assertIn("def _sample_result_from_rows", source)
        self.assertIn("def _counts_result_from_rows", source)
        self.assertIn("def _ensure_state", source)
        self.assertIn("self._state = self._runtime.statevector()", source)
        self.assertIn("return self._ensure_state()", source)
        self.assertIn("samples_to_binary_rows", source)
        self.assertIn("sample_rows_from_statevector", source, "legacy fallback should remain explicit")
        self.assertIn("def analytic_probability", source)
        self.assertIn("self._runtime.probabilities", source)
        self.assertIn("if not wires_to_trace: return all_probs", source)
        self.assertIn("def expval", source)
        self.assertIn("def var", source)
        self.assertIn("def execute", source)
        self.assertIn("def batch_execute", source)
        self.assertIn("_try_execute_batched_parameter_circuits", source)
        self.assertIn("measurement_names = [measurement.__class__.__name__", source)
        self.assertIn("reference_measurement_specs = []", source)
        self.assertIn("batched_values = []", source)
        self.assertIn("\"ExpectationMP\", \"VarianceMP\", \"ProbabilityMP\"", source)
        self.assertIn("\"ExpectationMP\", \"VarianceMP\"", source)
        self.assertIn("_pauli_square_terms(payload[1])", source)
        self.assertIn("self._runtime.probabilities_batch(probability_targets)", source)
        self.assertIn("self._runtime.apply_operation_batch", source)
        self.assertIn("\"PhaseShift\", \"ControlledPhaseShift\", \"IsingXX\", \"IsingYY\", \"IsingZZ\"", source)
        self.assertIn("self._runtime.probabilities_batch", source)
        self.assertIn("_analytic_measurements_use_native_pauli", source)
        self.assertIn("_skip_diagonalizing_rotations", source)
        self.assertIn("_diagonalizing_rotations_applied", source)
        self.assertIn("\"Hadamard\"", source)
        self.assertIn("\"SparseHamiltonian\"", source)
        self.assertIn("observable.name == \"Hadamard\"", source)
        self.assertIn("observable.name == \"Hermitian\"", source)
        self.assertIn("def _native_hermitian_expectation", source)
        self.assertIn("def _observable_batch_payload", source)
        self.assertIn("runtime.expectation_matrix", source)
        self.assertIn("runtime.expectation_matrix_batch", source)
        self.assertIn("def _projector_terms_from_observable", source)
        self.assertIn("observable.name != \"Projector\"", source)
        self.assertIn("_native_sparse_hamiltonian_moments", source)
        self.assertIn("_native_sparse_hamiltonian_moments_batch", source)
        self.assertIn("runtime.sparse_hamiltonian_moments", source)
        self.assertIn("runtime.sparse_hamiltonian_moments_batch", source)
        self.assertIn("_sparse_hamiltonian_moments", source)
        self.assertIn("sparse_matrix(wire_order=wire_order, format=\"csr\")", source)
        self.assertIn("rotation_ops = list(rotations or [])", source)
        self.assertIn("circuit_ops + rotation_ops", source)
        self.assertIn("def execute_and_gradients", source)
        self.assertIn("_capture_adjoint_reference_state", source)
        self.assertIn("def _apply_unitary", source)
        self.assertIn("_pauli_square_terms", source)
        self.assertIn("expectation_pauli_string", source)
        self.assertIn("_evaluate_pauli_terms_batch", source)
        self.assertIn("expectation_pauli_string_batch", source)

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
        self.assertIn("def _apply_phase_gate_batch", source)
        self.assertIn("def _apply_controlled_phase_gate", source)
        self.assertIn("def _apply_controlled_phase_gate_batch", source)
        self.assertIn("def _apply_rxx_gate", source)
        self.assertIn("def _apply_rxx_gate_batch", source)
        self.assertIn("def _apply_ryy_gate", source)
        self.assertIn("def _apply_ryy_gate_batch", source)
        self.assertIn("def _apply_rzz_gate", source)
        self.assertIn("def _apply_rzz_gate_batch", source)
        self.assertIn("def _apply_multirz_gate", source)
        self.assertIn("def _apply_pauli_evolution_gate", source)
        self.assertIn("_pauli_evolution_terms", source)
        self.assertIn("def _apply_circuit_batch", source)
        self.assertIn("self._runtime.apply_operation_batch", source)
        self.assertIn("decomposed p/cp/rxx/ryy/rzz parameters", source)
        self.assertIn("def _apply_sx_gate", source)
        self.assertIn("def _apply_u_gate", source)
        self.assertIn("def _apply_cy_gate", source)
        self.assertIn("def _apply_ccz_gate", source)
        self.assertIn("def _apply_ch_gate", source)
        self.assertIn("def _apply_dcx_gate", source)
        self.assertIn("def _apply_iswap_gate", source)
        self.assertIn("def _apply_ecr_gate", source)
        self.assertIn("def _apply_tdg_gate", source)
        self.assertIn("def _apply_rccx_gate", source)
        self.assertIn("def _apply_rcccx_gate", source)
        self.assertIn("def _apply_controlled_base_gate", source)
        self.assertIn("statevector_to_little_endian_wires", source)
        self.assertIn("self._runtime.set_statevector", source)
        self.assertIn("Operator(op).data", source)
        self.assertIn("statevector=False", source)
        self.assertIn("Target(num_qubits=int(num_qubits))", source)
        self.assertIn("MCXGate(3)", source)
        self.assertIn("raw_samples = self._runtime.measure(qubits_to_measure, shots)", source)
        self.assertIn("def _run_dynamic_sampling", source)
        self.assertIn("def _apply_circuit_trajectory", source)
        self.assertIn("if op.name == \"if_else\"", source)
        self.assertIn("if op.name == \"for_loop\"", source)
        self.assertIn("if op.name == \"switch_case\"", source)
        self.assertIn("if op.name == \"while_loop\"", source)
        self.assertIn("if op.name == \"break_loop\"", source)
        self.assertIn("if op.name == \"continue_loop\"", source)
        self.assertIn("_classical_value", source)
        self.assertIn("_for_loop_metadata", source)
        self.assertIn("_bind_for_loop_block", source)
        self.assertIn("return \"break\"", source)
        self.assertIn("return \"continue\"", source)
        self.assertIn("max_dynamic_loop_iterations", source)
        self.assertIn("measure_qubit", source)
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
        self.assertIn("def _observable_signature", source)
        self.assertIn("observable_cache", source)
        self.assertIn("indices_by_parameter", source)
        self.assertIn("_try_run_pub_batched_parameters", source)
        self.assertIn("observable_values_by_cache", source)
        self.assertIn("observable_cache_keys", source)
        self.assertIn("parameter_offsets", source)
        self.assertIn("_apply_circuit_batch", source)
        self.assertIn("_estimate_combined_observable_terms_batch", source)
        self.assertIn("_estimate_observable_plan_batch", source)
        self.assertIn("self._backend._apply_circuit", source)
        self.assertIn("estimate_observable_batch", source)
        self.assertIn("estimate_pauli_observable", source)
        self.assertIn("estimate_pauli_observable_batch", source)
        self.assertIn("expectation_pauli_string_batch", source)
        self.assertIn("expectation_matrix_batch", source)
        self.assertIn("shots\": 0", source)

    def test_qiskit_sampler_uses_native_sampling(self):
        with open(_QISKIT_SAMPLER, "r", encoding="utf-8") as f:
            source = f.read()

        self.assertIn("class RocQuantumSampler", source)
        self.assertIn("SamplerPub.coerce", source)
        self.assertIn("self._backend._apply_circuit", source)
        self.assertIn("self._backend._runtime.measure", source)
        self.assertIn("def _try_run_pub_batched_parameters", source)
        self.assertIn("self._backend._apply_circuit_batch", source)
        self.assertIn("self._backend._runtime.measure_batch", source)
        self.assertIn("\"batched_parameters\": True", source)
        self.assertIn("BitArray.from_bool_array", source)


if __name__ == "__main__":
    unittest.main()
