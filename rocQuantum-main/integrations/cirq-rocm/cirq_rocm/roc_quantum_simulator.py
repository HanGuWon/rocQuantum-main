# cirq-rocm/cirq_rocm/roc_quantum_simulator.py
import cirq
import numpy as np

from rocquantum.framework_runtime import RocQuantumRuntime, matrix_to_little_endian_wires

try:
    import rocquantum_bind
except ImportError:
    rocquantum_bind = None

CIRQ_TO_ROCQ_GATES = {
    type(cirq.X): "X",
    type(cirq.Y): "Y",
    type(cirq.Z): "Z",
    type(cirq.H): "H",
    type(cirq.S): "S",
    type(cirq.T): "T",
    type(cirq.CNOT): "CNOT",
    type(cirq.CZ): "CZ",
}


def _samples_to_bits(raw_samples, width):
    return np.array(
        [[(int(sample) >> bit) & 1 for bit in range(width)] for sample in raw_samples],
        dtype=np.int8,
    )


class RocQuantumSimulator(cirq.SimulatesFinalState, cirq.SimulatesSamples):
    def _require_binding(self):
        if rocquantum_bind is None:
            raise ImportError(
                "The 'rocquantum_bind' module is not installed. "
                "Build rocQuantum with native Python bindings before executing Cirq circuits."
            )
        return rocquantum_bind

    def _run(self, circuit, param_resolver, repetitions):
        qubit_order = sorted(circuit.all_qubits())
        runtime, q_map = self._execute_circuit(circuit, qubit_order)
        state_vector = None
        full_outcomes = None
        measurements = {}

        for op in circuit.all_operations():
            if isinstance(op.gate, cirq.MeasurementGate):
                key = op.gate.key
                indices = [q_map[q] for q in op.qubits]
                try:
                    raw_samples = runtime.measure(indices, repetitions)
                    values = _samples_to_bits(raw_samples, len(indices))
                except NotImplementedError:
                    if state_vector is None:
                        state_vector = runtime.statevector()
                        probs = np.abs(state_vector) ** 2
                        full_outcomes = np.random.choice(len(probs), size=repetitions, p=probs)
                    values = (full_outcomes[:, np.newaxis] >> indices) & 1
                measurements[key] = values
        return measurements

    def _execute_circuit(self, circuit, qubit_order):
        binding = self._require_binding()
        q_map = {q: i for i, q in enumerate(qubit_order)}
        runtime = RocQuantumRuntime.from_bindings(len(q_map), binding_module=binding)
        for op in circuit.all_operations():
            if isinstance(op.gate, cirq.MeasurementGate):
                continue
            gate_type = type(op.gate)
            indices = [q_map[q] for q in op.qubits]
            if gate_type in CIRQ_TO_ROCQ_GATES:
                runtime.apply_operation(CIRQ_TO_ROCQ_GATES[gate_type], indices)
            elif isinstance(op.gate, (cirq.MatrixGate, cirq.Rx, cirq.Ry, cirq.Rz)):
                matrix = matrix_to_little_endian_wires(cirq.unitary(op))
                runtime.apply_matrix(matrix, indices)
            else:
                raise TypeError(f"Unsupported gate: {op.gate}")
        execute = getattr(runtime.simulator, "Execute", None)
        if callable(execute):
            execute()
        return runtime, q_map

    def _get_final_statevector(self, circuit, qubit_order):
        runtime, _ = self._execute_circuit(circuit, qubit_order)
        return runtime.statevector()

    def _perform_final_state_simulation(self, circuit, qubit_order, initial_state):
        if not np.allclose(initial_state[0], 1):
            raise ValueError("Only simulation from |0> state is supported.")
        state_vector = self._get_final_statevector(circuit, qubit_order)
        return cirq.StateVectorTrialResult(params={}, measurements={}, final_simulator_state=state_vector)
