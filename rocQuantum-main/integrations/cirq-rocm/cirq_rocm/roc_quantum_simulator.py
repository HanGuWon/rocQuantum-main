# cirq-rocm/cirq_rocm/roc_quantum_simulator.py
import cirq
import numpy as np

try:
    import rocquantum_bind
except ImportError:
    raise ImportError("The 'rocquantum_bind' module is not installed.")

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
    def _run(self, circuit, param_resolver, repetitions):
        qubit_order = sorted(circuit.all_qubits())
        sim, q_map = self._execute_circuit(circuit, qubit_order)
        state_vector = None
        full_outcomes = None
        measurements = {}

        for op in circuit.all_operations():
            if isinstance(op.gate, cirq.MeasurementGate):
                key = op.gate.key
                indices = [q_map[q] for q in op.qubits]
                measure = getattr(sim, "measure", None)
                if callable(measure):
                    raw_samples = measure(indices, repetitions)
                    values = _samples_to_bits(raw_samples, len(indices))
                else:
                    if state_vector is None:
                        state_vector = sim.GetStateVector()
                        probs = np.abs(state_vector) ** 2
                        full_outcomes = np.random.choice(len(probs), size=repetitions, p=probs)
                    values = (full_outcomes[:, np.newaxis] >> indices) & 1
                measurements[key] = values
        return measurements

    def _execute_circuit(self, circuit, qubit_order):
        q_map = {q: i for i, q in enumerate(qubit_order)}
        sim = rocquantum_bind.QSim(num_qubits=len(q_map))
        for op in circuit.all_operations():
            if isinstance(op.gate, cirq.MeasurementGate):
                continue
            gate_type = type(op.gate)
            if gate_type in CIRQ_TO_ROCQ_GATES:
                indices = [q_map[q] for q in op.qubits]
                sim.ApplyGate(CIRQ_TO_ROCQ_GATES[gate_type], *indices)
            elif isinstance(op.gate, (cirq.MatrixGate, cirq.Rx, cirq.Ry, cirq.Rz)):
                matrix = cirq.unitary(op)
                sim.ApplyGate(matrix, q_map[op.qubits[0]])
            else:
                raise TypeError(f"Unsupported gate: {op.gate}")
        sim.Execute()
        return sim, q_map

    def _get_final_statevector(self, circuit, qubit_order):
        sim, _ = self._execute_circuit(circuit, qubit_order)
        return sim.GetStateVector()

    def _perform_final_state_simulation(self, circuit, qubit_order, initial_state):
        if not np.allclose(initial_state[0], 1):
            raise ValueError("Only simulation from |0> state is supported.")
        state_vector = self._get_final_statevector(circuit, qubit_order)
        return cirq.StateVectorTrialResult(params={}, measurements={}, final_simulator_state=state_vector)
