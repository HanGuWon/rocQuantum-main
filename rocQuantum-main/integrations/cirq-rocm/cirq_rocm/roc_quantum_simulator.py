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

class RocQuantumSimulator(cirq.SimulatesFinalState, cirq.SimulatesSamples):
    def _run(self, circuit, param_resolver, repetitions):
        qubit_order = sorted(circuit.all_qubits())
        state_vector = self._get_final_statevector(circuit, qubit_order)
        probs = np.abs(state_vector) ** 2
        outcomes = np.random.choice(len(probs), size=repetitions, p=probs)
        measurements = {}
        for op in circuit.all_operations():
            if isinstance(op.gate, cirq.MeasurementGate):
                key = op.gate.key
                indices = [qubit_order.index(q) for q in op.qubits]
                values = (outcomes[:, np.newaxis] >> indices) & 1
                measurements[key] = values
        return measurements

    def _get_final_statevector(self, circuit, qubit_order):
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
        return sim.GetStateVector()

    def _perform_final_state_simulation(self, circuit, qubit_order, initial_state):
        if not np.allclose(initial_state[0], 1):
            raise ValueError("Only simulation from |0> state is supported.")
        state_vector = self._get_final_statevector(circuit, qubit_order)
        return cirq.StateVectorTrialResult(params={}, measurements={}, final_simulator_state=state_vector)
