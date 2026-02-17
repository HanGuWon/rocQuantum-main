# pennylane_rocq/rocq_device.py
import pennylane as qml
from pennylane import QubitDevice
from pennylane.operation import Operation
import numpy as np

try:
    import rocquantum_bind
except ImportError:
    # Raise a friendly error message if the binding is not installed.
    raise ImportError(
        "The 'rocquantum_bind' module is not installed. "
        "Please build and install it from the rocQuantum-1 project."
    )

# Mapping from PennyLane operation names to rocQuantum names
PENNYLANE_TO_ROCQ_GATES = {
    "PauliX": "X", "PauliY": "Y", "PauliZ": "Z",
    "Hadamard": "H", "S": "S", "T": "T",
    "CNOT": "CNOT", "CZ": "CZ",
}

class RocQDevice(QubitDevice):
    name = "rocQuantum Simulator Device"
    short_name = "rocquantum.qpu"
    author = "Gemini"
    version = "0.1.0"

    operations = set(PENNYLANE_TO_ROCQ_GATES.keys()) | {"QubitUnitary", "RX", "RY", "RZ"}
    observables = {"PauliX", "PauliY", "PauliZ", "Identity", "Counts", "State"}

    def __init__(self, wires, shots=None, **kwargs):
        super().__init__(wires=wires, shots=shots)
        self.sim = None
        self._state = None
        self.reset()

    def reset(self):
        self.sim = rocquantum_bind.QSim(num_qubits=len(self.wires))
        self._state = None

    def apply(self, operations: list[Operation]):
        for op in operations:
            gate_name = op.name
            if gate_name in PENNYLANE_TO_ROCQ_GATES:
                wire_indices = [self.wire_map[w] for w in op.wires]
                self.sim.ApplyGate(PENNYLANE_TO_ROCQ_GATES[gate_name], *wire_indices)
            elif gate_name in ("RX", "RY", "RZ", "QubitUnitary"):
                matrix = qml.matrix(op)
                target_idx = self.wire_map[op.wires[0]]
                self.sim.ApplyGate(matrix.astype(np.complex128), target_idx)
            else:
                raise NotImplementedError(f"Operation {gate_name} not supported.")
        self.sim.Execute()
        self._state = self.sim.GetStateVector()

    @property
    def state(self):
        return self._state

    def generate_samples(self):
        probs = self.analytic_probability(wires=self.wires)
        num_states = len(probs)
        samples_count = np.random.multinomial(self.shots, probs)
        # Expand counts into individual shot outcomes
        sample_indices = np.repeat(np.arange(num_states), samples_count)
        num_wires = len(self.wires)
        # Convert each outcome index to a binary array (one row per shot)
        return np.array(
            [[(idx >> (num_wires - 1 - w)) & 1 for w in range(num_wires)]
             for idx in sample_indices],
            dtype=int,
        )

    def analytic_probability(self, wires=None):
        if self._state is None: return None
        all_probs = np.abs(self._state) ** 2
        if wires is None: return all_probs
        wires_to_trace = [i for i, w in enumerate(self.wires) if w not in wires]
        return self.marginal_prob(all_probs, wires_to_trace)