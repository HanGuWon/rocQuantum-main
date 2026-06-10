import numpy as np
import uuid
from collections import Counter

from qiskit.providers import BackendV2, Options
from qiskit.transpiler import Target
from qiskit.result import Result, ExperimentResult, ExperimentResultData
from qiskit.circuit import Measure
from qiskit.circuit.library import RXGate, RYGate, RZGate, CXGate, HGate, UnitaryGate

from rocquantum_bind import QuantumSimulator


def _format_qiskit_memory(sample, measured_items, memory_width):
    if memory_width <= 0:
        memory_width = len(measured_items)
    bits = ["0"] * memory_width
    for packed_bit, (classical_bit, _) in enumerate(measured_items):
        output_index = memory_width - 1 - classical_bit
        if 0 <= output_index < memory_width:
            bits[output_index] = "1" if ((int(sample) >> packed_bit) & 1) else "0"
    return "".join(bits)

class RocQuantumBackend(BackendV2):
    """
    rocQuantum Qiskit Backend.

    A Qiskit backend that interfaces with the rocQuantum C++/HIP simulator.
    """
    def __init__(self, provider, **kwargs):
        super().__init__(provider=provider, name="rocq_simulator", **kwargs)
        
        # The simulator is instantiated once and maintained for the lifetime of the backend
        # We assume a fixed, maximum number of qubits or re-instantiate if needed.
        # For simplicity, let's assume it's configured on first use.
        self._simulator = None
        self._num_qubits = 0

        # Define the target with supported gates
        self.target = Target()
        self.target.add_instruction(RXGate, name="rx")
        self.target.add_instruction(RYGate, name="ry")
        self.target.add_instruction(RZGate, name="rz")
        self.target.add_instruction(CXGate, name="cx") # CNOT
        self.target.add_instruction(HGate, name="h")
        self.target.add_instruction(UnitaryGate, name="unitary")
        self.target.add_instruction(Measure, name="measure")

    @classmethod
    def _default_options(cls):
        return Options(shots=1024)

    def _ensure_simulator(self, num_qubits):
        """Create or reset the simulator if the qubit count changes."""
        if self._simulator is None or self._num_qubits != num_qubits:
            self._simulator = QuantumSimulator(num_qubits)
            self._num_qubits = num_qubits
        else:
            self._simulator.reset()

    def run(self, run_input, **options):
        if not isinstance(run_input, list):
            run_input = [run_input]

        job_id = str(uuid.uuid4())
        shots = options.get("shots", self.options.shots)
        results = []

        for circuit in run_input:
            self._ensure_simulator(circuit.num_qubits)
            
            # Simplified translation loop
            measured_bits = {} # Map classical bit index to qubit index
            for instruction in circuit.data:
                op = instruction.operation
                q_indices = [circuit.find_bit(q).index for q in instruction.qubits]
                
                if op.name in ('rx', 'ry', 'rz'):
                    self._simulator.apply_gate(op.name.upper(), q_indices, op.params)
                elif op.name in ('cx', 'h'):
                    self._simulator.apply_gate(op.name.upper(), q_indices, [])
                elif op.name == 'unitary':
                    self._simulator.apply_matrix(op.to_matrix(), q_indices)
                elif op.name == 'measure':
                    # Store which classical bit this measurement corresponds to
                    c_index = circuit.find_bit(instruction.clbits[0]).index
                    measured_bits[c_index] = q_indices[0]

            # Perform measurement on the required qubits.  rocsvSample packs
            # result bits in the order qubits are requested, so keep the
            # classical-bit mapping explicit when formatting Qiskit memory.
            if measured_bits:
                measured_items = sorted(measured_bits.items())
            else: # If no measure op, measure all into matching bit positions.
                measured_items = [(idx, idx) for idx in range(circuit.num_qubits)]
            qubits_to_measure = [qubit for _, qubit in measured_items]

            raw_samples = self._simulator.measure(qubits_to_measure, shots)
            memory_width = getattr(circuit, "num_clbits", 0) or len(measured_items)
            memory = [_format_qiskit_memory(sample, measured_items, memory_width) for sample in raw_samples]
            formatted_counts = dict(Counter(memory))

            exp_data = ExperimentResultData(
                counts=formatted_counts,
                memory=memory
            )
            
            exp_result = ExperimentResult(
                shots=shots,
                success=True,
                data=exp_data,
                header=circuit.header
            )
            results.append(exp_result)

        return Result(
            backend_name=self.name,
            backend_version=self.backend_version,
            job_id=job_id,
            qobj_id=None,
            success=True,
            results=results
        )
