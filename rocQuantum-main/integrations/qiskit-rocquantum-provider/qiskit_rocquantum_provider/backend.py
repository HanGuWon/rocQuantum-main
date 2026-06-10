import cmath
import uuid

from qiskit.providers import BackendV2, Options
from qiskit.transpiler import Target
from qiskit.result import Result
try:
    from qiskit.result import ExperimentResult, ExperimentResultData
except ImportError:
    from qiskit.result.models import ExperimentResult, ExperimentResultData
from qiskit.circuit import Measure
from qiskit.circuit.library import (
    CCXGate,
    CCZGate,
    CPhaseGate,
    CSwapGate,
    CXGate,
    CZGate,
    CHGate,
    CYGate,
    CRXGate,
    CRYGate,
    CRZGate,
    DCXGate,
    ECRGate,
    HGate,
    IGate,
    RXGate,
    RYGate,
    RZGate,
    RXXGate,
    RYYGate,
    RZZGate,
    RCCXGate,
    RC3XGate,
    SGate,
    SdgGate,
    SXGate,
    SXdgGate,
    SwapGate,
    TGate,
    PhaseGate,
    UGate,
    UnitaryGate,
    XGate,
    YGate,
    ZGate,
    iSwapGate,
)

from rocquantum.framework_runtime import (
    RocQuantumRuntime,
    counts_from_memory,
    normalize_params,
    qiskit_memory_from_samples,
)

from .estimator import estimate_pauli_observable
from .job import RocQuantumJob


MATRIX_FALLBACK_OPS = {
    "ccx", "ccz", "ch", "cp", "crx", "cry", "crz", "cswap", "cy",
    "dcx", "ecr", "iswap",
    "p", "rccx", "rcccx", "rxx", "ryy", "rzz",
    "sx", "sxdg", "u", "unitary",
}


def _instruction_target(num_qubits):
    target = Target(num_qubits=int(num_qubits))
    target.add_instruction(CCXGate(), name="ccx")
    target.add_instruction(CCZGate(), name="ccz")
    target.add_instruction(CHGate(), name="ch")
    target.add_instruction(CPhaseGate(0.0), name="cp")
    target.add_instruction(CSwapGate(), name="cswap")
    target.add_instruction(CXGate(), name="cx")
    target.add_instruction(CZGate(), name="cz")
    target.add_instruction(CYGate(), name="cy")
    target.add_instruction(CRXGate(0.0), name="crx")
    target.add_instruction(CRYGate(0.0), name="cry")
    target.add_instruction(CRZGate(0.0), name="crz")
    target.add_instruction(DCXGate(), name="dcx")
    target.add_instruction(ECRGate(), name="ecr")
    target.add_instruction(HGate(), name="h")
    target.add_instruction(IGate(), name="id")
    target.add_instruction(iSwapGate(), name="iswap")
    target.add_instruction(PhaseGate(0.0), name="p")
    target.add_instruction(RCCXGate(), name="rccx")
    target.add_instruction(RC3XGate(), name="rcccx")
    target.add_instruction(RXGate(0.0), name="rx")
    target.add_instruction(RYGate(0.0), name="ry")
    target.add_instruction(RZGate(0.0), name="rz")
    target.add_instruction(RXXGate(0.0), name="rxx")
    target.add_instruction(RYYGate(0.0), name="ryy")
    target.add_instruction(RZZGate(0.0), name="rzz")
    target.add_instruction(SGate(), name="s")
    target.add_instruction(SdgGate(), name="sdg")
    target.add_instruction(SXGate(), name="sx")
    target.add_instruction(SXdgGate(), name="sxdg")
    target.add_instruction(SwapGate(), name="swap")
    target.add_instruction(TGate(), name="t")
    target.add_instruction(UGate(0.0, 0.0, 0.0), name="u")
    target.add_instruction(UnitaryGate([[1, 0], [0, 1]]), name="unitary")
    target.add_instruction(XGate(), name="x")
    target.add_instruction(YGate(), name="y")
    target.add_instruction(ZGate(), name="z")
    target.add_instruction(Measure(), name="measure")
    return target

class RocQuantumBackend(BackendV2):
    """
    rocQuantum Qiskit Backend.

    A Qiskit backend that interfaces with the rocQuantum C++/HIP simulator.
    """
    def __init__(self, provider=None, **kwargs):
        max_target_qubits = int(kwargs.pop("max_target_qubits", 64))
        super().__init__(provider=provider, name="rocq_simulator", **kwargs)

        # The simulator is instantiated once and maintained for the lifetime of the backend
        # We assume a fixed, maximum number of qubits or re-instantiate if needed.
        # For simplicity, let's assume it's configured on first use.
        self._runtime = None
        self._num_qubits = 0

        self._target = _instruction_target(max_target_qubits)

    @classmethod
    def _default_options(cls):
        return Options(shots=1024, memory=True, statevector=True, sampling=True)

    @property
    def target(self):
        return self._target

    @property
    def max_circuits(self):
        return None

    def _ensure_simulator(self, num_qubits):
        """Create or reset the simulator if the qubit count changes."""
        if self._runtime is None or self._num_qubits != num_qubits:
            self._runtime = RocQuantumRuntime.from_bindings(num_qubits)
            self._num_qubits = num_qubits
        else:
            self._runtime.reset()

    def _apply_global_phase(self, circuit):
        if circuit.num_qubits == 0:
            return

        phase = float(getattr(circuit, "global_phase", 0.0))
        if abs(phase) <= 1e-15:
            return

        phase_factor = cmath.exp(1j * phase)
        self._runtime.apply_operation(
            "unitary",
            [0],
            matrix=[
                [phase_factor, 0.0],
                [0.0, phase_factor],
            ],
        )

    def _apply_circuit(self, circuit, *, include_global_phase: bool = False):
        self._ensure_simulator(circuit.num_qubits)
        if include_global_phase:
            self._apply_global_phase(circuit)

        measured_bits = {}  # Map classical bit index to qubit index
        measurement_started = False
        for instruction in circuit.data:
            op = instruction.operation
            if op.name in {"barrier", "delay", "save_statevector"}:
                continue

            q_indices = [circuit.find_bit(q).index for q in instruction.qubits]

            if op.name == 'measure':
                measurement_started = True
                # Store which classical bit this measurement corresponds to
                c_index = circuit.find_bit(instruction.clbits[0]).index
                measured_bits[c_index] = q_indices[0]
                continue

            if measurement_started:
                raise ValueError(
                    "RocQuantumBackend only supports terminal measurements. "
                    f"Operation {op.name!r} appears after a measurement."
                )

            matrix = op.to_matrix() if op.name in MATRIX_FALLBACK_OPS else None
            self._runtime.apply_operation(
                op.name,
                q_indices,
                normalize_params(op.params),
                matrix=matrix,
            )

        return measured_bits

    @staticmethod
    def _requests_statevector(circuit):
        return any(instruction.operation.name == "save_statevector" for instruction in circuit.data)

    def estimate_expectation(self, circuit, observable):
        """Return a native Pauli expectation for a circuit and Qiskit observable."""
        self._apply_circuit(circuit, include_global_phase=False)
        return estimate_pauli_observable(self._runtime, observable, circuit.num_qubits)

    def run(self, run_input, **options):
        if not isinstance(run_input, list):
            run_input = [run_input]

        job_id = str(uuid.uuid4())
        shots = int(options.get("shots", self.options.shots))
        results = []

        for circuit in run_input:
            return_statevector = bool(
                options.get("statevector", self.options.statevector)
                or self._requests_statevector(circuit)
            )
            measured_bits = self._apply_circuit(circuit, include_global_phase=return_statevector)

            # Perform measurement on the required qubits.  rocsvSample packs
            # result bits in the order qubits are requested, so keep the
            # classical-bit mapping explicit when formatting Qiskit memory.
            if measured_bits:
                measured_items = sorted(measured_bits.items())
            else: # If no measure op, measure all into matching bit positions.
                measured_items = [(idx, idx) for idx in range(circuit.num_qubits)]
            qubits_to_measure = [qubit for _, qubit in measured_items]

            statevector = self._runtime.statevector() if return_statevector else None

            return_sampling = bool(options.get("sampling", self.options.sampling))
            data = {}
            result_shots = 0
            if return_sampling:
                raw_samples = self._runtime.measure(qubits_to_measure, shots)
                memory_width = getattr(circuit, "num_clbits", 0) or len(measured_items)
                memory = qiskit_memory_from_samples(raw_samples, measured_items, memory_width)
                formatted_counts = counts_from_memory(memory)
                data.update(
                    {
                        "counts": formatted_counts,
                        "memory": memory if options.get("memory", self.options.memory) else None,
                    }
                )
                result_shots = shots
            if return_statevector:
                data["statevector"] = statevector

            exp_data = ExperimentResultData(
                **data,
            )

            exp_result = ExperimentResult(
                shots=result_shots,
                success=True,
                data=exp_data,
                header=getattr(
                    circuit,
                    "header",
                    {"name": getattr(circuit, "name", None), "metadata": getattr(circuit, "metadata", None)},
                ),
            )
            results.append(exp_result)

        result = Result(
            backend_name=self.name,
            backend_version=getattr(self, "backend_version", "0.1.0"),
            job_id=job_id,
            qobj_id=None,
            success=True,
            results=results,
        )
        return RocQuantumJob(self, job_id, result)
