import cmath
import uuid

from qiskit.providers import BackendV2, Options
from qiskit.transpiler import Target
from qiskit.result import Result
try:
    from qiskit.result import ExperimentResult, ExperimentResultData
except ImportError:
    from qiskit.result.models import ExperimentResult, ExperimentResultData
from qiskit.circuit import Measure, Reset
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
    MCXGate,
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
    StatePreparation,
    TGate,
    TdgGate,
    PhaseGate,
    UGate,
    UnitaryGate,
    XGate,
    YGate,
    ZGate,
    iSwapGate,
)
from qiskit.quantum_info import Operator

from rocquantum.framework_runtime import (
    RocQuantumRuntime,
    counts_from_memory,
    normalize_params,
    qiskit_memory_from_samples,
    qiskit_sample_plan,
)

from .estimator import estimate_pauli_observable
from .job import RocQuantumJob


MATRIX_FALLBACK_OPS = {
    "ccx", "crx", "cry", "crz", "cswap",
    "state_preparation", "unitary",
}
MAX_AUTOMATIC_MATRIX_FALLBACK_QUBITS = 4
CONTROL_FLOW_OPS = {
    "break_loop", "continue_loop", "for_loop", "if_else", "switch_case", "while_loop",
}


def _instruction_condition(instruction):
    operation_condition = getattr(instruction.operation, "condition", None)
    if operation_condition is not None:
        return operation_condition
    return getattr(instruction, "condition", None)


def _state_preparation_matrix(op):
    if op.name == "initialize":
        return Operator(StatePreparation(op.params)).data
    return Operator(op).data


def _automatic_operation_matrix(op):
    if int(getattr(op, "num_qubits", 0)) > MAX_AUTOMATIC_MATRIX_FALLBACK_QUBITS:
        return None
    try:
        return op.to_matrix()
    except Exception:
        try:
            return Operator(op).data
        except Exception:
            return None


def _operation_matrix(op):
    if op.name == "state_preparation":
        return _state_preparation_matrix(op)
    if op.name in MATRIX_FALLBACK_OPS:
        return op.to_matrix()
    return _automatic_operation_matrix(op)


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
    target.add_instruction(MCXGate(3), name="mcx")
    target.add_instruction(PhaseGate(0.0), name="p")
    target.add_instruction(RCCXGate(), name="rccx")
    target.add_instruction(RC3XGate(), name="rcccx")
    target.add_instruction(RXGate(0.0), name="rx")
    target.add_instruction(RYGate(0.0), name="ry")
    target.add_instruction(RZGate(0.0), name="rz")
    target.add_instruction(Reset(), name="reset")
    target.add_instruction(RXXGate(0.0), name="rxx")
    target.add_instruction(RYYGate(0.0), name="ryy")
    target.add_instruction(RZZGate(0.0), name="rzz")
    target.add_instruction(SGate(), name="s")
    target.add_instruction(SdgGate(), name="sdg")
    target.add_instruction(SXGate(), name="sx")
    target.add_instruction(SXdgGate(), name="sxdg")
    target.add_instruction(SwapGate(), name="swap")
    target.add_instruction(TGate(), name="t")
    target.add_instruction(TdgGate(), name="tdg")
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
        return Options(shots=1024, memory=True, statevector=False, sampling=True)

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

    def _supports_native_phase_decomposition(self, include_global_phase):
        simulator = self._runtime.simulator
        has_gate_dispatch = callable(getattr(simulator, "apply_gate", None))
        has_matrix_dispatch = callable(getattr(simulator, "apply_matrix", None)) or callable(
            getattr(simulator, "ApplyGate", None)
        )
        return has_gate_dispatch and (has_matrix_dispatch or not include_global_phase)

    def _supports_native_parametric_decomposition(self):
        return callable(getattr(self._runtime.simulator, "apply_gate", None))

    def _apply_global_phase_value(self, phase, target=0):
        if abs(float(phase)) <= 1e-15:
            return

        phase_factor = cmath.exp(1j * float(phase))
        self._runtime.apply_operation(
            "unitary",
            [target],
            matrix=[
                [phase_factor, 0.0],
                [0.0, phase_factor],
            ],
        )

    def _apply_global_phase(self, circuit):
        if circuit.num_qubits == 0:
            return

        phase = float(getattr(circuit, "global_phase", 0.0))
        self._apply_global_phase_value(phase)

    def _apply_phase_gate(self, q_indices, theta, *, include_global_phase):
        if len(q_indices) != 1:
            raise ValueError("Qiskit p gate requires exactly one qubit.")

        target = q_indices[0]
        if include_global_phase:
            self._apply_global_phase_value(0.5 * theta, target=target)
        self._runtime.apply_operation("rz", [target], [theta])

    def _apply_controlled_phase_gate(self, q_indices, theta, *, include_global_phase):
        if len(q_indices) != 2:
            raise ValueError("Qiskit cp gate requires exactly two qubits.")

        control, target = q_indices
        if include_global_phase:
            self._apply_global_phase_value(0.25 * theta, target=control)
        self._runtime.apply_operation("rz", [control], [0.5 * theta])
        self._runtime.apply_operation("cx", [control, target])
        self._runtime.apply_operation("rz", [target], [-0.5 * theta])
        self._runtime.apply_operation("cx", [control, target])
        self._runtime.apply_operation("rz", [target], [0.5 * theta])

    def _apply_rzz_gate(self, q_indices, theta):
        if len(q_indices) != 2:
            raise ValueError("Qiskit rzz gate requires exactly two qubits.")

        control, target = q_indices
        self._runtime.apply_operation("cx", [control, target])
        self._runtime.apply_operation("rz", [target], [theta])
        self._runtime.apply_operation("cx", [control, target])

    def _apply_rxx_gate(self, q_indices, theta):
        if len(q_indices) != 2:
            raise ValueError("Qiskit rxx gate requires exactly two qubits.")

        control, target = q_indices
        self._runtime.apply_operation("cx", [control, target])
        self._runtime.apply_operation("rx", [control], [theta])
        self._runtime.apply_operation("cx", [control, target])

    def _apply_ryy_gate(self, q_indices, theta):
        if len(q_indices) != 2:
            raise ValueError("Qiskit ryy gate requires exactly two qubits.")

        for qubit in q_indices:
            self._runtime.apply_operation("rx", [qubit], [cmath.pi / 2])
        self._apply_rzz_gate(q_indices, theta)
        for qubit in q_indices:
            self._runtime.apply_operation("rx", [qubit], [-cmath.pi / 2])

    def _apply_sx_gate(self, q_indices, *, inverse=False, include_global_phase):
        if len(q_indices) != 1:
            raise ValueError("Qiskit sx gate requires exactly one qubit.")

        target = q_indices[0]
        sign = -1.0 if inverse else 1.0
        if include_global_phase:
            self._apply_global_phase_value(sign * cmath.pi / 4, target=target)
        self._runtime.apply_operation("rx", [target], [sign * cmath.pi / 2])

    def _apply_u_gate(self, q_indices, theta, phi, lam, *, include_global_phase):
        if len(q_indices) != 1:
            raise ValueError("Qiskit u gate requires exactly one qubit.")

        target = q_indices[0]
        if include_global_phase:
            self._apply_global_phase_value(0.5 * (phi + lam), target=target)
        self._runtime.apply_operation("rz", [target], [lam])
        self._runtime.apply_operation("ry", [target], [theta])
        self._runtime.apply_operation("rz", [target], [phi])

    def _apply_cy_gate(self, q_indices):
        if len(q_indices) != 2:
            raise ValueError("Qiskit cy gate requires exactly two qubits.")

        control, target = q_indices
        self._runtime.apply_operation("sdg", [target])
        self._runtime.apply_operation("cx", [control, target])
        self._runtime.apply_operation("s", [target])

    def _apply_ccz_gate(self, q_indices):
        if len(q_indices) != 3:
            raise ValueError("Qiskit ccz gate requires exactly three qubits.")

        control_a, control_b, target = q_indices
        self._runtime.apply_operation("h", [target])
        self._runtime.apply_operation("mcx", [control_a, control_b, target])
        self._runtime.apply_operation("h", [target])

    def _apply_ch_gate(self, q_indices):
        if len(q_indices) != 2:
            raise ValueError("Qiskit ch gate requires exactly two qubits.")

        control, target = q_indices
        self._runtime.apply_operation("ry", [target], [cmath.pi / 4])
        self._runtime.apply_operation("cx", [control, target])
        self._runtime.apply_operation("ry", [target], [-cmath.pi / 4])

    def _apply_dcx_gate(self, q_indices):
        if len(q_indices) != 2:
            raise ValueError("Qiskit dcx gate requires exactly two qubits.")

        control, target = q_indices
        self._runtime.apply_operation("cx", [control, target])
        self._runtime.apply_operation("cx", [target, control])

    def _apply_iswap_gate(self, q_indices):
        if len(q_indices) != 2:
            raise ValueError("Qiskit iswap gate requires exactly two qubits.")

        left, right = q_indices
        self._runtime.apply_operation("s", [left])
        self._runtime.apply_operation("s", [right])
        self._runtime.apply_operation("h", [left])
        self._runtime.apply_operation("cx", [left, right])
        self._runtime.apply_operation("cx", [right, left])
        self._runtime.apply_operation("h", [right])

    def _apply_ecr_gate(self, q_indices):
        if len(q_indices) != 2:
            raise ValueError("Qiskit ecr gate requires exactly two qubits.")

        control, target = q_indices
        self._runtime.apply_operation("s", [control])
        self._runtime.apply_operation("rx", [target], [cmath.pi / 2])
        self._runtime.apply_operation("cx", [control, target])
        self._runtime.apply_operation("x", [control])

    def _apply_tdg_gate(self, q_indices, include_global_phase):
        if len(q_indices) != 1:
            raise ValueError("Qiskit tdg gate requires exactly one qubit.")

        target = q_indices[0]
        if include_global_phase:
            self._apply_global_phase_value(-cmath.pi / 8, target=target)
        self._runtime.apply_operation("rz", [target], [-cmath.pi / 4])

    def _apply_rccx_gate(self, q_indices):
        if len(q_indices) != 3:
            raise ValueError("Qiskit rccx gate requires exactly three qubits.")

        control_a, control_b, target = q_indices
        quarter_pi = cmath.pi / 4
        self._runtime.apply_operation("h", [target])
        self._runtime.apply_operation("rz", [target], [quarter_pi])
        self._runtime.apply_operation("cx", [control_b, target])
        self._runtime.apply_operation("rz", [target], [-quarter_pi])
        self._runtime.apply_operation("cx", [control_a, target])
        self._runtime.apply_operation("rz", [target], [quarter_pi])
        self._runtime.apply_operation("cx", [control_b, target])
        self._runtime.apply_operation("rz", [target], [-quarter_pi])
        self._runtime.apply_operation("h", [target])

    def _apply_rcccx_gate(self, q_indices):
        if len(q_indices) != 4:
            raise ValueError("Qiskit rcccx gate requires exactly four qubits.")

        control_a, control_b, control_c, target = q_indices
        quarter_pi = cmath.pi / 4
        self._runtime.apply_operation("h", [target])
        self._runtime.apply_operation("rz", [target], [quarter_pi])
        self._runtime.apply_operation("cx", [control_c, target])
        self._runtime.apply_operation("rz", [target], [-quarter_pi])
        self._runtime.apply_operation("h", [target])
        self._runtime.apply_operation("cx", [control_a, target])
        self._runtime.apply_operation("rz", [target], [quarter_pi])
        self._runtime.apply_operation("cx", [control_b, target])
        self._runtime.apply_operation("rz", [target], [-quarter_pi])
        self._runtime.apply_operation("cx", [control_a, target])
        self._runtime.apply_operation("rz", [target], [quarter_pi])
        self._runtime.apply_operation("cx", [control_b, target])
        self._runtime.apply_operation("rz", [target], [-quarter_pi])
        self._runtime.apply_operation("h", [target])
        self._runtime.apply_operation("rz", [target], [quarter_pi])
        self._runtime.apply_operation("cx", [control_c, target])
        self._runtime.apply_operation("rz", [target], [-quarter_pi])
        self._runtime.apply_operation("h", [target])

    def _apply_controlled_base_gate(self, op, q_indices):
        num_controls = int(getattr(op, "num_ctrl_qubits", 0))
        base_gate = getattr(op, "base_gate", None)
        base_name = getattr(base_gate, "name", None)
        if num_controls < 1 or len(q_indices) != num_controls + 1:
            return False
        if base_name not in {"x", "h"}:
            return False
        if base_name == "h" and num_controls != 1:
            return False

        ctrl_state = getattr(op, "ctrl_state", None)
        ctrl_state = (1 << num_controls) - 1 if ctrl_state is None else int(ctrl_state)
        controls = list(q_indices[:num_controls])
        target = q_indices[num_controls]
        open_controls = [
            control
            for control_index, control in enumerate(controls)
            if not ((ctrl_state >> control_index) & 1)
        ]

        flipped = []
        try:
            for control in open_controls:
                self._runtime.apply_operation("x", [control])
                flipped.append(control)
            if base_name == "x":
                if len(controls) == 1:
                    self._runtime.apply_operation("cx", [controls[0], target])
                else:
                    self._runtime.apply_operation("mcx", controls + [target])
            else:
                self._apply_ch_gate([controls[0], target])
            for control in reversed(flipped):
                self._runtime.apply_operation("x", [control])
            flipped.clear()
        finally:
            for control in reversed(flipped):
                self._runtime.apply_operation("x", [control])
        return True

    def _apply_circuit(self, circuit, *, include_global_phase: bool = False):
        self._ensure_simulator(circuit.num_qubits)
        if include_global_phase:
            self._apply_global_phase(circuit)

        measured_bits = {}  # Map classical bit index to qubit index
        measurement_started = False
        touched_qubits = set()
        for instruction in circuit.data:
            op = instruction.operation
            if op.name in {"barrier", "delay", "save_statevector"}:
                continue

            if op.name in CONTROL_FLOW_OPS or getattr(op, "blocks", None) is not None:
                raise ValueError(
                    "RocQuantumBackend does not support Qiskit control-flow operations yet. "
                    "Transpile or decompose dynamic circuits before execution."
                )

            if _instruction_condition(instruction) is not None:
                raise ValueError(
                    "RocQuantumBackend does not support classically conditioned operations yet. "
                    "Transpile or decompose dynamic circuits before execution."
                )

            q_indices = [circuit.find_bit(q).index for q in instruction.qubits]

            if measurement_started and op.name != "measure":
                raise ValueError(
                    "RocQuantumBackend only supports terminal measurements. "
                    f"Operation {op.name!r} appears after a measurement."
                )

            if op.name == "initialize":
                reset_targets = set(q_indices)
                if touched_qubits & reset_targets:
                    raise ValueError(
                        "RocQuantumBackend only supports initialize before a qubit has been operated on. "
                        "Later initialize instructions require non-unitary reset support."
                    )
                self._runtime.apply_operation(
                    "state_preparation",
                    q_indices,
                    matrix=_state_preparation_matrix(op),
                )
                touched_qubits.update(q_indices)
                continue

            if op.name == "reset":
                reset_targets = set(q_indices)
                if touched_qubits & reset_targets:
                    raise ValueError(
                        "RocQuantumBackend only supports reset before a qubit has been operated on. "
                        "Mid-circuit reset requires non-unitary state reinitialization support."
                    )
                continue

            if op.name == 'measure':
                measurement_started = True
                # Store which classical bit this measurement corresponds to
                c_index = circuit.find_bit(instruction.clbits[0]).index
                measured_bits[c_index] = q_indices[0]
                continue

            if op.name in {"p", "cp"} and self._supports_native_phase_decomposition(include_global_phase):
                (theta,) = normalize_params(op.params)
                if op.name == "p":
                    self._apply_phase_gate(q_indices, theta, include_global_phase=include_global_phase)
                else:
                    self._apply_controlled_phase_gate(q_indices, theta, include_global_phase=include_global_phase)
                touched_qubits.update(q_indices)
                continue

            if op.name == "tdg" and self._supports_native_phase_decomposition(include_global_phase):
                self._apply_tdg_gate(q_indices, include_global_phase=include_global_phase)
                touched_qubits.update(q_indices)
                continue

            if op.name in {"rxx", "ryy", "rzz"} and self._supports_native_parametric_decomposition():
                (theta,) = normalize_params(op.params)
                if op.name == "rxx":
                    self._apply_rxx_gate(q_indices, theta)
                elif op.name == "ryy":
                    self._apply_ryy_gate(q_indices, theta)
                else:
                    self._apply_rzz_gate(q_indices, theta)
                touched_qubits.update(q_indices)
                continue

            if op.name in {"sx", "sxdg", "u"} and self._supports_native_phase_decomposition(include_global_phase):
                params = normalize_params(op.params)
                if op.name == "sx":
                    self._apply_sx_gate(q_indices, inverse=False, include_global_phase=include_global_phase)
                elif op.name == "sxdg":
                    self._apply_sx_gate(q_indices, inverse=True, include_global_phase=include_global_phase)
                else:
                    theta, phi, lam = params
                    self._apply_u_gate(q_indices, theta, phi, lam, include_global_phase=include_global_phase)
                touched_qubits.update(q_indices)
                continue

            if op.name in {
                "ch", "cy", "ccz", "dcx", "ecr", "iswap", "rccx", "rcccx"
            } and self._supports_native_parametric_decomposition():
                if op.name == "ch":
                    self._apply_ch_gate(q_indices)
                elif op.name == "cy":
                    self._apply_cy_gate(q_indices)
                elif op.name == "ccz":
                    self._apply_ccz_gate(q_indices)
                elif op.name == "dcx":
                    self._apply_dcx_gate(q_indices)
                elif op.name == "ecr":
                    self._apply_ecr_gate(q_indices)
                elif op.name == "iswap":
                    self._apply_iswap_gate(q_indices)
                elif op.name == "rccx":
                    self._apply_rccx_gate(q_indices)
                else:
                    self._apply_rcccx_gate(q_indices)
                touched_qubits.update(q_indices)
                continue

            if self._supports_native_parametric_decomposition():
                try:
                    if self._apply_controlled_base_gate(op, q_indices):
                        touched_qubits.update(q_indices)
                        continue
                except (NotImplementedError, RuntimeError, TypeError, ValueError):
                    pass

            matrix = _operation_matrix(op)
            if matrix is not None and op.name in {"state_preparation", "unitary"}:
                params = []
            else:
                params = normalize_params(op.params)
            self._runtime.apply_operation(
                op.name,
                q_indices,
                params,
                matrix=matrix,
            )
            touched_qubits.update(q_indices)

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
            qubits_to_measure, sample_offsets = qiskit_sample_plan(measured_items)

            statevector = self._runtime.statevector() if return_statevector else None

            return_sampling = bool(options.get("sampling", self.options.sampling))
            data = {}
            result_shots = 0
            if return_sampling:
                raw_samples = self._runtime.measure(qubits_to_measure, shots)
                memory_width = getattr(circuit, "num_clbits", 0) or len(measured_items)
                memory = qiskit_memory_from_samples(raw_samples, measured_items, memory_width, sample_offsets)
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
