import cmath
from itertools import combinations
import uuid

import numpy as np

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
    CSGate,
    CSdgGate,
    CSXGate,
    CSwapGate,
    CXGate,
    CZGate,
    CU1Gate,
    CU3Gate,
    CUGate,
    CHGate,
    CYGate,
    CRXGate,
    CRYGate,
    CRZGate,
    DCXGate,
    ECRGate,
    GlobalPhaseGate,
    HGate,
    IGate,
    MCPhaseGate,
    MCXGate,
    RXGate,
    RYGate,
    RZGate,
    RZXGate,
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
    U1Gate,
    U2Gate,
    U3Gate,
    PhaseGate,
    RGate,
    UGate,
    UnitaryGate,
    XGate,
    XXMinusYYGate,
    XXPlusYYGate,
    YGate,
    ZGate,
    iSwapGate,
)
from qiskit.quantum_info import Operator

from rocquantum.framework_runtime import (
    GATE_ALIASES,
    RocQuantumRuntime,
    counts_from_memory,
    normalize_gate_name,
    normalize_params,
    qiskit_memory_from_samples,
    qiskit_sample_plan,
    statevector_to_little_endian_wires,
)

from .estimator import estimate_observable
from .job import RocQuantumJob


MATRIX_FALLBACK_OPS = {
    "ccx", "crx", "cry", "crz", "cswap",
    "state_preparation", "unitary",
}
MAX_AUTOMATIC_MATRIX_FALLBACK_QUBITS = 4
CONTROL_FLOW_OPS = {
    "break_loop", "continue_loop", "for_loop", "if_else", "switch_case", "while_loop",
}
DEFAULT_MAX_DYNAMIC_LOOP_ITERATIONS = 1024


def _instruction_condition(instruction):
    operation_condition = getattr(instruction.operation, "condition", None)
    if operation_condition is not None:
        return operation_condition
    return getattr(instruction, "condition", None)


def _condition_matches(condition, circuit, classical_bits):
    if condition is None:
        return True
    if not isinstance(condition, tuple) or len(condition) != 2:
        raise NotImplementedError("RocQuantumBackend only supports tuple-style Qiskit conditions.")

    condition_bits, expected = condition
    if isinstance(expected, bool):
        expected_value = int(expected)
    else:
        expected_value = int(expected)

    return _classical_value(condition_bits, circuit, classical_bits) == expected_value


def _classical_value(bits_or_register, circuit, classical_bits):
    try:
        bit_index = circuit.find_bit(bits_or_register).index
    except Exception:
        bit_index = None

    if bit_index is not None:
        return int(classical_bits.get(bit_index, 0))

    try:
        bits = list(bits_or_register)
    except TypeError as exc:
        raise NotImplementedError("Unsupported Qiskit condition bit container.") from exc

    actual_value = 0
    for offset, bit in enumerate(bits):
        actual_value |= int(classical_bits.get(circuit.find_bit(bit).index, 0)) << offset
    return actual_value


def _is_default_switch_case(label):
    return str(label) == "<default case>"


def _for_loop_metadata(op):
    params = list(getattr(op, "params", ()) or ())
    if len(params) < 2:
        raise NotImplementedError("RocQuantumBackend requires Qiskit for_loop indexset metadata.")
    indexset, loop_parameter = params[0], params[1]
    try:
        loop_values = list(indexset)
    except TypeError as exc:
        raise NotImplementedError("RocQuantumBackend requires a finite static for_loop indexset.") from exc
    return loop_values, loop_parameter


def _bind_for_loop_block(block, loop_parameter, value):
    if loop_parameter is None:
        return block
    try:
        return block.assign_parameters({loop_parameter: value}, inplace=False)
    except Exception as exc:
        raise NotImplementedError(
            "RocQuantumBackend could not bind a Qiskit for_loop parameter for dynamic sampling."
        ) from exc


def _memory_from_classical_bits(classical_bits, measured_items, memory_width):
    if memory_width <= 0:
        memory_width = len(measured_items)
    bits = ["0"] * memory_width
    for classical_bit, _ in measured_items:
        output_index = memory_width - 1 - int(classical_bit)
        if 0 <= output_index < memory_width:
            bits[output_index] = "1" if int(classical_bits.get(int(classical_bit), 0)) else "0"
    return "".join(bits)


def _state_preparation_matrix(op):
    if op.name == "initialize":
        return Operator(StatePreparation(op.params)).data
    return Operator(op).data


def _state_preparation_vector(op):
    if op.name == "initialize":
        return StatePreparation(op.params).params
    return getattr(op, "params", None)


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


def _operation_runtime_params(op, matrix):
    try:
        return normalize_params(op.params)
    except (TypeError, ValueError):
        if matrix is not None:
            return []
        raise


def _parameter_value_matches(left, right):
    try:
        left_array = np.asarray(left)
        right_array = np.asarray(right)
        if left_array.shape or right_array.shape:
            return left_array.shape == right_array.shape and np.allclose(left_array, right_array)
    except (TypeError, ValueError):
        pass

    comparison = left == right
    if isinstance(comparison, np.ndarray):
        return bool(np.all(comparison))
    return bool(comparison)


def _parameter_lists_match(left, right):
    if len(left) != len(right):
        return False
    return all(_parameter_value_matches(left_value, right_value) for left_value, right_value in zip(left, right))


def _pauli_labels_commute(left, right):
    anti_commuting_positions = 0
    for left_pauli, right_pauli in zip(str(left).upper(), str(right).upper()):
        if left_pauli == "I" or right_pauli == "I" or left_pauli == right_pauli:
            continue
        anti_commuting_positions += 1
    return anti_commuting_positions % 2 == 0


def _pauli_evolution_terms(op):
    operator = getattr(op, "operator", None)
    if operator is None or isinstance(operator, list):
        return None

    if hasattr(operator, "to_list"):
        terms = [(str(label).upper(), complex(coeff)) for label, coeff in operator.to_list()]
    elif hasattr(operator, "to_label"):
        terms = [(str(operator.to_label()).upper(), 1.0 + 0.0j)]
    else:
        return None

    if not terms:
        return []

    label_length = len(terms[0][0])
    if any(len(label) != label_length for label, _ in terms):
        return None
    if any(abs(complex(coeff).imag) > 1e-12 for _, coeff in terms):
        return None

    for index, (left_label, _) in enumerate(terms):
        for right_label, _ in terms[index + 1:]:
            if not _pauli_labels_commute(left_label, right_label):
                return None

    return terms


def _instruction_target(num_qubits):
    target = Target(num_qubits=int(num_qubits))
    target.add_instruction(CCXGate(), name="ccx")
    target.add_instruction(CCZGate(), name="ccz")
    target.add_instruction(CHGate(), name="ch")
    target.add_instruction(HGate().control(2, annotated=False), name="cch")
    target.add_instruction(HGate().control(3, annotated=False), name="c3h")
    target.add_instruction(CPhaseGate(0.0), name="cp")
    target.add_instruction(CSGate(), name="cs")
    target.add_instruction(SGate().control(2, annotated=False), name="ccs")
    target.add_instruction(SGate().control(3, annotated=False), name="c3s")
    target.add_instruction(CSdgGate(), name="csdg")
    target.add_instruction(SdgGate().control(2, annotated=False), name="ccsdg")
    target.add_instruction(SdgGate().control(3, annotated=False), name="c3sdg")
    target.add_instruction(CSXGate(), name="csx")
    target.add_instruction(SXGate().control(2, annotated=False), name="ccsx")
    target.add_instruction(SXGate().control(3, annotated=False), name="c3sx")
    target.add_instruction(TGate().control(1, annotated=False), name="ct")
    target.add_instruction(TGate().control(2, annotated=False), name="cct")
    target.add_instruction(TGate().control(3, annotated=False), name="c3t")
    target.add_instruction(TdgGate().control(1, annotated=False), name="ctdg")
    target.add_instruction(TdgGate().control(2, annotated=False), name="cctdg")
    target.add_instruction(TdgGate().control(3, annotated=False), name="c3tdg")
    target.add_instruction(CSwapGate(), name="cswap")
    target.add_instruction(CXGate(), name="cx")
    target.add_instruction(CZGate(), name="cz")
    target.add_instruction(CU1Gate(0.0), name="cu1")
    target.add_instruction(CU3Gate(0.0, 0.0, 0.0), name="cu3")
    target.add_instruction(CUGate(0.0, 0.0, 0.0, 0.0), name="cu")
    target.add_instruction(CYGate(), name="cy")
    target.add_instruction(YGate().control(2, annotated=False), name="ccy")
    target.add_instruction(YGate().control(3, annotated=False), name="c3y")
    target.add_instruction(ZGate().control(3, annotated=False), name="c3z")
    target.add_instruction(CRXGate(0.0), name="crx")
    target.add_instruction(RXGate(0.0).control(2, annotated=False), name="ccrx")
    target.add_instruction(RXGate(0.0).control(3, annotated=False), name="c3rx")
    target.add_instruction(CRYGate(0.0), name="cry")
    target.add_instruction(RYGate(0.0).control(2, annotated=False), name="ccry")
    target.add_instruction(RYGate(0.0).control(3, annotated=False), name="c3ry")
    target.add_instruction(CRZGate(0.0), name="crz")
    target.add_instruction(RZGate(0.0).control(2, annotated=False), name="ccrz")
    target.add_instruction(RZGate(0.0).control(3, annotated=False), name="c3rz")
    target.add_instruction(RGate(0.0, 0.0).control(1, annotated=False), name="cr")
    target.add_instruction(RGate(0.0, 0.0).control(2, annotated=False), name="ccr")
    target.add_instruction(RGate(0.0, 0.0).control(3, annotated=False), name="c3r")
    target.add_instruction(DCXGate(), name="dcx")
    target.add_instruction(ECRGate(), name="ecr")
    target.add_instruction(GlobalPhaseGate(0.0), name="global_phase")
    target.add_instruction(HGate(), name="h")
    target.add_instruction(IGate(), name="id")
    target.add_instruction(iSwapGate(), name="iswap")
    target.add_instruction(MCPhaseGate(0.0, 2), name="mcphase")
    target.add_instruction(MCXGate(3), name="mcx")
    target.add_instruction(PhaseGate(0.0), name="p")
    target.add_instruction(RGate(0.0, 0.0), name="r")
    target.add_instruction(RCCXGate(), name="rccx")
    target.add_instruction(RC3XGate(), name="rcccx")
    target.add_instruction(RXGate(0.0), name="rx")
    target.add_instruction(RYGate(0.0), name="ry")
    target.add_instruction(RZGate(0.0), name="rz")
    target.add_instruction(RZXGate(0.0), name="rzx")
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
    target.add_instruction(U1Gate(0.0), name="u1")
    target.add_instruction(U1Gate(0.0).control(2, annotated=False), name="mcu1")
    target.add_instruction(U2Gate(0.0, 0.0), name="u2")
    target.add_instruction(U2Gate(0.0, 0.0).control(1, annotated=False), name="cu2")
    target.add_instruction(U2Gate(0.0, 0.0).control(2, annotated=False), name="ccu2")
    target.add_instruction(U2Gate(0.0, 0.0).control(3, annotated=False), name="c3u2")
    target.add_instruction(U3Gate(0.0, 0.0, 0.0), name="u3")
    target.add_instruction(U3Gate(0.0, 0.0, 0.0).control(2, annotated=False), name="ccu3")
    target.add_instruction(U3Gate(0.0, 0.0, 0.0).control(3, annotated=False), name="c3u3")
    target.add_instruction(UGate(0.0, 0.0, 0.0), name="u")
    target.add_instruction(UnitaryGate([[1, 0], [0, 1]]), name="unitary")
    target.add_instruction(XGate(), name="x")
    target.add_instruction(XXMinusYYGate(0.0, 0.0), name="xx_minus_yy")
    target.add_instruction(XXPlusYYGate(0.0, 0.0), name="xx_plus_yy")
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
        return Options(
            shots=1024,
            memory=True,
            statevector=False,
            sampling=True,
            max_dynamic_loop_iterations=DEFAULT_MAX_DYNAMIC_LOOP_ITERATIONS,
        )

    @property
    def target(self):
        return self._target

    @property
    def max_circuits(self):
        return None

    def _ensure_simulator(self, num_qubits, batch_size=1):
        """Create or reset the simulator if the qubit count changes."""
        batch_size = int(batch_size)
        runtime_batch_size = 1 if self._runtime is None else self._runtime.batch_size()
        if self._runtime is None or self._num_qubits != num_qubits or runtime_batch_size != batch_size:
            self._runtime = RocQuantumRuntime.from_bindings(num_qubits, batch_size=batch_size)
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

    @staticmethod
    def _has_runtime_reset(circuit):
        touched_qubits = set()
        for instruction in circuit.data:
            op = instruction.operation
            if op.name in {"barrier", "delay", "save_statevector", "measure"}:
                continue

            q_indices = [circuit.find_bit(q).index for q in instruction.qubits]
            if op.name == "reset":
                if touched_qubits & set(q_indices):
                    return True
                continue

            touched_qubits.update(q_indices)

        return False

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
        try:
            self._runtime.apply_operation("p", [target], [theta])
            return
        except (NotImplementedError, RuntimeError, TypeError, ValueError):
            pass
        if include_global_phase:
            self._apply_global_phase_value(0.5 * theta, target=target)
        self._runtime.apply_operation("rz", [target], [theta])

    def _apply_phase_gate_batch(self, q_indices, thetas):
        if len(q_indices) != 1:
            raise ValueError("Qiskit p gate requires exactly one qubit.")

        try:
            self._runtime.apply_operation_batch("p", [q_indices[0]], thetas)
            return
        except (NotImplementedError, RuntimeError, TypeError, ValueError):
            pass
        self._runtime.apply_operation_batch("rz", [q_indices[0]], thetas)

    def _apply_controlled_phase_gate(self, q_indices, theta, *, include_global_phase):
        if len(q_indices) != 2:
            raise ValueError("Qiskit cp gate requires exactly two qubits.")

        control, target = q_indices
        try:
            self._runtime.apply_operation("cp", [control, target], [theta])
            return
        except (NotImplementedError, RuntimeError, TypeError, ValueError):
            pass
        if include_global_phase:
            self._apply_global_phase_value(0.25 * theta, target=control)
        self._runtime.apply_operation("rz", [control], [0.5 * theta])
        self._runtime.apply_operation("cx", [control, target])
        self._runtime.apply_operation("rz", [target], [-0.5 * theta])
        self._runtime.apply_operation("cx", [control, target])
        self._runtime.apply_operation("rz", [target], [0.5 * theta])

    def _apply_controlled_phase_gate_batch(self, q_indices, thetas):
        if len(q_indices) != 2:
            raise ValueError("Qiskit cp gate requires exactly two qubits.")

        control, target = q_indices
        try:
            self._runtime.apply_operation_batch("cp", [control, target], thetas)
            return
        except (NotImplementedError, RuntimeError, TypeError, ValueError):
            pass
        half_thetas = [0.5 * theta for theta in thetas]
        self._runtime.apply_operation_batch("rz", [control], half_thetas)
        self._runtime.apply_operation("cx", [control, target])
        self._runtime.apply_operation_batch("rz", [target], [-theta for theta in half_thetas])
        self._runtime.apply_operation("cx", [control, target])
        self._runtime.apply_operation_batch("rz", [target], half_thetas)

    def _apply_rzz_gate(self, q_indices, theta):
        if len(q_indices) != 2:
            raise ValueError("Qiskit rzz gate requires exactly two qubits.")

        control, target = q_indices
        self._runtime.apply_operation("cx", [control, target])
        self._runtime.apply_operation("rz", [target], [theta])
        self._runtime.apply_operation("cx", [control, target])

    def _apply_rzz_gate_batch(self, q_indices, thetas):
        if len(q_indices) != 2:
            raise ValueError("Qiskit rzz gate requires exactly two qubits.")

        control, target = q_indices
        self._runtime.apply_operation("cx", [control, target])
        self._runtime.apply_operation_batch("rz", [target], thetas)
        self._runtime.apply_operation("cx", [control, target])

    def _apply_rzx_gate(self, q_indices, theta):
        if len(q_indices) != 2:
            raise ValueError("Qiskit rzx gate requires exactly two qubits.")

        control, target = q_indices
        self._runtime.apply_operation("h", [target])
        self._apply_rzz_gate([control, target], theta)
        self._runtime.apply_operation("h", [target])

    def _apply_rzx_gate_batch(self, q_indices, thetas):
        if len(q_indices) != 2:
            raise ValueError("Qiskit rzx gate requires exactly two qubits.")

        control, target = q_indices
        self._runtime.apply_operation("h", [target])
        self._apply_rzz_gate_batch([control, target], thetas)
        self._runtime.apply_operation("h", [target])

    def _apply_multirz_gate(self, q_indices, theta):
        if not q_indices:
            raise ValueError("Qiskit MultiRZ decomposition requires at least one active qubit.")
        if len(q_indices) == 1:
            self._runtime.apply_operation("rz", [q_indices[0]], [theta])
            return

        target = q_indices[-1]
        for control in q_indices[:-1]:
            self._runtime.apply_operation("cx", [control, target])
        self._runtime.apply_operation("rz", [target], [theta])
        for control in reversed(q_indices[:-1]):
            self._runtime.apply_operation("cx", [control, target])

    def _apply_multirz_gate_batch(self, q_indices, thetas):
        if not q_indices:
            raise ValueError("Qiskit MultiRZ decomposition requires at least one active qubit.")
        if len(q_indices) == 1:
            self._runtime.apply_operation_batch("rz", [q_indices[0]], thetas)
            return

        target = q_indices[-1]
        for control in q_indices[:-1]:
            self._runtime.apply_operation("cx", [control, target])
        self._runtime.apply_operation_batch("rz", [target], thetas)
        for control in reversed(q_indices[:-1]):
            self._runtime.apply_operation("cx", [control, target])

    def _apply_multi_controlled_phase_gate(self, q_indices, theta, *, include_global_phase):
        if len(q_indices) < 2:
            raise ValueError("Qiskit multi-controlled phase requires at least two qubits.")

        qubit_count = len(q_indices)
        if include_global_phase:
            self._apply_global_phase_value(theta / (1 << qubit_count), target=q_indices[0])

        for subset_size in range(1, qubit_count + 1):
            angle = ((-1) ** (subset_size + 1)) * theta / (1 << (qubit_count - 1))
            for subset in combinations(q_indices, subset_size):
                self._apply_multirz_gate(list(subset), angle)

    def _apply_multi_controlled_phase_gate_batch(self, q_indices, thetas):
        if len(q_indices) < 2:
            raise ValueError("Qiskit multi-controlled phase requires at least two qubits.")

        qubit_count = len(q_indices)
        for subset_size in range(1, qubit_count + 1):
            scale = ((-1) ** (subset_size + 1)) / (1 << (qubit_count - 1))
            scaled_thetas = [scale * theta for theta in thetas]
            for subset in combinations(q_indices, subset_size):
                self._apply_multirz_gate_batch(list(subset), scaled_thetas)

    def _apply_multi_controlled_rz_gate(self, q_indices, theta, *, include_global_phase):
        if len(q_indices) < 2:
            raise ValueError("Qiskit controlled-rz gate requires at least two qubits.")

        controls = list(q_indices[:-1])
        target = q_indices[-1]
        if len(controls) == 1:
            self._runtime.apply_operation("crz", [controls[0], target], [theta])
            return

        self._apply_multi_controlled_phase_gate(
            controls,
            -0.5 * theta,
            include_global_phase=include_global_phase,
        )
        self._apply_multi_controlled_phase_gate(
            controls + [target],
            theta,
            include_global_phase=include_global_phase,
        )

    def _apply_multi_controlled_rz_gate_batch(self, q_indices, thetas):
        if len(q_indices) < 2:
            raise ValueError("Qiskit controlled-rz gate requires at least two qubits.")

        controls = list(q_indices[:-1])
        target = q_indices[-1]
        if len(controls) == 1:
            self._runtime.apply_operation_batch("crz", [controls[0], target], thetas)
            return

        self._apply_multi_controlled_phase_gate_batch(
            controls,
            [-0.5 * theta for theta in thetas],
        )
        self._apply_multi_controlled_phase_gate_batch(controls + [target], thetas)

    def _apply_multi_controlled_rx_gate(self, q_indices, theta, *, include_global_phase):
        if len(q_indices) < 2:
            raise ValueError("Qiskit controlled-rx gate requires at least two qubits.")

        if len(q_indices) == 2:
            self._runtime.apply_operation("crx", q_indices, [theta])
            return

        target = q_indices[-1]
        self._runtime.apply_operation("h", [target])
        self._apply_multi_controlled_rz_gate(
            q_indices,
            theta,
            include_global_phase=include_global_phase,
        )
        self._runtime.apply_operation("h", [target])

    def _apply_multi_controlled_rx_gate_batch(self, q_indices, thetas):
        if len(q_indices) < 2:
            raise ValueError("Qiskit controlled-rx gate requires at least two qubits.")

        if len(q_indices) == 2:
            self._runtime.apply_operation_batch("crx", q_indices, thetas)
            return

        target = q_indices[-1]
        self._runtime.apply_operation("h", [target])
        self._apply_multi_controlled_rz_gate_batch(q_indices, thetas)
        self._runtime.apply_operation("h", [target])

    def _apply_multi_controlled_ry_gate(self, q_indices, theta, *, include_global_phase):
        if len(q_indices) < 2:
            raise ValueError("Qiskit controlled-ry gate requires at least two qubits.")

        if len(q_indices) == 2:
            self._runtime.apply_operation("cry", q_indices, [theta])
            return

        target = q_indices[-1]
        self._runtime.apply_operation("sdg", [target])
        self._runtime.apply_operation("h", [target])
        self._apply_multi_controlled_rz_gate(
            q_indices,
            theta,
            include_global_phase=include_global_phase,
        )
        self._runtime.apply_operation("h", [target])
        self._runtime.apply_operation("s", [target])

    def _apply_multi_controlled_ry_gate_batch(self, q_indices, thetas):
        if len(q_indices) < 2:
            raise ValueError("Qiskit controlled-ry gate requires at least two qubits.")

        if len(q_indices) == 2:
            self._runtime.apply_operation_batch("cry", q_indices, thetas)
            return

        target = q_indices[-1]
        self._runtime.apply_operation("sdg", [target])
        self._runtime.apply_operation("h", [target])
        self._apply_multi_controlled_rz_gate_batch(q_indices, thetas)
        self._runtime.apply_operation("h", [target])
        self._runtime.apply_operation("s", [target])

    def _apply_multi_controlled_r_gate(self, q_indices, theta, phi, *, include_global_phase):
        if len(q_indices) < 2:
            raise ValueError("Qiskit controlled-r gate requires at least two qubits.")

        target = q_indices[-1]
        self._runtime.apply_operation("rz", [target], [-phi])
        self._apply_multi_controlled_rx_gate(
            q_indices,
            theta,
            include_global_phase=include_global_phase,
        )
        self._runtime.apply_operation("rz", [target], [phi])

    def _apply_multi_controlled_r_gate_batch(self, q_indices, thetas, phis):
        if len(q_indices) < 2:
            raise ValueError("Qiskit controlled-r gate requires at least two qubits.")

        target = q_indices[-1]
        self._runtime.apply_operation_batch("rz", [target], [-phi for phi in phis])
        self._apply_multi_controlled_rx_gate_batch(q_indices, thetas)
        self._runtime.apply_operation_batch("rz", [target], phis)

    def _apply_multi_controlled_u3_gate(self, q_indices, theta, phi, lam, *, include_global_phase):
        if len(q_indices) < 2:
            raise ValueError("Qiskit controlled-u3 gate requires at least two qubits.")

        controls = list(q_indices[:-1])
        half_sum = 0.5 * (phi + lam)
        if len(controls) == 1:
            self._apply_phase_gate(
                [controls[0]],
                half_sum,
                include_global_phase=include_global_phase,
            )
        else:
            self._apply_multi_controlled_phase_gate(
                controls,
                half_sum,
                include_global_phase=include_global_phase,
            )
        self._apply_multi_controlled_rz_gate(q_indices, lam, include_global_phase=include_global_phase)
        self._apply_multi_controlled_ry_gate(q_indices, theta, include_global_phase=include_global_phase)
        self._apply_multi_controlled_rz_gate(q_indices, phi, include_global_phase=include_global_phase)

    def _apply_multi_controlled_u3_gate_batch(self, q_indices, thetas, phis, lams):
        if len(q_indices) < 2:
            raise ValueError("Qiskit controlled-u3 gate requires at least two qubits.")

        controls = list(q_indices[:-1])
        half_sums = [0.5 * (phi + lam) for phi, lam in zip(phis, lams)]
        if len(controls) == 1:
            self._apply_phase_gate_batch([controls[0]], half_sums)
        else:
            self._apply_multi_controlled_phase_gate_batch(controls, half_sums)
        self._apply_multi_controlled_rz_gate_batch(q_indices, lams)
        self._apply_multi_controlled_ry_gate_batch(q_indices, thetas)
        self._apply_multi_controlled_rz_gate_batch(q_indices, phis)

    def _apply_pauli_rotation_gate(self, q_indices, label, theta):
        if len(label) != len(q_indices):
            raise ValueError("PauliEvolution label length must match the target qubits.")

        active = [
            (q_indices[local_index], pauli)
            for local_index, pauli in enumerate(reversed(str(label).upper()))
            if pauli != "I"
        ]
        if not active:
            return False
        if len(active) == 1:
            qubit, pauli = active[0]
            if pauli == "X":
                self._runtime.apply_operation("rx", [qubit], [theta])
            elif pauli == "Y":
                self._runtime.apply_operation("ry", [qubit], [theta])
            elif pauli == "Z":
                self._runtime.apply_operation("rz", [qubit], [theta])
            else:
                raise ValueError("PauliEvolution labels may only contain I, X, Y, or Z.")
            return True

        for qubit, pauli in active:
            if pauli == "X":
                self._runtime.apply_operation("h", [qubit])
            elif pauli == "Y":
                self._runtime.apply_operation("rx", [qubit], [cmath.pi / 2])
            elif pauli != "Z":
                raise ValueError("PauliEvolution labels may only contain I, X, Y, or Z.")

        self._apply_multirz_gate([qubit for qubit, _ in active], theta)

        for qubit, pauli in active:
            if pauli == "X":
                self._runtime.apply_operation("h", [qubit])
            elif pauli == "Y":
                self._runtime.apply_operation("rx", [qubit], [-cmath.pi / 2])

        return True

    def _apply_pauli_rotation_gate_batch(self, q_indices, label, thetas):
        if len(label) != len(q_indices):
            raise ValueError("PauliEvolution label length must match the target qubits.")

        active = [
            (q_indices[local_index], pauli)
            for local_index, pauli in enumerate(reversed(str(label).upper()))
            if pauli != "I"
        ]
        if not active:
            return False
        if len(active) == 1:
            qubit, pauli = active[0]
            if pauli == "X":
                self._runtime.apply_operation_batch("rx", [qubit], thetas)
            elif pauli == "Y":
                self._runtime.apply_operation_batch("ry", [qubit], thetas)
            elif pauli == "Z":
                self._runtime.apply_operation_batch("rz", [qubit], thetas)
            else:
                raise ValueError("PauliEvolution labels may only contain I, X, Y, or Z.")
            return True

        for qubit, pauli in active:
            if pauli == "X":
                self._runtime.apply_operation("h", [qubit])
            elif pauli == "Y":
                self._runtime.apply_operation("rx", [qubit], [cmath.pi / 2])
            elif pauli != "Z":
                raise ValueError("PauliEvolution labels may only contain I, X, Y, or Z.")

        self._apply_multirz_gate_batch([qubit for qubit, _ in active], thetas)

        for qubit, pauli in active:
            if pauli == "X":
                self._runtime.apply_operation("h", [qubit])
            elif pauli == "Y":
                self._runtime.apply_operation("rx", [qubit], [-cmath.pi / 2])

        return True

    def _apply_rxx_gate(self, q_indices, theta):
        if len(q_indices) != 2:
            raise ValueError("Qiskit rxx gate requires exactly two qubits.")

        control, target = q_indices
        self._runtime.apply_operation("cx", [control, target])
        self._runtime.apply_operation("rx", [control], [theta])
        self._runtime.apply_operation("cx", [control, target])

    def _apply_rxx_gate_batch(self, q_indices, thetas):
        if len(q_indices) != 2:
            raise ValueError("Qiskit rxx gate requires exactly two qubits.")

        control, target = q_indices
        self._runtime.apply_operation("cx", [control, target])
        self._runtime.apply_operation_batch("rx", [control], thetas)
        self._runtime.apply_operation("cx", [control, target])

    def _apply_ryy_gate(self, q_indices, theta):
        if len(q_indices) != 2:
            raise ValueError("Qiskit ryy gate requires exactly two qubits.")

        for qubit in q_indices:
            self._runtime.apply_operation("rx", [qubit], [cmath.pi / 2])
        self._apply_rzz_gate(q_indices, theta)
        for qubit in q_indices:
            self._runtime.apply_operation("rx", [qubit], [-cmath.pi / 2])

    def _apply_ryy_gate_batch(self, q_indices, thetas):
        if len(q_indices) != 2:
            raise ValueError("Qiskit ryy gate requires exactly two qubits.")

        for qubit in q_indices:
            self._runtime.apply_operation("rx", [qubit], [cmath.pi / 2])
        self._apply_rzz_gate_batch(q_indices, thetas)
        for qubit in q_indices:
            self._runtime.apply_operation("rx", [qubit], [-cmath.pi / 2])

    def _apply_xx_plus_yy_gate(self, q_indices, theta, beta, *, include_global_phase):
        if len(q_indices) != 2:
            raise ValueError("Qiskit xx_plus_yy gate requires exactly two qubits.")

        left, right = q_indices
        self._runtime.apply_operation("rz", [left], [beta])
        self._runtime.apply_operation("sdg", [right])
        self._apply_sx_gate([right], inverse=False, include_global_phase=include_global_phase)
        self._runtime.apply_operation("s", [right])
        self._runtime.apply_operation("s", [left])
        self._runtime.apply_operation("cx", [right, left])
        self._runtime.apply_operation("ry", [right], [-theta / 2])
        self._runtime.apply_operation("ry", [left], [-theta / 2])
        self._runtime.apply_operation("cx", [right, left])
        self._runtime.apply_operation("sdg", [left])
        self._runtime.apply_operation("sdg", [right])
        self._apply_sx_gate([right], inverse=True, include_global_phase=include_global_phase)
        self._runtime.apply_operation("s", [right])
        self._runtime.apply_operation("rz", [left], [-beta])

    def _apply_xx_plus_yy_gate_batch(self, q_indices, thetas, betas):
        if len(q_indices) != 2:
            raise ValueError("Qiskit xx_plus_yy gate requires exactly two qubits.")

        left, right = q_indices
        self._runtime.apply_operation_batch("rz", [left], betas)
        self._runtime.apply_operation("sdg", [right])
        self._apply_sx_gate([right], inverse=False, include_global_phase=False)
        self._runtime.apply_operation("s", [right])
        self._runtime.apply_operation("s", [left])
        self._runtime.apply_operation("cx", [right, left])
        self._runtime.apply_operation_batch("ry", [right], [-theta / 2 for theta in thetas])
        self._runtime.apply_operation_batch("ry", [left], [-theta / 2 for theta in thetas])
        self._runtime.apply_operation("cx", [right, left])
        self._runtime.apply_operation("sdg", [left])
        self._runtime.apply_operation("sdg", [right])
        self._apply_sx_gate([right], inverse=True, include_global_phase=False)
        self._runtime.apply_operation("s", [right])
        self._runtime.apply_operation_batch("rz", [left], [-beta for beta in betas])

    def _apply_xx_minus_yy_gate(self, q_indices, theta, beta, *, include_global_phase):
        if len(q_indices) != 2:
            raise ValueError("Qiskit xx_minus_yy gate requires exactly two qubits.")

        left, right = q_indices
        self._runtime.apply_operation("rz", [right], [-beta])
        self._runtime.apply_operation("sdg", [left])
        self._apply_sx_gate([left], inverse=False, include_global_phase=include_global_phase)
        self._runtime.apply_operation("s", [left])
        self._runtime.apply_operation("s", [right])
        self._runtime.apply_operation("cx", [left, right])
        self._runtime.apply_operation("ry", [left], [theta / 2])
        self._runtime.apply_operation("ry", [right], [-theta / 2])
        self._runtime.apply_operation("cx", [left, right])
        self._runtime.apply_operation("sdg", [right])
        self._runtime.apply_operation("sdg", [left])
        self._apply_sx_gate([left], inverse=True, include_global_phase=include_global_phase)
        self._runtime.apply_operation("s", [left])
        self._runtime.apply_operation("rz", [right], [beta])

    def _apply_xx_minus_yy_gate_batch(self, q_indices, thetas, betas):
        if len(q_indices) != 2:
            raise ValueError("Qiskit xx_minus_yy gate requires exactly two qubits.")

        left, right = q_indices
        self._runtime.apply_operation_batch("rz", [right], [-beta for beta in betas])
        self._runtime.apply_operation("sdg", [left])
        self._apply_sx_gate([left], inverse=False, include_global_phase=False)
        self._runtime.apply_operation("s", [left])
        self._runtime.apply_operation("s", [right])
        self._runtime.apply_operation("cx", [left, right])
        self._runtime.apply_operation_batch("ry", [left], [theta / 2 for theta in thetas])
        self._runtime.apply_operation_batch("ry", [right], [-theta / 2 for theta in thetas])
        self._runtime.apply_operation("cx", [left, right])
        self._runtime.apply_operation("sdg", [right])
        self._runtime.apply_operation("sdg", [left])
        self._apply_sx_gate([left], inverse=True, include_global_phase=False)
        self._runtime.apply_operation("s", [left])
        self._runtime.apply_operation_batch("rz", [right], betas)

    def _apply_pauli_evolution_gate(self, op, q_indices, *, include_global_phase):
        terms = _pauli_evolution_terms(op)
        if terms is None:
            return False

        (time,) = normalize_params(op.params)
        applied = False
        for label, coeff in terms:
            coeff = complex(coeff).real
            if set(str(label).upper()) <= {"I"}:
                if include_global_phase:
                    self._apply_global_phase_value(-float(time) * coeff)
                applied = True
                continue

            theta = 2.0 * float(time) * coeff
            applied = self._apply_pauli_rotation_gate(q_indices, label, theta) or applied

        return applied

    def _apply_pauli_evolution_gate_batch(self, op, q_indices, times):
        terms = _pauli_evolution_terms(op)
        if terms is None:
            return False

        applied = False
        for label, coeff in terms:
            coeff = complex(coeff).real
            if set(str(label).upper()) <= {"I"}:
                applied = True
                continue

            thetas = [2.0 * float(time) * coeff for time in times]
            applied = self._apply_pauli_rotation_gate_batch(q_indices, label, thetas) or applied

        return applied

    def _apply_sx_gate(self, q_indices, *, inverse=False, include_global_phase):
        if len(q_indices) != 1:
            raise ValueError("Qiskit sx gate requires exactly one qubit.")

        target = q_indices[0]
        sign = -1.0 if inverse else 1.0
        if include_global_phase:
            self._apply_global_phase_value(sign * cmath.pi / 4, target=target)
        self._runtime.apply_operation("rx", [target], [sign * cmath.pi / 2])

    def _apply_csx_gate(self, q_indices, *, include_global_phase):
        if len(q_indices) != 2:
            raise ValueError("Qiskit csx gate requires exactly two qubits.")

        control, target = q_indices
        self._apply_phase_gate([control], cmath.pi / 4, include_global_phase=include_global_phase)
        self._runtime.apply_operation("crx", [control, target], [cmath.pi / 2])

    def _apply_u_gate(self, q_indices, theta, phi, lam, *, include_global_phase):
        if len(q_indices) != 1:
            raise ValueError("Qiskit u gate requires exactly one qubit.")

        target = q_indices[0]
        if include_global_phase:
            self._apply_global_phase_value(0.5 * (phi + lam), target=target)
        self._runtime.apply_operation("rz", [target], [lam])
        self._runtime.apply_operation("ry", [target], [theta])
        self._runtime.apply_operation("rz", [target], [phi])

    def _apply_u_gate_batch(self, q_indices, thetas, phis, lams):
        if len(q_indices) != 1:
            raise ValueError("Qiskit u gate requires exactly one qubit.")

        target = q_indices[0]
        self._runtime.apply_operation_batch("rz", [target], lams)
        self._runtime.apply_operation_batch("ry", [target], thetas)
        self._runtime.apply_operation_batch("rz", [target], phis)

    def _apply_cu3_gate(self, q_indices, theta, phi, lam, *, include_global_phase):
        if len(q_indices) != 2:
            raise ValueError("Qiskit cu3 gate requires exactly two qubits.")

        control, target = q_indices
        half_sum = 0.5 * (phi + lam)
        self._apply_phase_gate([control], half_sum, include_global_phase=include_global_phase)
        self._apply_phase_gate([target], 0.5 * theta, include_global_phase=include_global_phase)
        self._runtime.apply_operation("cx", [control, target])
        self._apply_u_gate(
            [target],
            -0.5 * theta,
            0.0,
            -half_sum,
            include_global_phase=include_global_phase,
        )
        self._runtime.apply_operation("cx", [control, target])
        self._apply_u_gate([target], 0.5 * theta, phi, 0.0, include_global_phase=include_global_phase)

    def _apply_cu3_gate_batch(self, q_indices, thetas, phis, lams):
        if len(q_indices) != 2:
            raise ValueError("Qiskit cu3 gate requires exactly two qubits.")

        control, target = q_indices
        half_sums = [0.5 * (phi + lam) for phi, lam in zip(phis, lams)]
        half_thetas = [0.5 * theta for theta in thetas]
        self._apply_phase_gate_batch([control], half_sums)
        self._apply_phase_gate_batch([target], half_thetas)
        self._runtime.apply_operation("cx", [control, target])
        self._apply_u_gate_batch(
            [target],
            [-theta for theta in half_thetas],
            [0.0 for _ in half_thetas],
            [-value for value in half_sums],
        )
        self._runtime.apply_operation("cx", [control, target])
        self._apply_u_gate_batch([target], half_thetas, phis, [0.0 for _ in half_thetas])

    def _apply_cu_gate(self, q_indices, theta, phi, lam, gamma, *, include_global_phase):
        if len(q_indices) != 2:
            raise ValueError("Qiskit cu gate requires exactly two qubits.")

        control = q_indices[0]
        self._apply_phase_gate([control], gamma, include_global_phase=include_global_phase)
        self._apply_cu3_gate(q_indices, theta, phi, lam, include_global_phase=include_global_phase)

    def _apply_cu_gate_batch(self, q_indices, thetas, phis, lams, gammas):
        if len(q_indices) != 2:
            raise ValueError("Qiskit cu gate requires exactly two qubits.")

        self._apply_phase_gate_batch([q_indices[0]], gammas)
        self._apply_cu3_gate_batch(q_indices, thetas, phis, lams)

    def _apply_r_gate(self, q_indices, theta, phi, *, include_global_phase):
        if len(q_indices) != 1:
            raise ValueError("Qiskit r gate requires exactly one qubit.")

        self._apply_u_gate(
            q_indices,
            theta,
            phi - cmath.pi / 2,
            cmath.pi / 2 - phi,
            include_global_phase=include_global_phase,
        )

    def _apply_r_gate_batch(self, q_indices, thetas, phis):
        if len(q_indices) != 1:
            raise ValueError("Qiskit r gate requires exactly one qubit.")

        self._apply_u_gate_batch(
            q_indices,
            thetas,
            [phi - cmath.pi / 2 for phi in phis],
            [cmath.pi / 2 - phi for phi in phis],
        )

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
        try:
            self._runtime.apply_operation("tdg", [target])
            return
        except (NotImplementedError, RuntimeError, TypeError, ValueError):
            pass
        if include_global_phase:
            self._apply_global_phase_value(-cmath.pi / 8, target=target)
        self._runtime.apply_operation("rz", [target], [-cmath.pi / 4])

    def _apply_pauli_gate(self, op, q_indices):
        if op.name != "pauli" or len(getattr(op, "params", ()) or ()) != 1:
            return False

        label = str(op.params[0]).upper()
        if len(label) != len(q_indices):
            return False

        for target, pauli in zip(q_indices, reversed(label)):
            if pauli == "I":
                continue
            if pauli not in {"X", "Y", "Z"}:
                return False
            self._runtime.apply_operation(pauli.lower(), [target])
        return True

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

    def _apply_controlled_base_gate(self, op, q_indices, *, include_global_phase):
        num_controls = int(getattr(op, "num_ctrl_qubits", 0))
        base_gate = getattr(op, "base_gate", None)
        base_name = getattr(base_gate, "name", None)
        if num_controls < 1 or len(q_indices) != num_controls + 1:
            return False
        if base_name not in {"x", "h", "y", "z", "rx", "ry", "rz", "r", "p", "s", "sdg", "t", "tdg", "sx", "u1", "u2", "u3", "u"}:
            return False
        if base_name == "u" and num_controls != 1:
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
            elif base_name == "h":
                if len(controls) == 1:
                    self._apply_ch_gate([controls[0], target])
                else:
                    self._runtime.apply_operation("ry", [target], [cmath.pi / 4])
                    self._runtime.apply_operation("mcx", controls + [target])
                    self._runtime.apply_operation("ry", [target], [-cmath.pi / 4])
            elif base_name == "y":
                if len(controls) == 1:
                    self._apply_cy_gate([controls[0], target])
                else:
                    self._runtime.apply_operation("sdg", [target])
                    self._runtime.apply_operation("mcx", controls + [target])
                    self._runtime.apply_operation("s", [target])
            elif base_name == "rx":
                (theta,) = normalize_params(op.params)
                self._apply_multi_controlled_rx_gate(
                    controls + [target],
                    theta,
                    include_global_phase=include_global_phase,
                )
            elif base_name == "ry":
                (theta,) = normalize_params(op.params)
                self._apply_multi_controlled_ry_gate(
                    controls + [target],
                    theta,
                    include_global_phase=include_global_phase,
                )
            elif base_name == "rz":
                (theta,) = normalize_params(op.params)
                self._apply_multi_controlled_rz_gate(
                    controls + [target],
                    theta,
                    include_global_phase=include_global_phase,
                )
            elif base_name == "r":
                theta, phi = normalize_params(op.params)
                self._apply_multi_controlled_r_gate(
                    controls + [target],
                    theta,
                    phi,
                    include_global_phase=include_global_phase,
                )
            elif base_name == "p":
                (theta,) = normalize_params(op.params)
                if len(controls) == 1:
                    self._apply_controlled_phase_gate(
                        [controls[0], target],
                        theta,
                        include_global_phase=include_global_phase,
                    )
                else:
                    self._apply_multi_controlled_phase_gate(
                        controls + [target],
                        theta,
                        include_global_phase=include_global_phase,
                    )
            elif base_name in {"s", "sdg", "t", "tdg"}:
                fixed_phase = {
                    "s": cmath.pi / 2,
                    "sdg": -cmath.pi / 2,
                    "t": cmath.pi / 4,
                    "tdg": -cmath.pi / 4,
                }[base_name]
                if len(controls) == 1:
                    self._apply_controlled_phase_gate(
                        [controls[0], target],
                        fixed_phase,
                        include_global_phase=include_global_phase,
                    )
                else:
                    self._apply_multi_controlled_phase_gate(
                        controls + [target],
                        fixed_phase,
                        include_global_phase=include_global_phase,
                    )
            elif base_name == "sx":
                if len(controls) == 1:
                    self._apply_csx_gate([controls[0], target], include_global_phase=include_global_phase)
                else:
                    self._runtime.apply_operation("h", [target])
                    self._apply_multi_controlled_phase_gate(
                        controls + [target],
                        cmath.pi / 2,
                        include_global_phase=include_global_phase,
                    )
                    self._runtime.apply_operation("h", [target])
            elif base_name == "u1":
                (theta,) = normalize_params(op.params)
                if len(controls) == 1:
                    self._apply_controlled_phase_gate(
                        [controls[0], target],
                        theta,
                        include_global_phase=include_global_phase,
                    )
                else:
                    self._apply_multi_controlled_phase_gate(
                        controls + [target],
                        theta,
                        include_global_phase=include_global_phase,
                    )
            elif base_name == "u2":
                phi, lam = normalize_params(op.params)
                self._apply_multi_controlled_u3_gate(
                    controls + [target],
                    cmath.pi / 2,
                    phi,
                    lam,
                    include_global_phase=include_global_phase,
                )
            elif base_name == "u3":
                theta, phi, lam = normalize_params(op.params)
                if len(controls) == 1:
                    self._apply_cu3_gate(
                        [controls[0], target],
                        theta,
                        phi,
                        lam,
                        include_global_phase=include_global_phase,
                    )
                else:
                    self._apply_multi_controlled_u3_gate(
                        controls + [target],
                        theta,
                        phi,
                        lam,
                        include_global_phase=include_global_phase,
                    )
            elif base_name == "u":
                theta, phi, lam, gamma = normalize_params(op.params)
                self._apply_cu_gate(
                    [controls[0], target],
                    theta,
                    phi,
                    lam,
                    gamma,
                    include_global_phase=include_global_phase,
                )
            elif len(controls) == 1:
                self._runtime.apply_operation("cz", [controls[0], target])
            elif len(controls) == 2:
                self._apply_ccz_gate([controls[0], controls[1], target])
            else:
                self._runtime.apply_operation("h", [target])
                self._runtime.apply_operation("mcx", controls + [target])
                self._runtime.apply_operation("h", [target])
            for control in reversed(flipped):
                self._runtime.apply_operation("x", [control])
            flipped.clear()
        finally:
            for control in reversed(flipped):
                self._runtime.apply_operation("x", [control])
        return True

    def _apply_controlled_base_gate_batch(self, op, q_indices, thetas):
        num_controls = int(getattr(op, "num_ctrl_qubits", 0))
        base_gate = getattr(op, "base_gate", None)
        base_name = getattr(base_gate, "name", None)
        if num_controls < 1 or len(q_indices) != num_controls + 1 or base_name not in {"rx", "ry", "rz", "p", "u1"}:
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

            if base_name == "rx":
                self._apply_multi_controlled_rx_gate_batch(controls + [target], thetas)
            elif base_name == "ry":
                self._apply_multi_controlled_ry_gate_batch(controls + [target], thetas)
            elif base_name == "rz":
                self._apply_multi_controlled_rz_gate_batch(controls + [target], thetas)
            elif len(controls) == 1:
                self._apply_controlled_phase_gate_batch([controls[0], target], thetas)
            else:
                self._apply_multi_controlled_phase_gate_batch(controls + [target], thetas)
            for control in reversed(flipped):
                self._runtime.apply_operation("x", [control])
            flipped.clear()
            return True
        finally:
            for control in reversed(flipped):
                self._runtime.apply_operation("x", [control])

    def _apply_controlled_r_gate_batch(self, op, q_indices, params_by_circuit):
        num_controls = int(getattr(op, "num_ctrl_qubits", 0))
        base_gate = getattr(op, "base_gate", None)
        base_name = getattr(base_gate, "name", None)
        if (
            num_controls < 1
            or len(q_indices) != num_controls + 1
            or base_name != "r"
            or not all(len(params) == 2 for params in params_by_circuit)
        ):
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
        thetas = [params[0] for params in params_by_circuit]
        phis = [params[1] for params in params_by_circuit]

        flipped = []
        try:
            for control in open_controls:
                self._runtime.apply_operation("x", [control])
                flipped.append(control)
            self._apply_multi_controlled_r_gate_batch(controls + [target], thetas, phis)
            for control in reversed(flipped):
                self._runtime.apply_operation("x", [control])
            flipped.clear()
            return True
        finally:
            for control in reversed(flipped):
                self._runtime.apply_operation("x", [control])

    def _apply_controlled_u2_u3_gate_batch(self, op, q_indices, params_by_circuit):
        num_controls = int(getattr(op, "num_ctrl_qubits", 0))
        base_gate = getattr(op, "base_gate", None)
        base_name = getattr(base_gate, "name", None)
        if num_controls < 1 or len(q_indices) != num_controls + 1 or base_name not in {"u2", "u3"}:
            return False
        if base_name == "u3" and num_controls == 1:
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

        if base_name == "u2" and all(len(params) == 2 for params in params_by_circuit):
            phis = [params[0] for params in params_by_circuit]
            lams = [params[1] for params in params_by_circuit]
            thetas = [cmath.pi / 2 for _ in phis]
        elif base_name == "u3" and all(len(params) == 3 for params in params_by_circuit):
            thetas = [params[0] for params in params_by_circuit]
            phis = [params[1] for params in params_by_circuit]
            lams = [params[2] for params in params_by_circuit]
        else:
            return False

        flipped = []
        try:
            for control in open_controls:
                self._runtime.apply_operation("x", [control])
                flipped.append(control)
            self._apply_multi_controlled_u3_gate_batch(controls + [target], thetas, phis, lams)
            for control in reversed(flipped):
                self._runtime.apply_operation("x", [control])
            flipped.clear()
            return True
        finally:
            for control in reversed(flipped):
                self._runtime.apply_operation("x", [control])

    def _apply_controlled_u_gate_batch(self, op, q_indices, params_by_circuit):
        num_controls = int(getattr(op, "num_ctrl_qubits", 0))
        base_gate = getattr(op, "base_gate", None)
        base_name = getattr(base_gate, "name", None)
        if num_controls != 1 or len(q_indices) != 2 or base_name not in {"u1", "u3", "u"}:
            return False

        ctrl_state = getattr(op, "ctrl_state", None)
        ctrl_state = 1 if ctrl_state is None else int(ctrl_state)
        control, target = q_indices
        flipped = False
        try:
            if ctrl_state == 0:
                self._runtime.apply_operation("x", [control])
                flipped = True
            elif ctrl_state != 1:
                return False

            if base_name == "u1" and all(len(params) == 1 for params in params_by_circuit):
                self._apply_controlled_phase_gate_batch(
                    [control, target],
                    [params[0] for params in params_by_circuit],
                )
                return True
            if base_name == "u3" and all(len(params) == 3 for params in params_by_circuit):
                thetas = [params[0] for params in params_by_circuit]
                phis = [params[1] for params in params_by_circuit]
                lams = [params[2] for params in params_by_circuit]
                self._apply_cu3_gate_batch([control, target], thetas, phis, lams)
                return True
            if base_name == "u" and all(len(params) == 4 for params in params_by_circuit):
                thetas = [params[0] for params in params_by_circuit]
                phis = [params[1] for params in params_by_circuit]
                lams = [params[2] for params in params_by_circuit]
                gammas = [params[3] for params in params_by_circuit]
                self._apply_cu_gate_batch([control, target], thetas, phis, lams, gammas)
                return True
        finally:
            if flipped:
                self._runtime.apply_operation("x", [control])

        return False

    def _apply_generic_controlled_matrix(self, op, q_indices):
        controlled_matrix = getattr(self._runtime, "apply_controlled_matrix", None)
        if not callable(controlled_matrix):
            return False
        if normalize_gate_name(getattr(op, "name", "")) in GATE_ALIASES.values():
            return False

        num_controls = int(getattr(op, "num_ctrl_qubits", 0))
        base_gate = getattr(op, "base_gate", None)
        if num_controls < 1 or base_gate is None or len(q_indices) <= num_controls:
            return False

        target_count = len(q_indices) - num_controls
        base_matrix = _automatic_operation_matrix(base_gate)
        if base_matrix is None or getattr(base_matrix, "shape", None) != (1 << target_count, 1 << target_count):
            return False

        ctrl_state = getattr(op, "ctrl_state", None)
        ctrl_state = (1 << num_controls) - 1 if ctrl_state is None else int(ctrl_state)
        controls = list(q_indices[:num_controls])
        targets = list(q_indices[num_controls:])
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
            self._runtime.apply_controlled_matrix(base_matrix, controls, targets)
            for control in reversed(flipped):
                self._runtime.apply_operation("x", [control])
            flipped.clear()
        finally:
            for control in reversed(flipped):
                self._runtime.apply_operation("x", [control])

        return True

    def _apply_quantum_operation(self, op, q_indices, *, include_global_phase, touched_qubits, circuit_num_qubits):
        if op.name == "global_phase":
            if include_global_phase and circuit_num_qubits > 0:
                (phase,) = normalize_params(op.params)
                self._apply_global_phase_value(phase)
            return

        if op.name in {"p", "cp"} and self._supports_native_parametric_decomposition():
            (theta,) = normalize_params(op.params)
            if op.name == "p":
                self._apply_phase_gate(q_indices, theta, include_global_phase=include_global_phase)
            else:
                self._apply_controlled_phase_gate(q_indices, theta, include_global_phase=include_global_phase)
            touched_qubits.update(q_indices)
            return

        if op.name == "tdg" and self._supports_native_phase_decomposition(include_global_phase):
            self._apply_tdg_gate(q_indices, include_global_phase=include_global_phase)
            touched_qubits.update(q_indices)
            return

        if op.name in {"rxx", "ryy", "rzz", "rzx"} and self._supports_native_parametric_decomposition():
            (theta,) = normalize_params(op.params)
            if op.name == "rxx":
                self._apply_rxx_gate(q_indices, theta)
            elif op.name == "ryy":
                self._apply_ryy_gate(q_indices, theta)
            elif op.name == "rzz":
                self._apply_rzz_gate(q_indices, theta)
            else:
                self._apply_rzx_gate(q_indices, theta)
            touched_qubits.update(q_indices)
            return

        if op.name in {"xx_plus_yy", "xx_minus_yy"} and self._supports_native_phase_decomposition(include_global_phase):
            theta, beta = normalize_params(op.params)
            if op.name == "xx_plus_yy":
                self._apply_xx_plus_yy_gate(
                    q_indices,
                    theta,
                    beta,
                    include_global_phase=include_global_phase,
                )
            else:
                self._apply_xx_minus_yy_gate(
                    q_indices,
                    theta,
                    beta,
                    include_global_phase=include_global_phase,
                )
            touched_qubits.update(q_indices)
            return

        if op.name == "PauliEvolution" and self._supports_native_parametric_decomposition():
            try:
                if self._apply_pauli_evolution_gate(
                    op,
                    q_indices,
                    include_global_phase=include_global_phase,
                ):
                    touched_qubits.update(q_indices)
                    return
            except (NotImplementedError, RuntimeError, TypeError, ValueError):
                pass

        if op.name in {
            "sx", "sxdg", "u", "u1", "u2", "u3", "r"
        } and self._supports_native_phase_decomposition(include_global_phase):
            params = normalize_params(op.params)
            if op.name == "sx":
                self._apply_sx_gate(q_indices, inverse=False, include_global_phase=include_global_phase)
            elif op.name == "sxdg":
                self._apply_sx_gate(q_indices, inverse=True, include_global_phase=include_global_phase)
            elif op.name == "u":
                theta, phi, lam = params
                self._apply_u_gate(q_indices, theta, phi, lam, include_global_phase=include_global_phase)
            elif op.name == "u1":
                (theta,) = params
                self._apply_phase_gate(q_indices, theta, include_global_phase=include_global_phase)
            elif op.name == "u2":
                phi, lam = params
                self._apply_u_gate(q_indices, cmath.pi / 2, phi, lam, include_global_phase=include_global_phase)
            elif op.name == "u3":
                theta, phi, lam = params
                self._apply_u_gate(q_indices, theta, phi, lam, include_global_phase=include_global_phase)
            else:
                theta, phi = params
                self._apply_r_gate(q_indices, theta, phi, include_global_phase=include_global_phase)
            touched_qubits.update(q_indices)
            return

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
            return

        if self._supports_native_parametric_decomposition() and self._apply_pauli_gate(op, q_indices):
            touched_qubits.update(q_indices)
            return

        if self._supports_native_parametric_decomposition():
            try:
                if self._apply_controlled_base_gate(
                    op,
                    q_indices,
                    include_global_phase=include_global_phase,
                ):
                    touched_qubits.update(q_indices)
                    return
            except (NotImplementedError, RuntimeError, TypeError, ValueError):
                pass

        try:
            if self._apply_generic_controlled_matrix(op, q_indices):
                touched_qubits.update(q_indices)
                return
        except (NotImplementedError, RuntimeError, TypeError, ValueError):
            pass

        if op.name == "state_preparation" and not touched_qubits and len(q_indices) == circuit_num_qubits:
            statevector = _state_preparation_vector(op)
            if statevector is not None:
                try:
                    self._runtime.set_statevector(
                        statevector_to_little_endian_wires(statevector)
                    )
                    touched_qubits.update(q_indices)
                    return
                except (NotImplementedError, RuntimeError, TypeError, ValueError):
                    pass

        matrix = _operation_matrix(op)
        params = _operation_runtime_params(op, matrix)
        self._runtime.apply_operation(
            op.name,
            q_indices,
            params,
            matrix=matrix,
        )
        touched_qubits.update(q_indices)

    def _apply_circuit(self, circuit, *, include_global_phase: bool = False, allow_runtime_reset: bool = False):
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

            if op.name == "global_phase":
                if include_global_phase and circuit.num_qubits > 0:
                    (phase,) = normalize_params(op.params)
                    self._apply_global_phase_value(phase)
                continue

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
                statevector = _state_preparation_vector(op)
                if statevector is not None and len(q_indices) == circuit.num_qubits:
                    try:
                        self._runtime.set_statevector(
                            statevector_to_little_endian_wires(statevector)
                        )
                    except (NotImplementedError, RuntimeError, TypeError, ValueError):
                        self._runtime.apply_operation(
                            "state_preparation",
                            q_indices,
                            matrix=_state_preparation_matrix(op),
                        )
                else:
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
                    if not allow_runtime_reset:
                        raise ValueError(
                            "RocQuantumBackend reset after a qubit has been operated on requires "
                            "shot-by-shot sampling through QuantumSimulator.reset_qubit()."
                        )
                    for target in q_indices:
                        self._runtime.reset_qubit(target)
                    touched_qubits.update(q_indices)
                continue

            if op.name == 'measure':
                measurement_started = True
                # Store which classical bit this measurement corresponds to
                c_index = circuit.find_bit(instruction.clbits[0]).index
                measured_bits[c_index] = q_indices[0]
                continue

            self._apply_quantum_operation(
                op,
                q_indices,
                include_global_phase=include_global_phase,
                touched_qubits=touched_qubits,
                circuit_num_qubits=circuit.num_qubits,
            )

        return measured_bits

    def _apply_circuit_batch(self, circuits, *, include_global_phase: bool = False):
        circuits = list(circuits)
        if not circuits:
            raise ValueError("_apply_circuit_batch requires at least one circuit.")
        if include_global_phase:
            raise NotImplementedError("Batched Qiskit estimator execution does not preserve global phase yet.")

        num_qubits = int(circuits[0].num_qubits)
        instruction_count = len(circuits[0].data)
        for circuit in circuits:
            if int(circuit.num_qubits) != num_qubits or len(circuit.data) != instruction_count:
                raise NotImplementedError("Batched Qiskit execution requires identical circuit structure.")
            if self._has_runtime_reset(circuit):
                raise NotImplementedError("Batched Qiskit execution does not support runtime reset.")

        self._ensure_simulator(num_qubits, batch_size=len(circuits))

        measured_bits = {}
        measurement_started = False
        touched_qubits = set()
        for position, reference_instruction in enumerate(circuits[0].data):
            reference_op = reference_instruction.operation
            if reference_op.name in {"barrier", "delay", "global_phase", "save_statevector"}:
                continue
            if reference_op.name in CONTROL_FLOW_OPS or getattr(reference_op, "blocks", None) is not None:
                raise NotImplementedError("Batched Qiskit execution does not support control-flow operations.")
            if _instruction_condition(reference_instruction) is not None:
                raise NotImplementedError("Batched Qiskit execution does not support conditioned operations.")

            q_indices = [circuits[0].find_bit(q).index for q in reference_instruction.qubits]
            if measurement_started and reference_op.name != "measure":
                raise NotImplementedError("Batched Qiskit execution only supports terminal measurements.")

            if reference_op.name == "reset":
                if touched_qubits & set(q_indices):
                    raise NotImplementedError("Batched Qiskit execution does not support runtime reset.")
                for circuit in circuits[1:]:
                    instruction = circuit.data[position]
                    current_op = instruction.operation
                    current_q_indices = [circuit.find_bit(q).index for q in instruction.qubits]
                    if current_op.name != "reset" or current_q_indices != q_indices:
                        raise NotImplementedError("Batched Qiskit execution requires identical reset layout.")
                    if _instruction_condition(instruction) is not None:
                        raise NotImplementedError("Batched Qiskit execution does not support conditioned operations.")
                continue

            if reference_op.name == "measure":
                measurement_started = True
                if len(q_indices) != 1 or len(reference_instruction.clbits) != 1:
                    raise NotImplementedError("Batched Qiskit execution supports one-qubit measure instructions.")
                c_index = circuits[0].find_bit(reference_instruction.clbits[0]).index
                for circuit in circuits[1:]:
                    instruction = circuit.data[position]
                    current_op = instruction.operation
                    current_q_indices = [circuit.find_bit(q).index for q in instruction.qubits]
                    current_c_index = circuit.find_bit(instruction.clbits[0]).index if instruction.clbits else None
                    if (
                        current_op.name != "measure"
                        or current_q_indices != q_indices
                        or current_c_index != c_index
                    ):
                        raise NotImplementedError("Batched Qiskit execution requires identical measurement layout.")
                    if _instruction_condition(instruction) is not None:
                        raise NotImplementedError("Batched Qiskit execution does not support conditioned operations.")
                measured_bits[c_index] = q_indices[0]
                continue

            ops = []
            params_by_circuit = []
            for circuit in circuits:
                instruction = circuit.data[position]
                op = instruction.operation
                current_q_indices = [circuit.find_bit(q).index for q in instruction.qubits]
                if op.name != reference_op.name or current_q_indices != q_indices:
                    raise NotImplementedError("Batched Qiskit execution requires identical operation layout.")
                if _instruction_condition(instruction) is not None:
                    raise NotImplementedError("Batched Qiskit execution does not support conditioned operations.")
                ops.append(op)

            if reference_op.name in {"initialize", "state_preparation"}:
                if touched_qubits or len(q_indices) != num_qubits:
                    raise NotImplementedError(
                        "Batched Qiskit execution only supports initial full-wire state preparation."
                    )
                statevectors = []
                for op in ops:
                    statevector = _state_preparation_vector(op)
                    if statevector is None:
                        raise NotImplementedError("Batched Qiskit execution requires state preparation vectors.")
                    statevectors.append(statevector_to_little_endian_wires(statevector))
                self._runtime.set_statevectors(statevectors)
                touched_qubits.update(q_indices)
                continue

            params_by_circuit = [list(getattr(op, "params", []) or []) for op in ops]
            try:
                normalized_params_by_circuit = [normalize_params(params) for params in params_by_circuit]
            except (TypeError, ValueError):
                normalized_params_by_circuit = None

            if normalized_params_by_circuit is not None and all(
                len(params) == 1 for params in normalized_params_by_circuit
            ):
                thetas = [params[0] for params in normalized_params_by_circuit]
                if self._apply_controlled_base_gate_batch(reference_op, q_indices, thetas):
                    touched_qubits.update(q_indices)
                    continue

            if normalized_params_by_circuit is not None and self._apply_controlled_u_gate_batch(
                reference_op,
                q_indices,
                normalized_params_by_circuit,
            ):
                touched_qubits.update(q_indices)
                continue

            if normalized_params_by_circuit is not None and self._apply_controlled_r_gate_batch(
                reference_op,
                q_indices,
                normalized_params_by_circuit,
            ):
                touched_qubits.update(q_indices)
                continue

            if normalized_params_by_circuit is not None and self._apply_controlled_u2_u3_gate_batch(
                reference_op,
                q_indices,
                normalized_params_by_circuit,
            ):
                touched_qubits.update(q_indices)
                continue

            if normalized_params_by_circuit is not None and reference_op.name in {
                "rx", "ry", "rz", "crx", "cry", "crz"
            } and all(
                len(params) == 1 for params in normalized_params_by_circuit
            ):
                self._runtime.apply_operation_batch(
                    reference_op.name,
                    q_indices,
                    [params[0] for params in normalized_params_by_circuit],
                )
                touched_qubits.update(q_indices)
                continue

            if normalized_params_by_circuit is not None and reference_op.name in {
                "p", "cp", "rxx", "ryy", "rzz", "rzx"
            } and all(
                len(params) == 1 for params in normalized_params_by_circuit
            ):
                thetas = [params[0] for params in normalized_params_by_circuit]
                if reference_op.name == "p":
                    self._apply_phase_gate_batch(q_indices, thetas)
                elif reference_op.name == "cp":
                    self._apply_controlled_phase_gate_batch(q_indices, thetas)
                elif reference_op.name == "rxx":
                    self._apply_rxx_gate_batch(q_indices, thetas)
                elif reference_op.name == "ryy":
                    self._apply_ryy_gate_batch(q_indices, thetas)
                elif reference_op.name == "rzz":
                    self._apply_rzz_gate_batch(q_indices, thetas)
                else:
                    self._apply_rzx_gate_batch(q_indices, thetas)
                touched_qubits.update(q_indices)
                continue

            if normalized_params_by_circuit is not None and reference_op.name == "PauliEvolution" and all(
                len(params) == 1 for params in normalized_params_by_circuit
            ):
                terms = _pauli_evolution_terms(reference_op)
                if terms is not None and all(_pauli_evolution_terms(op) == terms for op in ops[1:]):
                    times = [params[0] for params in normalized_params_by_circuit]
                    if self._apply_pauli_evolution_gate_batch(reference_op, q_indices, times):
                        touched_qubits.update(q_indices)
                        continue

            if normalized_params_by_circuit is not None and reference_op.name == "u" and all(
                len(params) == 3 for params in normalized_params_by_circuit
            ):
                thetas = [params[0] for params in normalized_params_by_circuit]
                phis = [params[1] for params in normalized_params_by_circuit]
                lams = [params[2] for params in normalized_params_by_circuit]
                self._apply_u_gate_batch(q_indices, thetas, phis, lams)
                touched_qubits.update(q_indices)
                continue

            if normalized_params_by_circuit is not None and reference_op.name == "u1" and all(
                len(params) == 1 for params in normalized_params_by_circuit
            ):
                self._apply_phase_gate_batch(q_indices, [params[0] for params in normalized_params_by_circuit])
                touched_qubits.update(q_indices)
                continue

            if normalized_params_by_circuit is not None and reference_op.name == "u2" and all(
                len(params) == 2 for params in normalized_params_by_circuit
            ):
                phis = [params[0] for params in normalized_params_by_circuit]
                lams = [params[1] for params in normalized_params_by_circuit]
                self._apply_u_gate_batch(q_indices, [cmath.pi / 2 for _ in phis], phis, lams)
                touched_qubits.update(q_indices)
                continue

            if normalized_params_by_circuit is not None and reference_op.name == "u3" and all(
                len(params) == 3 for params in normalized_params_by_circuit
            ):
                thetas = [params[0] for params in normalized_params_by_circuit]
                phis = [params[1] for params in normalized_params_by_circuit]
                lams = [params[2] for params in normalized_params_by_circuit]
                self._apply_u_gate_batch(q_indices, thetas, phis, lams)
                touched_qubits.update(q_indices)
                continue

            if normalized_params_by_circuit is not None and reference_op.name == "r" and all(
                len(params) == 2 for params in normalized_params_by_circuit
            ):
                thetas = [params[0] for params in normalized_params_by_circuit]
                phis = [params[1] for params in normalized_params_by_circuit]
                self._apply_r_gate_batch(q_indices, thetas, phis)
                touched_qubits.update(q_indices)
                continue

            if normalized_params_by_circuit is not None and reference_op.name in {
                "xx_plus_yy", "xx_minus_yy"
            } and all(
                len(params) == 2 for params in normalized_params_by_circuit
            ):
                thetas = [params[0] for params in normalized_params_by_circuit]
                betas = [params[1] for params in normalized_params_by_circuit]
                if reference_op.name == "xx_plus_yy":
                    self._apply_xx_plus_yy_gate_batch(q_indices, thetas, betas)
                else:
                    self._apply_xx_minus_yy_gate_batch(q_indices, thetas, betas)
                touched_qubits.update(q_indices)
                continue

            first_params = params_by_circuit[0]
            if any(not _parameter_lists_match(params, first_params) for params in params_by_circuit[1:]):
                raise NotImplementedError(
                    "Batched Qiskit execution only supports varying RX/RY/RZ, CRX/CRY/CRZ, "
                    f"fixed unitary/controlled-unitary operations, open-control controlled rotations/phase, "
                    f"p/cp, u/u1/u2/u3/r/cu/cu1/cu3, decomposed rxx/ryy/rzz/rzx/xx_plus_yy/xx_minus_yy, and PauliEvolution parameters, "
                    f"got {reference_op.name!r}."
                )

            self._apply_quantum_operation(
                reference_op,
                q_indices,
                include_global_phase=False,
                touched_qubits=touched_qubits,
                circuit_num_qubits=num_qubits,
            )

        return measured_bits

    @staticmethod
    def _requests_statevector(circuit):
        return any(instruction.operation.name == "save_statevector" for instruction in circuit.data)

    @staticmethod
    def _has_dynamic_circuit(circuit):
        measurement_started = False
        for instruction in circuit.data:
            op = instruction.operation
            if op.name in {"barrier", "delay", "global_phase", "save_statevector"}:
                continue
            if op.name in CONTROL_FLOW_OPS or getattr(op, "blocks", None) is not None:
                return True
            if _instruction_condition(instruction) is not None:
                return True
            if measurement_started and op.name != "measure":
                return True
            if op.name == "measure":
                measurement_started = True
        return False

    def _measure_qubit_for_trajectory(self, qubit):
        measure_qubit = getattr(self._runtime, "measure_qubit", None)
        if not callable(measure_qubit):
            raise NotImplementedError(
                "Dynamic Qiskit sampling requires a state-collapsing measure_qubit binding."
            )
        return int(measure_qubit(int(qubit)))

    def _apply_control_flow_block(
        self,
        block,
        instruction,
        circuit,
        classical_bits,
        measured_bits,
        touched_qubits,
        *,
        include_global_phase=False,
        max_dynamic_loop_iterations=DEFAULT_MAX_DYNAMIC_LOOP_ITERATIONS,
    ):
        for block_instruction in block.data:
            mapped_qargs = [
                instruction.qubits[block.find_bit(qubit).index]
                for qubit in block_instruction.qubits
            ]
            mapped_cargs = [
                instruction.clbits[block.find_bit(clbit).index]
                for clbit in block_instruction.clbits
            ]
            mapped_instruction = block_instruction.replace(qubits=tuple(mapped_qargs), clbits=tuple(mapped_cargs))
            signal = self._apply_trajectory_instruction(
                mapped_instruction,
                circuit,
                classical_bits,
                measured_bits,
                touched_qubits,
                include_global_phase=include_global_phase,
                max_dynamic_loop_iterations=max_dynamic_loop_iterations,
            )
            if signal is not None:
                return signal

    def _apply_trajectory_instruction(
        self,
        instruction,
        circuit,
        classical_bits,
        measured_bits,
        touched_qubits,
        *,
        include_global_phase=False,
        max_dynamic_loop_iterations=DEFAULT_MAX_DYNAMIC_LOOP_ITERATIONS,
    ):
        op = instruction.operation
        if op.name in {"barrier", "delay", "global_phase", "save_statevector"}:
            return

        if op.name == "break_loop":
            return "break"
        if op.name == "continue_loop":
            return "continue"

        q_indices = [circuit.find_bit(q).index for q in instruction.qubits]
        c_indices = [circuit.find_bit(c).index for c in instruction.clbits]

        if op.name == "if_else" and getattr(op, "blocks", None) is not None:
            blocks = tuple(op.blocks)
            if len(blocks) not in {1, 2}:
                raise NotImplementedError("RocQuantumBackend supports if_else with one or two blocks.")
            branch_index = 0 if _condition_matches(op.condition, circuit, classical_bits) else 1
            if branch_index >= len(blocks):
                return

            block = blocks[branch_index]
            return self._apply_control_flow_block(
                block,
                instruction,
                circuit,
                classical_bits,
                measured_bits,
                touched_qubits,
                include_global_phase=include_global_phase,
                max_dynamic_loop_iterations=max_dynamic_loop_iterations,
            )
            return

        if op.name == "for_loop" and getattr(op, "blocks", None) is not None:
            blocks = tuple(op.blocks)
            if len(blocks) != 1:
                raise NotImplementedError("RocQuantumBackend supports for_loop with exactly one body block.")
            loop_values, loop_parameter = _for_loop_metadata(op)
            for loop_value in loop_values:
                block = _bind_for_loop_block(blocks[0], loop_parameter, loop_value)
                signal = self._apply_control_flow_block(
                    block,
                    instruction,
                    circuit,
                    classical_bits,
                    measured_bits,
                    touched_qubits,
                    include_global_phase=include_global_phase,
                    max_dynamic_loop_iterations=max_dynamic_loop_iterations,
                )
                if signal == "break":
                    break
                if signal == "continue":
                    continue
                if signal is not None:
                    return signal
            return

        if op.name == "switch_case" and getattr(op, "blocks", None) is not None:
            cases = op.cases()
            if not hasattr(cases, "items"):
                raise NotImplementedError("RocQuantumBackend requires mapping-style Qiskit switch cases.")
            target_value = _classical_value(op.target, circuit, classical_bits)
            default_block = None
            selected_block = None
            for label, block in cases.items():
                if _is_default_switch_case(label):
                    default_block = block
                elif int(label) == target_value:
                    selected_block = block
                    break
            block = selected_block if selected_block is not None else default_block
            if block is None:
                return
            return self._apply_control_flow_block(
                block,
                instruction,
                circuit,
                classical_bits,
                measured_bits,
                touched_qubits,
                include_global_phase=include_global_phase,
                max_dynamic_loop_iterations=max_dynamic_loop_iterations,
            )
            return

        if op.name == "while_loop" and getattr(op, "blocks", None) is not None:
            blocks = tuple(op.blocks)
            if len(blocks) != 1:
                raise NotImplementedError("RocQuantumBackend supports while_loop with exactly one body block.")
            iterations = 0
            while _condition_matches(op.condition, circuit, classical_bits):
                if iterations >= int(max_dynamic_loop_iterations):
                    raise RuntimeError(
                        "RocQuantumBackend while_loop exceeded max_dynamic_loop_iterations."
                    )
                iterations += 1
                signal = self._apply_control_flow_block(
                    blocks[0],
                    instruction,
                    circuit,
                    classical_bits,
                    measured_bits,
                    touched_qubits,
                    include_global_phase=include_global_phase,
                    max_dynamic_loop_iterations=max_dynamic_loop_iterations,
                )
                if signal == "break":
                    break
                if signal == "continue":
                    continue
                if signal is not None:
                    return signal
            return

        if op.name in CONTROL_FLOW_OPS or getattr(op, "blocks", None) is not None:
            raise NotImplementedError(
                f"RocQuantumBackend dynamic sampling does not support {op.name!r} yet."
            )

        condition = _instruction_condition(instruction)
        if condition is not None and not _condition_matches(condition, circuit, classical_bits):
            return

        if op.name == "measure":
            if len(q_indices) != 1 or len(c_indices) != 1:
                raise NotImplementedError("Dynamic Qiskit sampling supports one-qubit measure instructions.")
            outcome = self._measure_qubit_for_trajectory(q_indices[0])
            classical_bits[c_indices[0]] = outcome
            measured_bits[c_indices[0]] = q_indices[0]
            return

        if op.name == "reset":
            for target in q_indices:
                if target in touched_qubits:
                    self._runtime.reset_qubit(target)
            touched_qubits.update(q_indices)
            return

        if op.name == "initialize":
            reset_targets = set(q_indices)
            if touched_qubits & reset_targets:
                raise ValueError(
                    "RocQuantumBackend only supports initialize before a qubit has been operated on. "
                    "Later initialize instructions require non-unitary reset support."
                )
            statevector = _state_preparation_vector(op)
            if statevector is not None and len(q_indices) == circuit.num_qubits:
                try:
                    self._runtime.set_statevector(
                        statevector_to_little_endian_wires(statevector)
                    )
                except (NotImplementedError, RuntimeError, TypeError, ValueError):
                    self._runtime.apply_operation(
                        "state_preparation",
                        q_indices,
                        matrix=_state_preparation_matrix(op),
                    )
            else:
                self._runtime.apply_operation(
                    "state_preparation",
                    q_indices,
                    matrix=_state_preparation_matrix(op),
                )
            touched_qubits.update(q_indices)
            return

        self._apply_quantum_operation(
            op,
            q_indices,
            include_global_phase=include_global_phase,
            touched_qubits=touched_qubits,
            circuit_num_qubits=circuit.num_qubits,
        )

    def _apply_circuit_trajectory(self, circuit, *, max_dynamic_loop_iterations=DEFAULT_MAX_DYNAMIC_LOOP_ITERATIONS):
        self._ensure_simulator(circuit.num_qubits)
        classical_bits = {}
        measured_bits = {}
        touched_qubits = set()
        for instruction in circuit.data:
            signal = self._apply_trajectory_instruction(
                instruction,
                circuit,
                classical_bits,
                measured_bits,
                touched_qubits,
                max_dynamic_loop_iterations=max_dynamic_loop_iterations,
            )
            if signal is not None:
                raise RuntimeError(
                    f"RocQuantumBackend encountered {signal}_loop outside an active loop."
                )
        return measured_bits, classical_bits

    def estimate_expectation(self, circuit, observable):
        """Return a native expectation for a circuit and Qiskit observable."""
        if self._has_runtime_reset(circuit) or self._has_dynamic_circuit(circuit):
            raise ValueError(
                "RocQuantumBackend estimator does not support shot-trajectory circuits yet. "
                "Use backend.run(..., sampling=True) for shot-by-shot sampling."
            )
        self._apply_circuit(circuit, include_global_phase=False)
        return estimate_observable(self._runtime, observable, circuit.num_qubits)

    def _run_runtime_reset_sampling(self, circuit, shots, memory_enabled):
        memory = []
        measured_items = None
        memory_width = getattr(circuit, "num_clbits", 0)

        for _ in range(int(shots)):
            measured_bits = self._apply_circuit(
                circuit,
                include_global_phase=False,
                allow_runtime_reset=True,
            )
            current_items = (
                sorted(measured_bits.items())
                if measured_bits
                else [(idx, idx) for idx in range(circuit.num_qubits)]
            )
            if measured_items is None:
                measured_items = current_items

            qubits_to_measure, sample_offsets = qiskit_sample_plan(current_items)
            raw_sample = self._runtime.measure(qubits_to_measure, 1)
            width = memory_width or len(current_items)
            memory.extend(qiskit_memory_from_samples(raw_sample, current_items, width, sample_offsets))

        formatted_counts = counts_from_memory(memory)
        return {
            "counts": formatted_counts,
            "memory": memory if memory_enabled else None,
        }

    def _run_dynamic_sampling(self, circuit, shots, memory_enabled, max_dynamic_loop_iterations):
        memory = []
        measured_items = None
        memory_width = getattr(circuit, "num_clbits", 0)

        for _ in range(int(shots)):
            measured_bits, classical_bits = self._apply_circuit_trajectory(
                circuit,
                max_dynamic_loop_iterations=max_dynamic_loop_iterations,
            )
            current_items = (
                sorted(measured_bits.items())
                if measured_bits
                else [(idx, idx) for idx in range(circuit.num_qubits)]
            )
            if measured_items is None:
                measured_items = current_items

            width = memory_width or len(current_items)
            if measured_bits:
                memory.append(_memory_from_classical_bits(classical_bits, current_items, width))
            else:
                qubits_to_measure, sample_offsets = qiskit_sample_plan(current_items)
                raw_sample = self._runtime.measure(qubits_to_measure, 1)
                memory.extend(qiskit_memory_from_samples(raw_sample, current_items, width, sample_offsets))

        formatted_counts = counts_from_memory(memory)
        return {
            "counts": formatted_counts,
            "memory": memory if memory_enabled else None,
        }

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
            return_sampling = bool(options.get("sampling", self.options.sampling))
            has_runtime_reset = self._has_runtime_reset(circuit)
            has_dynamic_circuit = self._has_dynamic_circuit(circuit)

            if has_dynamic_circuit:
                if return_statevector:
                    raise ValueError(
                        "RocQuantumBackend cannot return a single statevector for shot-trajectory circuits."
                    )
                if not return_sampling:
                    raise ValueError("RocQuantumBackend dynamic circuits require sampling=True.")

                data = self._run_dynamic_sampling(
                    circuit,
                    shots,
                    bool(options.get("memory", self.options.memory)),
                    int(options.get(
                        "max_dynamic_loop_iterations",
                        self.options.max_dynamic_loop_iterations,
                    )),
                )
                exp_data = ExperimentResultData(**data)
                results.append(
                    ExperimentResult(
                        shots=shots,
                        success=True,
                        data=exp_data,
                        header=getattr(
                            circuit,
                            "header",
                            {"name": getattr(circuit, "name", None), "metadata": getattr(circuit, "metadata", None)},
                        ),
                    )
                )
                continue

            if has_runtime_reset:
                if return_statevector:
                    raise ValueError(
                        "RocQuantumBackend cannot return a single statevector for circuits with "
                        "runtime reset because reset is a non-unitary shot-trajectory operation."
                    )
                if not return_sampling:
                    raise ValueError("RocQuantumBackend runtime reset requires sampling=True.")

                data = self._run_runtime_reset_sampling(
                    circuit,
                    shots,
                    bool(options.get("memory", self.options.memory)),
                )
                exp_data = ExperimentResultData(**data)
                results.append(
                    ExperimentResult(
                        shots=shots,
                        success=True,
                        data=exp_data,
                        header=getattr(
                            circuit,
                            "header",
                            {"name": getattr(circuit, "name", None), "metadata": getattr(circuit, "metadata", None)},
                        ),
                    )
                )
                continue

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
