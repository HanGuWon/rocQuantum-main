# Copyright (c) 2025-2026, rocQuantum Developers.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Core abstract classes for the rocQuantum Quantum Error Correction (QEC) framework.

This module defines the blueprints for specifying error-correcting codes,
decoders, and the experimental orchestrator. The design is fundamentally based
on a "Circuit Fragmentation" strategy.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Callable, Any, Optional, Tuple

# --- rocQuantum Imports ---
try:
    import rocq
    from rocq.operator import PauliOperator
    from rocq.kernel import QuantumKernel
except ImportError:
    rocq = None  # type: ignore
    PauliOperator = None  # type: ignore
    QuantumKernel = None  # type: ignore

# --- Type Hinting Definitions ---
AnsatzKernel = Callable[..., None]


class QuantumErrorCode(ABC):
    """Abstract base class for defining a quantum error-correcting code."""
    @abstractmethod
    def generate_stabilizer_circuits(
        self,
        initial_state_kernel: AnsatzKernel,
        num_qubits: int,
        backend: str = "state_vector"
    ) -> List[Any]:
        """Generates the sequence of circuits for measuring each stabilizer."""
        pass

    @abstractmethod
    def define_logical_operators(self) -> Dict[str, PauliOperator]:
        """Defines the logical operators for the code."""
        pass

class Decoder(ABC):
    """Abstract base class for a QEC decoder."""
    @abstractmethod
    def decode(self, syndrome: List[int]) -> PauliOperator:
        """Processes a syndrome to determine the required correction."""
        pass

class QEC_Experiment:
    """Orchestrates a QEC experiment using a "Circuit Fragmentation" strategy."""
    def __init__(self, backend: str = "state_vector"):
        if rocq is None:
            raise RuntimeError(
                "Canonical 'rocq' package is not available. Install the Python package "
                "before running QEC experiments."
            )
        self.backend = backend

    def run_single_round(
        self,
        code: QuantumErrorCode,
        decoder: Decoder,
        initial_state_kernel: AnsatzKernel,
        num_qubits: int,
        ancilla_qubit_indices: List[int]
    ) -> Dict[str, Any]:
        """Executes a single, complete round of quantum error correction."""
        print("Step 1: Generating stabilizer measurement circuit fragments...")
        stabilizer_circuits = code.generate_stabilizer_circuits(
            initial_state_kernel, num_qubits, self.backend
        )
        if len(stabilizer_circuits) != len(ancilla_qubit_indices):
            raise ValueError(
                "Number of ancilla_qubit_indices must match the number of generated "
                "stabilizer circuits."
            )

        print("Step 2: Measuring syndrome by executing each fragment...")
        syndrome = []
        for i, stab_program in enumerate(stabilizer_circuits):
            ancilla_idx = ancilla_qubit_indices[i]
            if not hasattr(stab_program, "circuit_ref") or not hasattr(stab_program.circuit_ref, "measure"):
                raise NotImplementedError(
                    "QEC fragment execution requires 'circuit_ref.measure(...)' support. "
                    "The canonical backend bridge is not fully wired yet."
                )
            outcome, _ = stab_program.circuit_ref.measure(ancilla_idx)
            print(f"  - Measured stabilizer {i} on ancilla q[{ancilla_idx}]: outcome = {outcome}")
            syndrome.append(outcome)

        print(f"\nStep 3: Decoding syndrome {syndrome}...")
        correction_operator = decoder.decode(syndrome)
        print(f"  - Decoder determined correction: {correction_operator}")

        # Note: Final state calculation can be added here if needed.
        # For now, the primary goal is verifying the syndrome and correction.

        return {
            "syndrome": syndrome,
            "correction_applied": str(correction_operator),
            "logical_operators": code.define_logical_operators(),
        }


def _validate_repetition_bits(initial_bits: Optional[List[int]]) -> List[int]:
    bits = list(initial_bits or [0, 0, 0])
    if len(bits) != 3 or any(bit not in (0, 1) for bit in bits):
        raise ValueError("initial_bits must be a length-3 list containing only 0 or 1.")
    return bits


def _validate_error_qubit(error_qubit: Optional[int]) -> None:
    if error_qubit is not None and error_qubit not in (0, 1, 2):
        raise ValueError("error_qubit must be one of 0, 1, 2, or None.")


def _validate_repetition_rounds(rounds: int) -> None:
    if rounds <= 0:
        raise ValueError("rounds must be positive.")


def _validate_error_qubit_schedule(
    error_qubits: Optional[List[Optional[int]]],
    rounds: int,
) -> List[Optional[int]]:
    if error_qubits is None:
        return [None] * rounds
    schedule = list(error_qubits)
    if len(schedule) != rounds:
        raise ValueError("error_qubits length must match rounds.")
    for error_qubit in schedule:
        _validate_error_qubit(error_qubit)
    return schedule


def _syndrome_from_bitstring(bitstring: str) -> List[int]:
    if len(bitstring) < 2:
        bitstring = bitstring.zfill(2)
    # rocq packs measured qubits in request order, then formats the integer as
    # a big-endian bitstring. For qubits [3, 4], q3 is the rightmost bit.
    return [int(bitstring[-1]), int(bitstring[-2])]


def _syndrome_key(syndrome: List[int]) -> str:
    return f"{syndrome[0]}{syndrome[1]}"


def _correction_qubit_for_syndrome(syndrome: List[int]) -> Optional[int]:
    lookup: Dict[Tuple[int, int], Optional[int]] = {
        (0, 0): None,
        (1, 0): 0,
        (1, 1): 1,
        (0, 1): 2,
    }
    return lookup.get((syndrome[0], syndrome[1]))


def _apply_known_bit_flip(bits: List[int], qubit: Optional[int]) -> List[int]:
    corrected = list(bits)
    if qubit is not None:
        corrected[qubit] ^= 1
    return corrected


def _logical_successful_shots(analysis: Dict[str, Any]) -> int:
    return sum(
        outcome["count"]
        for outcome in analysis["decoded_outcomes"]
        if outcome["logical_success"] is True
    )


def repetition_syndrome_histogram(counts: Dict[str, int]) -> Dict[str, int]:
    """Aggregate raw ancilla bitstring counts into repetition-code syndromes."""
    if not counts:
        raise ValueError("No syndrome samples were produced.")

    histogram = {"00": 0, "10": 0, "11": 0, "01": 0}
    for bitstring, count in counts.items():
        if not isinstance(bitstring, str) or any(bit not in "01" for bit in bitstring):
            raise ValueError("counts keys must be binary strings.")
        if not isinstance(count, int) or count < 0:
            raise ValueError("counts values must be non-negative integers.")
        syndrome = _syndrome_from_bitstring(bitstring)
        histogram[_syndrome_key(syndrome)] += count
    return histogram


def _most_likely_syndrome(counts: Dict[str, int]) -> List[int]:
    histogram = repetition_syndrome_histogram(counts)
    syndrome_key = max(histogram.items(), key=lambda item: item[1])[0]
    return [int(syndrome_key[0]), int(syndrome_key[1])]


def analyze_repetition_code_counts(
    counts: Dict[str, int],
    initial_bits: Optional[List[int]] = None,
    error_qubit: Optional[int] = None,
    expected_logical_bit: Optional[int] = None,
) -> Dict[str, Any]:
    """Decode sampled repetition-code syndromes and summarize correction quality."""
    bits = _validate_repetition_bits(initial_bits)
    _validate_error_qubit(error_qubit)
    if expected_logical_bit is not None and expected_logical_bit not in (0, 1):
        raise ValueError("expected_logical_bit must be 0, 1, or None.")

    encoded_logical_bit = bits[0] if bits.count(bits[0]) == 3 else None
    if expected_logical_bit is None:
        expected_logical_bit = encoded_logical_bit

    observed_bits = _apply_known_bit_flip(bits, error_qubit)
    histogram = repetition_syndrome_histogram(counts)
    total_shots = sum(histogram.values())
    if total_shots <= 0:
        raise ValueError("Syndrome sample counts must contain at least one shot.")

    decoded_outcomes = []
    successful_shots = 0
    for syndrome_key, count in histogram.items():
        syndrome = [int(syndrome_key[0]), int(syndrome_key[1])]
        correction_qubit = _correction_qubit_for_syndrome(syndrome)
        corrected_bits = _apply_known_bit_flip(observed_bits, correction_qubit)
        success = None
        if expected_logical_bit is not None:
            success = corrected_bits == [expected_logical_bit] * 3
            if success:
                successful_shots += count
        decoded_outcomes.append(
            {
                "syndrome": syndrome,
                "count": count,
                "correction_qubit": correction_qubit,
                "corrected_data_bits": corrected_bits,
                "logical_success": success,
            }
        )

    most_likely_syndrome = _most_likely_syndrome(counts)
    most_likely = next(
        outcome
        for outcome in decoded_outcomes
        if outcome["syndrome"] == most_likely_syndrome
    )
    logical_success_rate = None
    if expected_logical_bit is not None:
        logical_success_rate = successful_shots / total_shots

    return {
        "syndrome_histogram": histogram,
        "total_shots": total_shots,
        "initial_data_bits": bits,
        "observed_data_bits": observed_bits,
        "expected_logical_bit": expected_logical_bit,
        "decoded_outcomes": decoded_outcomes,
        "most_likely_syndrome": most_likely_syndrome,
        "most_likely_correction_qubit": most_likely["correction_qubit"],
        "most_likely_corrected_data_bits": most_likely["corrected_data_bits"],
        "logical_success_rate": logical_success_rate,
    }


def analyze_repetition_code_rounds(
    round_counts: List[Dict[str, int]],
    initial_bits: Optional[List[int]] = None,
    error_qubits: Optional[List[Optional[int]]] = None,
    expected_logical_bit: Optional[int] = None,
) -> Dict[str, Any]:
    """Aggregate repeated 3-qubit repetition-code syndrome rounds.

    Each round is decoded with the same lookup-table repetition decoder model
    as ``analyze_repetition_code_counts``. The most likely correction from one
    round feeds the next round's known data-bit state, giving a small
    classical feed-forward workflow over the existing sampled helper.
    """
    if round_counts is None:
        raise ValueError("round_counts must contain at least one round.")
    counts_by_round = list(round_counts)
    if not counts_by_round:
        raise ValueError("round_counts must contain at least one round.")
    _validate_repetition_rounds(len(counts_by_round))
    error_schedule = _validate_error_qubit_schedule(error_qubits, len(counts_by_round))

    initial_data_bits = _validate_repetition_bits(initial_bits)
    expected_for_rounds = expected_logical_bit
    if expected_for_rounds is None and initial_data_bits.count(initial_data_bits[0]) == 3:
        expected_for_rounds = initial_data_bits[0]
    current_bits = list(initial_data_bits)
    aggregate_histogram = {"00": 0, "10": 0, "11": 0, "01": 0}
    round_results = []
    total_shots = 0
    successful_shots = 0
    success_is_known = False
    correction_summary = {"none": 0, "q0": 0, "q1": 0, "q2": 0}

    for round_index, counts in enumerate(counts_by_round):
        if not isinstance(counts, dict):
            raise ValueError("round_counts entries must be count dictionaries.")

        analysis = analyze_repetition_code_counts(
            counts,
            initial_bits=current_bits,
            error_qubit=error_schedule[round_index],
            expected_logical_bit=expected_for_rounds,
        )
        for syndrome_key, count in analysis["syndrome_histogram"].items():
            aggregate_histogram[syndrome_key] += count

        total_shots += analysis["total_shots"]
        if analysis["logical_success_rate"] is not None:
            success_is_known = True
            successful_shots += _logical_successful_shots(analysis)

        correction_qubit = analysis["most_likely_correction_qubit"]
        correction_key = "none" if correction_qubit is None else f"q{correction_qubit}"
        correction_summary[correction_key] += 1
        current_bits = analysis["most_likely_corrected_data_bits"]
        round_results.append(
            {
                "round_index": round_index,
                "counts": dict(counts),
                "injected_error_qubit": error_schedule[round_index],
                "syndrome": analysis["most_likely_syndrome"],
                "correction_qubit": correction_qubit,
                "corrected_data_bits": current_bits,
                "logical_success_rate": analysis["logical_success_rate"],
                "analysis": analysis,
            }
        )

    logical_success_rate = None
    if success_is_known:
        logical_success_rate = successful_shots / total_shots

    return {
        "rounds": len(counts_by_round),
        "round_results": round_results,
        "aggregate_syndrome_histogram": aggregate_histogram,
        "correction_summary": correction_summary,
        "total_shots": total_shots,
        "initial_data_bits": initial_data_bits,
        "final_data_bits": current_bits,
        "error_qubits": error_schedule,
        "expected_logical_bit": expected_for_rounds,
        "logical_success_rate": logical_success_rate,
        "experimental_supported_subset": "three_qubit_repetition_repeated_rounds",
    }


def run_repetition_code_single_round(
    initial_bits: Optional[List[int]] = None,
    error_qubit: Optional[int] = None,
    shots: int = 1,
    backend: str = "state_vector",
) -> Dict[str, Any]:
    """Run one experimental 3-qubit bit-flip repetition-code syndrome round.

    This helper uses 3 data qubits plus two ancillas and samples the two
    stabilizer ancillas at the end of the circuit. It is intentionally a small
    executable subset, not a full fault-tolerant QEC framework.
    """
    if rocq is None:
        raise RuntimeError(
            "Canonical 'rocq' package is not available. Install the Python package "
            "before running QEC experiments."
        )
    if shots <= 0:
        raise ValueError("shots must be positive.")

    bits = _validate_repetition_bits(initial_bits)
    _validate_error_qubit(error_qubit)

    @rocq.kernel
    def repetition_round():
        q = rocq.qvec(5)
        for idx, bit in enumerate(bits):
            if bit:
                rocq.x(q[idx])
        if error_qubit is not None:
            rocq.x(q[error_qubit])

        rocq.cnot(q[0], q[3])
        rocq.cnot(q[1], q[3])
        rocq.cnot(q[1], q[4])
        rocq.cnot(q[2], q[4])

    counts = rocq.sample(repetition_round, shots, backend=backend, qubits=[3, 4])
    syndrome = _most_likely_syndrome(counts)
    analysis = analyze_repetition_code_counts(counts, bits, error_qubit=error_qubit)

    from rocquantum.qec.decoders.repetition_decoder import RepetitionCodeDecoder

    correction = RepetitionCodeDecoder().decode(syndrome)
    return {
        "syndrome": syndrome,
        "counts": counts,
        "correction": correction,
        "correction_applied": correction.to_string(),
        "analysis": analysis,
        "logical_success_rate": analysis["logical_success_rate"],
        "most_likely_corrected_data_bits": analysis["most_likely_corrected_data_bits"],
        "experimental_supported_subset": "three_qubit_repetition_single_round",
    }


def run_repetition_code_rounds(
    initial_bits: Optional[List[int]] = None,
    error_qubits: Optional[List[Optional[int]]] = None,
    rounds: int = 1,
    shots: int = 1,
    backend: str = "state_vector",
) -> Dict[str, Any]:
    """Run repeated experimental 3-qubit repetition-code syndrome rounds."""
    _validate_repetition_rounds(rounds)
    if shots <= 0:
        raise ValueError("shots must be positive.")

    bits = _validate_repetition_bits(initial_bits)
    error_schedule = _validate_error_qubit_schedule(error_qubits, rounds)
    current_bits = list(bits)
    round_counts = []

    for round_index in range(rounds):
        result = run_repetition_code_single_round(
            initial_bits=current_bits,
            error_qubit=error_schedule[round_index],
            shots=shots,
            backend=backend,
        )
        round_counts.append(result["counts"])
        current_bits = result["most_likely_corrected_data_bits"]

    analysis = analyze_repetition_code_rounds(
        round_counts,
        initial_bits=bits,
        error_qubits=error_schedule,
    )
    analysis["shots_per_round"] = shots
    analysis["backend"] = backend
    return analysis
