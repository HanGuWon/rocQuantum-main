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
from collections.abc import Mapping
import math
from numbers import Integral, Real
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

_REPETITION_SYNDROME_KEYS = ("00", "10", "11", "01")
_FALLBACK_SUPPORTED_BACKENDS = (
    "state_vector",
    "density_matrix",
    "stabilizer",
    "tableau",
    "clifford",
)


def _validate_binary_count_key(bitstring: str, label: str) -> str:
    if (
        not isinstance(bitstring, str)
        or not bitstring
        or any(bit not in "01" for bit in bitstring)
    ):
        raise ValueError(f"{label} keys must be non-empty binary strings.")
    return bitstring


def _validate_single_ancilla_count_key(bitstring: str) -> str:
    bitstring = _validate_binary_count_key(bitstring, "Ancilla sample counts")
    if len(bitstring) != 1:
        raise ValueError("Ancilla sample counts keys must contain exactly one bit.")
    return bitstring


def _validate_nonnegative_count(count: int, label: str) -> int:
    if isinstance(count, bool) or not isinstance(count, Integral):
        raise ValueError(f"{label} values must be non-negative integers.")
    value = int(count)
    if value < 0:
        raise ValueError(f"{label} values must be non-negative integers.")
    return value


def _validate_binary_bit(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be 0 or 1.")
    bit = int(value)
    if bit not in (0, 1):
        raise ValueError(f"{name} must be 0 or 1.")
    return bit


def _validate_optional_binary_bit(value: Optional[int], name: str) -> Optional[int]:
    if value is None:
        return None
    return _validate_binary_bit(value, name)


def _validate_optional_measurement_error_probability(
    value: Optional[float],
) -> Optional[float]:
    message = "measurement_error_probability must be a finite real number in [0, 0.5)."
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(message)
    probability = float(value)
    if not math.isfinite(probability) or probability < 0.0 or probability >= 0.5:
        raise ValueError(message)
    return probability


def _supported_backend_names() -> Tuple[str, ...]:
    if rocq is None or not hasattr(rocq, "runtime_capabilities"):
        return _FALLBACK_SUPPORTED_BACKENDS
    capabilities = rocq.runtime_capabilities()
    return tuple(capabilities.get("supported_backends", _FALLBACK_SUPPORTED_BACKENDS))


def _validate_backend_name(backend: str) -> str:
    supported = _supported_backend_names()
    if not isinstance(backend, str) or backend not in supported:
        raise ValueError(f"backend must be one of: {list(supported)}.")
    return backend


def _validate_initial_state_kernel(initial_state_kernel):
    if initial_state_kernel is not None and not callable(initial_state_kernel):
        raise ValueError("initial_state_kernel must be callable or None.")
    return initial_state_kernel


def _validate_boolean_option(value, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean.")
    return value


def _validate_code_and_decoder(code, decoder) -> None:
    if not callable(getattr(code, "generate_stabilizer_circuits", None)):
        raise ValueError("code must define a callable generate_stabilizer_circuits method.")
    if not callable(getattr(code, "define_logical_operators", None)):
        raise ValueError("code must define a callable define_logical_operators method.")
    if not callable(getattr(decoder, "decode", None)):
        raise ValueError("decoder must define a callable decode method.")


def _validate_stabilizer_circuits(stabilizer_circuits) -> List[Any]:
    if isinstance(stabilizer_circuits, (str, bytes, Mapping)):
        raise ValueError("generate_stabilizer_circuits must return a sequence of circuit fragments.")
    try:
        fragments = list(stabilizer_circuits)
    except TypeError as exc:
        raise ValueError(
            "generate_stabilizer_circuits must return a sequence of circuit fragments."
        ) from exc
    if not fragments:
        raise ValueError("generate_stabilizer_circuits must return at least one circuit fragment.")
    return fragments


def _validate_decoder_correction(correction_operator):
    if PauliOperator is not None and isinstance(correction_operator, PauliOperator):
        return correction_operator
    raise ValueError("decoder.decode must return a rocq.operator.PauliOperator correction.")


def _validate_logical_operators(logical_operators) -> Dict[str, Any]:
    if not isinstance(logical_operators, Mapping):
        raise ValueError(
            "define_logical_operators must return a mapping of names to "
            "rocq.operator.PauliOperator values."
        )

    validated = {}
    for name, operator in logical_operators.items():
        if not isinstance(name, str) or not name:
            raise ValueError("logical operator names must be non-empty strings.")
        if PauliOperator is None or not isinstance(operator, PauliOperator):
            raise ValueError("logical operator values must be rocq.operator.PauliOperator instances.")
        validated[name] = operator
    return validated


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


def _most_likely_single_bit(counts: Dict[str, int]) -> int:
    if not counts:
        raise ValueError("No ancilla samples were produced.")

    total_shots = 0
    for bitstring, count in counts.items():
        _validate_single_ancilla_count_key(bitstring)
        total_shots += _validate_nonnegative_count(count, "Ancilla sample counts")

    if total_shots <= 0:
        raise ValueError("Ancilla sample counts must contain at least one shot.")

    bitstring, _ = max(counts.items(), key=lambda item: item[1])
    return int(bitstring[-1])


def _validate_positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return int(value)


def _validate_ancilla_qubit_indices(ancilla_qubit_indices, num_qubits: int) -> List[int]:
    if isinstance(ancilla_qubit_indices, (bool, str, bytes)):
        raise ValueError("ancilla_qubit_indices must be a sequence of integer qubit indices.")
    try:
        raw_indices = list(ancilla_qubit_indices)
    except TypeError as exc:
        raise ValueError(
            "ancilla_qubit_indices must be a sequence of integer qubit indices."
        ) from exc

    normalized = []
    for index in raw_indices:
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise ValueError("ancilla_qubit_indices entries must be integer qubit indices.")
        qubit = int(index)
        if qubit < 0 or qubit >= num_qubits:
            raise ValueError("ancilla_qubit_indices entries must be in range for num_qubits.")
        normalized.append(qubit)
    if len(set(normalized)) != len(normalized):
        raise ValueError("ancilla_qubit_indices entries must be unique.")
    return normalized


class QEC_Experiment:
    """Orchestrates a QEC experiment using a "Circuit Fragmentation" strategy."""
    def __init__(self, backend: str = "state_vector", verbose: bool = False):
        if rocq is None:
            raise RuntimeError(
                "Canonical 'rocq' package is not available. Install the Python package "
                "before running QEC experiments."
            )
        self.backend = _validate_backend_name(backend)
        self.verbose = _validate_boolean_option(verbose, "verbose")

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def run_single_round(
        self,
        code: QuantumErrorCode,
        decoder: Decoder,
        initial_state_kernel: AnsatzKernel,
        num_qubits: int,
        ancilla_qubit_indices: List[int],
        shots: int = 1,
    ) -> Dict[str, Any]:
        """Executes a single, complete round of quantum error correction."""
        shots = _validate_positive_integer(shots, "shots")
        num_qubits = _validate_positive_integer(num_qubits, "num_qubits")
        ancilla_qubit_indices = _validate_ancilla_qubit_indices(
            ancilla_qubit_indices,
            num_qubits,
        )
        initial_state_kernel = _validate_initial_state_kernel(initial_state_kernel)
        _validate_code_and_decoder(code, decoder)

        self._log("Step 1: Generating stabilizer measurement circuit fragments...")
        stabilizer_circuits = code.generate_stabilizer_circuits(
            initial_state_kernel, num_qubits, self.backend
        )
        stabilizer_circuits = _validate_stabilizer_circuits(stabilizer_circuits)
        if len(stabilizer_circuits) != len(ancilla_qubit_indices):
            raise ValueError(
                "Number of ancilla_qubit_indices must match the number of generated "
                "stabilizer circuits."
            )
        logical_operators = _validate_logical_operators(code.define_logical_operators())

        self._log("Step 2: Measuring syndrome by executing each fragment...")
        syndrome = []
        for i, stab_program in enumerate(stabilizer_circuits):
            ancilla_idx = ancilla_qubit_indices[i]
            if hasattr(stab_program, "circuit_ref") and hasattr(stab_program.circuit_ref, "measure"):
                outcome, _ = stab_program.circuit_ref.measure(ancilla_idx)
                outcome = _validate_binary_bit(outcome, "ancilla measurement outcome")
            else:
                counts = rocq.sample(
                    stab_program,
                    shots,
                    backend=self.backend,
                    qubits=[ancilla_idx],
                )
                outcome = _most_likely_single_bit(counts)
            self._log(f"  - Measured stabilizer {i} on ancilla q[{ancilla_idx}]: outcome = {outcome}")
            syndrome.append(outcome)

        self._log(f"\nStep 3: Decoding syndrome {syndrome}...")
        correction_operator = _validate_decoder_correction(decoder.decode(syndrome))
        self._log(f"  - Decoder determined correction: {correction_operator}")
        correction_text = correction_operator.to_string()

        # Note: Final state calculation can be added here if needed.
        # For now, the primary goal is verifying the syndrome and correction.

        return {
            "syndrome": syndrome,
            "correction_applied": correction_text,
            "logical_operators": logical_operators,
            "shots": shots,
        }


def _validate_repetition_bits(initial_bits: Optional[List[int]]) -> List[int]:
    try:
        bits = [0, 0, 0] if initial_bits is None else list(initial_bits)
    except TypeError as exc:
        raise ValueError(
            "initial_bits must be a length-3 list containing only 0 or 1."
        ) from exc
    if len(bits) != 3:
        raise ValueError("initial_bits must be a length-3 list containing only 0 or 1.")
    return [_validate_binary_bit(bit, "initial_bits") for bit in bits]


def _validate_error_qubit(error_qubit: Optional[int]) -> Optional[int]:
    if error_qubit is None:
        return None
    if isinstance(error_qubit, bool) or not isinstance(error_qubit, Integral):
        raise ValueError("error_qubit must be one of 0, 1, 2, or None.")
    qubit = int(error_qubit)
    if qubit not in (0, 1, 2):
        raise ValueError("error_qubit must be one of 0, 1, 2, or None.")
    return qubit


def _validate_repetition_rounds(rounds: int) -> int:
    return _validate_positive_integer(rounds, "rounds")


def _validate_error_qubit_schedule(
    error_qubits: Optional[List[Optional[int]]],
    rounds: int,
) -> List[Optional[int]]:
    if error_qubits is None:
        return [None] * rounds
    try:
        schedule = list(error_qubits)
    except TypeError as exc:
        raise ValueError("error_qubits must be a list of 0, 1, 2, or None values.") from exc
    if len(schedule) != rounds:
        raise ValueError("error_qubits length must match rounds.")
    return [_validate_error_qubit(error_qubit) for error_qubit in schedule]


def _syndrome_from_bitstring(bitstring: str) -> List[int]:
    if len(bitstring) > 2:
        raise ValueError("Repetition-code syndrome count keys must contain at most two bits.")
    if len(bitstring) < 2:
        bitstring = bitstring.zfill(2)
    # rocq packs measured qubits in request order, then formats the integer as
    # a big-endian bitstring. For qubits [3, 4], q3 is the rightmost bit.
    return [int(bitstring[-1]), int(bitstring[-2])]


def _syndrome_key(syndrome: List[int]) -> str:
    return f"{syndrome[0]}{syndrome[1]}"


def _select_syndrome_from_scores(scores: Dict[str, float]) -> List[int]:
    syndrome_key = max(_REPETITION_SYNDROME_KEYS, key=lambda key: scores.get(key, 0.0))
    return [int(syndrome_key[0]), int(syndrome_key[1])]


def _inverse_measurement_weight(true_bit: str, observed_bit: str, probability: float) -> float:
    if probability == 0.0:
        return 1.0 if true_bit == observed_bit else 0.0
    scale = 1.0 / (1.0 - (2.0 * probability))
    return ((1.0 - probability) if true_bit == observed_bit else -probability) * scale


def _measurement_mitigated_syndrome_scores(
    histogram: Dict[str, int],
    measurement_error_probability: float,
) -> Dict[str, float]:
    probability = _validate_optional_measurement_error_probability(
        measurement_error_probability
    )
    if probability is None:
        raise ValueError("measurement_error_probability is required.")

    scores: Dict[str, float] = {}
    for true_key in _REPETITION_SYNDROME_KEYS:
        score = 0.0
        for observed_key, count in histogram.items():
            weight = (
                _inverse_measurement_weight(true_key[0], observed_key[0], probability)
                * _inverse_measurement_weight(true_key[1], observed_key[1], probability)
            )
            score += weight * count
        scores[true_key] = 0.0 if abs(score) < 1e-12 else score
    return scores


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

    histogram = {key: 0 for key in _REPETITION_SYNDROME_KEYS}
    for bitstring, count in counts.items():
        _validate_binary_count_key(bitstring, "counts")
        count = _validate_nonnegative_count(count, "counts")
        syndrome = _syndrome_from_bitstring(bitstring)
        histogram[_syndrome_key(syndrome)] += count
    return histogram


def mitigate_repetition_syndrome_counts(
    counts: Dict[str, int],
    measurement_error_probability: float,
) -> Dict[str, float]:
    """Estimate true repetition-code syndrome counts after readout bit flips.

    The mitigation model assumes each measured syndrome bit flips independently
    with the supplied probability. This is a small classical post-processing
    helper for the experimental repetition-code subset, not a full QEC decoder.
    """
    histogram = repetition_syndrome_histogram(counts)
    return _measurement_mitigated_syndrome_scores(
        histogram,
        measurement_error_probability,
    )


def _most_likely_syndrome(
    counts: Dict[str, int],
    measurement_error_probability: Optional[float] = None,
) -> List[int]:
    histogram = repetition_syndrome_histogram(counts)
    probability = _validate_optional_measurement_error_probability(
        measurement_error_probability
    )
    if probability is None:
        return _select_syndrome_from_scores(
            {key: float(value) for key, value in histogram.items()}
        )
    return _select_syndrome_from_scores(
        _measurement_mitigated_syndrome_scores(histogram, probability)
    )


def analyze_repetition_code_counts(
    counts: Dict[str, int],
    initial_bits: Optional[List[int]] = None,
    error_qubit: Optional[int] = None,
    expected_logical_bit: Optional[int] = None,
    measurement_error_probability: Optional[float] = None,
) -> Dict[str, Any]:
    """Decode sampled repetition-code syndromes and summarize correction quality."""
    bits = _validate_repetition_bits(initial_bits)
    error_qubit = _validate_error_qubit(error_qubit)
    expected_logical_bit = _validate_optional_binary_bit(
        expected_logical_bit,
        "expected_logical_bit",
    )
    measurement_error_probability = _validate_optional_measurement_error_probability(
        measurement_error_probability
    )

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

    if measurement_error_probability is None:
        syndrome_scores = {key: float(value) for key, value in histogram.items()}
        syndrome_source = "raw_histogram"
        mitigated_syndrome_scores = None
    else:
        mitigated_syndrome_scores = _measurement_mitigated_syndrome_scores(
            histogram,
            measurement_error_probability,
        )
        syndrome_scores = mitigated_syndrome_scores
        syndrome_source = "measurement_mitigated"

    most_likely_syndrome = _select_syndrome_from_scores(syndrome_scores)
    most_likely = next(
        outcome
        for outcome in decoded_outcomes
        if outcome["syndrome"] == most_likely_syndrome
    )
    logical_success_rate = None
    if expected_logical_bit is not None:
        logical_success_rate = successful_shots / total_shots

    result = {
        "syndrome_histogram": histogram,
        "total_shots": total_shots,
        "initial_data_bits": bits,
        "observed_data_bits": observed_bits,
        "expected_logical_bit": expected_logical_bit,
        "decoded_outcomes": decoded_outcomes,
        "most_likely_syndrome": most_likely_syndrome,
        "most_likely_syndrome_source": syndrome_source,
        "most_likely_correction_qubit": most_likely["correction_qubit"],
        "most_likely_corrected_data_bits": most_likely["corrected_data_bits"],
        "logical_success_rate": logical_success_rate,
        "measurement_error_probability": measurement_error_probability,
    }
    if mitigated_syndrome_scores is not None:
        result["mitigated_syndrome_scores"] = mitigated_syndrome_scores
    return result


def analyze_repetition_code_rounds(
    round_counts: List[Dict[str, int]],
    initial_bits: Optional[List[int]] = None,
    error_qubits: Optional[List[Optional[int]]] = None,
    expected_logical_bit: Optional[int] = None,
    measurement_error_probability: Optional[float] = None,
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
    measurement_error_probability = _validate_optional_measurement_error_probability(
        measurement_error_probability
    )

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
            measurement_error_probability=measurement_error_probability,
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
        "measurement_error_probability": measurement_error_probability,
        "experimental_supported_subset": "three_qubit_repetition_repeated_rounds",
    }


def run_repetition_code_single_round(
    initial_bits: Optional[List[int]] = None,
    error_qubit: Optional[int] = None,
    shots: int = 1,
    backend: str = "state_vector",
    measurement_error_probability: Optional[float] = None,
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
    backend = _validate_backend_name(backend)
    shots = _validate_positive_integer(shots, "shots")
    measurement_error_probability = _validate_optional_measurement_error_probability(
        measurement_error_probability
    )

    bits = _validate_repetition_bits(initial_bits)
    error_qubit = _validate_error_qubit(error_qubit)

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
    syndrome = _most_likely_syndrome(
        counts,
        measurement_error_probability=measurement_error_probability,
    )
    analysis = analyze_repetition_code_counts(
        counts,
        bits,
        error_qubit=error_qubit,
        measurement_error_probability=measurement_error_probability,
    )

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
        "measurement_error_probability": measurement_error_probability,
        "experimental_supported_subset": "three_qubit_repetition_single_round",
    }


def run_repetition_code_rounds(
    initial_bits: Optional[List[int]] = None,
    error_qubits: Optional[List[Optional[int]]] = None,
    rounds: int = 1,
    shots: int = 1,
    backend: str = "state_vector",
    measurement_error_probability: Optional[float] = None,
) -> Dict[str, Any]:
    """Run repeated experimental 3-qubit repetition-code syndrome rounds."""
    backend = _validate_backend_name(backend)
    rounds = _validate_repetition_rounds(rounds)
    shots = _validate_positive_integer(shots, "shots")
    measurement_error_probability = _validate_optional_measurement_error_probability(
        measurement_error_probability
    )

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
            measurement_error_probability=measurement_error_probability,
        )
        round_counts.append(result["counts"])
        current_bits = result["most_likely_corrected_data_bits"]

    analysis = analyze_repetition_code_rounds(
        round_counts,
        initial_bits=bits,
        error_qubits=error_schedule,
        measurement_error_probability=measurement_error_probability,
    )
    analysis["shots_per_round"] = shots
    analysis["backend"] = backend
    analysis["measurement_error_probability"] = measurement_error_probability
    return analysis
