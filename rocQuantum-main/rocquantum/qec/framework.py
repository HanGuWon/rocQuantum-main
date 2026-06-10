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
from typing import List, Dict, Callable, Any, Optional

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


def _most_likely_syndrome(counts: Dict[str, int]) -> List[int]:
    if not counts:
        raise ValueError("No syndrome samples were produced.")
    bitstring = max(counts.items(), key=lambda item: item[1])[0]
    if len(bitstring) < 2:
        bitstring = bitstring.zfill(2)
    # rocq packs measured qubits in request order, then formats the integer as
    # a big-endian bitstring. For qubits [3, 4], q3 is the rightmost bit.
    return [int(bitstring[-1]), int(bitstring[-2])]


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

    bits = list(initial_bits or [0, 0, 0])
    if len(bits) != 3 or any(bit not in (0, 1) for bit in bits):
        raise ValueError("initial_bits must be a length-3 list containing only 0 or 1.")
    if error_qubit is not None and error_qubit not in (0, 1, 2):
        raise ValueError("error_qubit must be one of 0, 1, 2, or None.")

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

    from rocquantum.qec.decoders.repetition_decoder import RepetitionCodeDecoder

    correction = RepetitionCodeDecoder().decode(syndrome)
    return {
        "syndrome": syndrome,
        "counts": counts,
        "correction": correction,
        "correction_applied": correction.to_string(),
        "experimental_supported_subset": "three_qubit_repetition_single_round",
    }
