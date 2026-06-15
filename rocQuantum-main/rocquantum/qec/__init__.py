"""Experimental QEC helpers for rocQuantum."""

from .framework import (
    QEC_Experiment,
    analyze_repetition_code_counts,
    analyze_repetition_code_rounds,
    mitigate_repetition_syndrome_counts,
    run_repetition_code_rounds,
    run_repetition_code_single_round,
)

_QEC_ENTRY_POINTS = (
    "QEC_Experiment.run_single_round",
    "run_repetition_code_single_round",
    "run_repetition_code_rounds",
    "analyze_repetition_code_counts",
    "analyze_repetition_code_rounds",
    "mitigate_repetition_syndrome_counts",
)

_QEC_SUPPORTED_FEATURES = (
    "generic sampled stabilizer-fragment orchestration",
    "three-qubit bit-flip repetition-code single-round sampling",
    "sequential repeated-round repetition-code aggregation",
    "lookup-table single-X repetition-code correction",
    "syndrome histogram and logical-success analysis",
    "independent syndrome-bit readout-error mitigation",
    "positive-integer shot/round/num_qubits, ancilla-index, syndrome, and bool-safe count/bit validation",
)

_QEC_UNSUPPORTED_FEATURES = (
    "fault-tolerant logical workflow execution",
    "mid-circuit measurement and classical feedback",
    "general stabilizer or surface-code decoder stack",
    "noise-aware decoder models beyond independent syndrome readout flips",
    "GPU-resident performance-tuned syndrome extraction",
    "broad CUDA-QX QEC library parity",
)


def qec_capabilities():
    """Return the advertised experimental QEC-layer contract."""

    return {
        "status": "experimental_partial",
        "comparison_target": "CUDA-QX QEC libraries",
        "entry_points": list(_QEC_ENTRY_POINTS),
        "supported_features": list(_QEC_SUPPORTED_FEATURES),
        "unsupported_features": list(_QEC_UNSUPPORTED_FEATURES),
        "supported_code_family": "three-qubit bit-flip repetition code",
        "measurement_error_model": "independent syndrome-bit flips with p in [0, 0.5)",
        "runtime": "canonical rocq.sample() over supported local backends",
        "docs": "rocquantum/qec/README.md",
        "performance_note": (
            "This is a correctness-oriented experimental Python layer; ROCm "
            "performance proof requires self-hosted ROCm CI or real hardware."
        ),
    }


def capabilities():
    """Alias for callers that inspect the qec package directly."""

    return qec_capabilities()


__all__ = [
    "QEC_Experiment",
    "analyze_repetition_code_counts",
    "analyze_repetition_code_rounds",
    "capabilities",
    "mitigate_repetition_syndrome_counts",
    "qec_capabilities",
    "run_repetition_code_rounds",
    "run_repetition_code_single_round",
]
