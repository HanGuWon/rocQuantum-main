"""Experimental QEC helpers for rocQuantum."""

from .framework import (
    QEC_Experiment,
    analyze_repetition_code_counts,
    analyze_repetition_code_rounds,
    mitigate_repetition_syndrome_counts,
    run_repetition_code_rounds,
    run_repetition_code_single_round,
    _supported_backend_names as _runtime_supported_backend_names,
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
    "sequential sampled repeated-round classical feed-forward over most-likely corrections",
    "lookup-table single-X repetition-code correction",
    "syndrome histogram and logical-success analysis",
    "independent syndrome-bit readout-error mitigation",
    "positive-integer shot/round/num_qubits, backend, verbose-option, code/decoder interface, non-empty non-mapping stabilizer-fragment sequence, logical-operator result, decoder-correction result, unique ancilla-index, callable-or-None initial-state, one-bit ancilla sample, syndrome, and bool-safe count/bit validation",
)

_QEC_UNSUPPORTED_FEATURES = (
    "fault-tolerant logical workflow execution",
    "in-circuit mid-circuit measurement with dynamic classical feedback",
    "general stabilizer or surface-code decoder stack",
    "noise-aware decoder models beyond independent syndrome readout flips",
    "GPU-resident performance-tuned syndrome extraction",
    "broad CUDA-QX QEC library parity",
)
_QEC_EXECUTION_SCOPE = {
    "quantum_runtime": "canonical_rocq_sample_local_backends",
    "feedback": "sequential_sampled_classical_post_processing",
    "mid_circuit_dynamic_feedback": "unsupported",
    "decoder_scope": "lookup_table_repetition_code",
    "noise_model": "independent_syndrome_readout_flips",
    "distributed_qec_execution": "unsupported",
}
_QEC_HARDWARE_EVIDENCE = {
    "probe_performed": False,
    "native_rocm_device_required_for_performance_claim": True,
    "capability_query_is_runtime_proof": False,
}


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
        "supported_backends": list(_runtime_supported_backend_names()),
        "runtime": "canonical rocq.sample() over supported local backends",
        "execution_scope": dict(_QEC_EXECUTION_SCOPE),
        "hardware_evidence": dict(_QEC_HARDWARE_EVIDENCE),
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
