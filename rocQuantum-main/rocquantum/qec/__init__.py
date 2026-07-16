"""Experimental QEC helpers for rocQuantum."""

from .capacity import generate_random_bit_flips, sample_code_capacity
from .codes import (
    Code,
    CodeMetadata,
    RepetitionCode,
    SteaneCode,
    ThreeQubitRepetitionCode,
    code,
    get_available_codes,
    get_code,
    register_code,
)
from .decoders import (
    AsyncDecoderResult,
    BatchDecoderResult,
    BeliefPropagationDecoder,
    Decoder,
    DecoderResult,
    RepetitionCodeDecoder,
    SingleErrorLUTDecoder,
    decoder,
    get_available_decoders,
    get_decoder,
    register_decoder,
)
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
    "get_code",
    "get_available_codes",
    "get_decoder",
    "get_available_decoders",
    "generate_random_bit_flips",
    "sample_code_capacity",
    "QEC_Experiment.run_single_round",
    "run_repetition_code_single_round",
    "run_repetition_code_rounds",
    "analyze_repetition_code_counts",
    "analyze_repetition_code_rounds",
    "mitigate_repetition_syndrome_counts",
)

_QEC_SUPPORTED_FEATURES = (
    "validated dense GF(2) CSS code metadata and extension-point registry",
    "arbitrary odd-distance repetition and seven-qubit Steane code metadata",
    "host single-error lookup-table and belief-propagation decoders",
    "scalar, batch, and future-compatible asynchronous decoder entry points",
    "pure-NumPy reproducible bit-flip and code-capacity sampling",
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
    "correlated/device-calibrated decoder noise models beyond independent priors",
    "GPU-resident performance-tuned syndrome extraction",
    "broad CUDA-QX QEC library parity",
    "CUDA-QX tensor-network decoder parity",
    "CUDA-QX real-time decoder and device-resident QEC parity",
)
_QEC_EXECUTION_SCOPE = {
    "quantum_runtime": "canonical_rocq_sample_local_backends",
    "feedback": "sequential_sampled_classical_post_processing",
    "mid_circuit_dynamic_feedback": "unsupported",
    # Retain the legacy sampled-workflow field while exposing the broader
    # decoder library separately for callers that inspect the new registry.
    "decoder_scope": "lookup_table_repetition_code",
    "decoder_library_scope": "host_single_error_lut_and_dense_sum_product_belief_propagation",
    "noise_model": "independent_bit_flip_code_capacity_and_syndrome_readout_flips",
    "distributed_qec_execution": "unsupported",
}
_QEC_HARDWARE_EVIDENCE = {
    "probe_performed": False,
    "native_rocm_device_required_for_performance_claim": True,
    "capability_query_is_runtime_proof": False,
}
_QEC_FEATURE_EVIDENCE = {
    "code_metadata_registry": {
        "status": "host_contract_tested",
        "evidence": "dense uint8 GF(2) validation plus built-in repetition/Steane tests",
    },
    "single_error_lut_decoder": {
        "status": "host_contract_tested",
        "evidence": "exhaustive zero/single-error syndrome tests",
    },
    "belief_propagation_decoder": {
        "status": "host_contract_tested",
        "evidence": "pure-NumPy repetition-code convergence tests",
    },
    "code_capacity_sampling": {
        "status": "host_contract_tested",
        "evidence": "seeded NumPy tests and exact GF(2) syndrome invariants",
    },
    "sampled_repetition_runtime": {
        "status": "local_runtime_contract_tested",
        "evidence": "canonical rocq.sample local-backend contract tests",
    },
    "native_rocm_execution": {
        "status": "not_hardware_tested",
        "evidence": "no AMD GPU is available in the current environment",
    },
    "tensor_network_and_realtime_qec": {
        "status": "unsupported",
        "evidence": "not implemented and no parity claim is made",
    },
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
        "supported_code_families": [
            "odd-distance bit-flip repetition code metadata",
            "seven-qubit Steane CSS code metadata",
            "legacy three-qubit sampled repetition workflow",
        ],
        "available_codes": get_available_codes(),
        "available_host_decoders": get_available_decoders(),
        "measurement_error_model": "independent syndrome-bit flips with p in [0, 0.5)",
        "supported_backends": list(_runtime_supported_backend_names()),
        "runtime": "canonical rocq.sample() over supported local backends",
        "execution_scope": dict(_QEC_EXECUTION_SCOPE),
        "hardware_evidence": dict(_QEC_HARDWARE_EVIDENCE),
        "feature_evidence": {
            name: dict(evidence) for name, evidence in _QEC_FEATURE_EVIDENCE.items()
        },
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
    "AsyncDecoderResult",
    "BatchDecoderResult",
    "BeliefPropagationDecoder",
    "Code",
    "CodeMetadata",
    "Decoder",
    "DecoderResult",
    "QEC_Experiment",
    "RepetitionCode",
    "RepetitionCodeDecoder",
    "SingleErrorLUTDecoder",
    "SteaneCode",
    "ThreeQubitRepetitionCode",
    "analyze_repetition_code_counts",
    "analyze_repetition_code_rounds",
    "capabilities",
    "code",
    "decoder",
    "generate_random_bit_flips",
    "get_available_codes",
    "get_available_decoders",
    "get_code",
    "get_decoder",
    "mitigate_repetition_syndrome_counts",
    "qec_capabilities",
    "register_code",
    "register_decoder",
    "run_repetition_code_rounds",
    "run_repetition_code_single_round",
    "sample_code_capacity",
]
