"""Experimental higher-level solver helpers for rocQuantum."""

from .qaoa import (
    QAOAResult,
    get_operator_pool,
    get_num_qaoa_parameters,
    make_maxcut_qaoa_kernel,
    maxcut_cost_operator,
    qaoa,
    solve_maxcut_qaoa,
)
from .adapt import adapt_vqe
from .chemistry import MolecularHamiltonian, create_molecule, jordan_wigner
from .vqe_solver import (
    ObserveExecutionType,
    ObserveIteration,
    Optimizer,
    SciPyOptimizer,
    VQE_Solver,
    vqe,
)
from .vqe_solver import _supported_backend_names as _runtime_supported_backend_names

_SOLVER_ENTRY_POINTS = (
    "VQE_Solver.evaluate_energy",
    "VQE_Solver.estimate_gradient",
    "VQE_Solver.solve",
    "vqe",
    "adapt_vqe",
    "qaoa",
    "get_operator_pool",
    "jordan_wigner",
    "MolecularHamiltonian.from_integrals",
    "make_maxcut_qaoa_kernel",
    "get_num_qaoa_parameters",
    "maxcut_cost_operator",
    "solve_maxcut_qaoa",
)

_SOLVER_SUPPORTED_FEATURES = (
    "VQE one-shot energy evaluation through rocq.observe()",
    "VQE solve loop with a pluggable Optimizer interface",
    "CUDA-QX-style functional VQE result and immutable observe trace",
    "CUDA-QX parameter-vector callable ansatz adaptation and kernel composition",
    "documented SciPy method, jac, callback, and options keyword forwarding",
    "registered cobyla/lbfgs and SciPy-compatible optimizer normalization",
    "recorder-proven exact parameter-shift with precision-safe finite-difference fallback",
    "generic real Pauli-sum QAOA with custom mixer, full and counterdiabatic parameterizations",
    "tuple-unpackable QAOA result with canonical SampleResult final configuration",
    "single-process finite-difference ADAPT-VQE reference workflow",
    "CUDA-QX QAOA operator pool generation",
    "Jordan-Wigner transformation from precomputed one- and two-body integrals",
    "MaxCut QAOA H/CNOT/RZ/RX ansatz construction",
    "CUDA-QX-style QAOA parameter-count helper for the supported gamma/beta ansatz",
    "weighted MaxCut cost operator construction",
    "MaxCut solve wrapper that maximizes cut value through a negated VQE objective",
    "QAOA edge-list and edge-weight mapping normalization",
    "VQE ansatz and optimizer result parameter-count validation",
    "QuantumOperator objective, ansatz-kernel, finite-real parameter, energy, backend, gradient-method, verbose-option, optimizer-result, optimizer-interface, and optimizer-option validation",
)

_SOLVER_UNSUPPORTED_FEATURES = (
    "geometry/XYZ/PySCF chemistry Hamiltonian builders",
    "Bravyi-Kitaev and production UCC state-preparation families",
    "shot-based VQE and ADAPT-VQE expectation estimation",
    "production optimizer suite or hybrid workflow scheduler",
    "GPU-resident native adjoint differentiation",
    "distributed or multi-QPU solver execution",
    "broad CUDA-QX hybrid-algorithm library parity",
)
_SOLVER_EXECUTION_SCOPE = {
    "quantum_runtime": "canonical_rocq_observe_local_backends",
    "classical_optimizer": "host_python_optimizer_loop",
    "gradients": "host_parameter_shift_or_finite_difference",
    "gradient_safety": "exact_two_point_shift_only_for_proven_direct_single_rotations",
    "native_adjoint": "unsupported",
    "distributed_solver_execution": "unsupported",
    "workflow_scheduler": "unsupported",
}
_SOLVER_HARDWARE_EVIDENCE = {
    "probe_performed": False,
    "native_rocm_device_required_for_performance_claim": True,
    "capability_query_is_runtime_proof": False,
}

_SOLVER_FEATURE_STATUS = {
    "functional_vqe": "host_reference_verified",
    "generic_qaoa": "host_reference_verified",
    "adapt_vqe": "host_reference_verified_single_process",
    "jordan_wigner_precomputed_integrals": "host_reference_verified",
    "geometry_pyscf_driver": "unimplemented",
    "mqpu_mpi": "unimplemented",
    "native_rocm_performance": "rocm_accelerated_unverified",
    "cuda_vendor_specific_workflows": "unsupported_vendor_specific",
}


def solver_capabilities():
    """Return the advertised experimental solver-layer contract."""

    return {
        "status": "experimental_partial",
        "comparison_target": "CUDA-QX higher-level solver libraries",
        "entry_points": list(_SOLVER_ENTRY_POINTS),
        "supported_features": list(_SOLVER_SUPPORTED_FEATURES),
        "unsupported_features": list(_SOLVER_UNSUPPORTED_FEATURES),
        "supported_backends": list(_runtime_supported_backend_names()),
        "runtime": "canonical rocq.observe() over supported local backends",
        "execution_scope": dict(_SOLVER_EXECUTION_SCOPE),
        "hardware_evidence": dict(_SOLVER_HARDWARE_EVIDENCE),
        "feature_status": dict(_SOLVER_FEATURE_STATUS),
        "optional_dependencies": {
            "scipy": "required only when using the default SciPyOptimizer",
        },
        "docs": "rocquantum/solvers/README.md",
        "performance_note": (
            "This is a correctness-oriented experimental Python layer; ROCm "
            "performance proof requires self-hosted ROCm CI or real hardware."
        ),
    }


def capabilities():
    """Alias for callers that inspect the solvers package directly."""

    return solver_capabilities()


__all__ = [
    "MolecularHamiltonian",
    "ObserveExecutionType",
    "ObserveIteration",
    "Optimizer",
    "QAOAResult",
    "SciPyOptimizer",
    "VQE_Solver",
    "adapt_vqe",
    "capabilities",
    "create_molecule",
    "get_operator_pool",
    "get_num_qaoa_parameters",
    "jordan_wigner",
    "make_maxcut_qaoa_kernel",
    "maxcut_cost_operator",
    "qaoa",
    "solver_capabilities",
    "solve_maxcut_qaoa",
    "vqe",
]
