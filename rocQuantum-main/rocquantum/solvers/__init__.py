"""Experimental higher-level solver helpers for rocQuantum."""

from .qaoa import (
    get_num_qaoa_parameters,
    make_maxcut_qaoa_kernel,
    maxcut_cost_operator,
    solve_maxcut_qaoa,
)
from .vqe_solver import Optimizer, SciPyOptimizer, VQE_Solver
from .vqe_solver import _supported_backend_names as _runtime_supported_backend_names

_SOLVER_ENTRY_POINTS = (
    "VQE_Solver.evaluate_energy",
    "VQE_Solver.estimate_gradient",
    "VQE_Solver.solve",
    "make_maxcut_qaoa_kernel",
    "get_num_qaoa_parameters",
    "maxcut_cost_operator",
    "solve_maxcut_qaoa",
)

_SOLVER_SUPPORTED_FEATURES = (
    "VQE one-shot energy evaluation through rocq.observe()",
    "VQE solve loop with a pluggable Optimizer interface",
    "parameter-shift and finite-difference gradient estimates",
    "MaxCut QAOA H/CNOT/RZ/RX ansatz construction",
    "CUDA-QX-style QAOA parameter-count helper for the supported gamma/beta ansatz",
    "weighted MaxCut cost operator construction",
    "MaxCut solve wrapper that maximizes cut value through a negated VQE objective",
    "QAOA edge-list and edge-weight mapping normalization",
    "VQE ansatz and optimizer result parameter-count validation",
    "QuantumOperator objective, ansatz-kernel, finite-real parameter, energy, backend, gradient-method, verbose-option, optimizer-result, optimizer-interface, and optimizer-option validation",
)

_SOLVER_UNSUPPORTED_FEATURES = (
    "chemistry-specific Hamiltonian builders",
    "production optimizer suite or hybrid workflow scheduler",
    "GPU-resident native adjoint differentiation",
    "distributed or multi-QPU solver execution",
    "broad CUDA-QX hybrid-algorithm library parity",
)


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
    "Optimizer",
    "SciPyOptimizer",
    "VQE_Solver",
    "capabilities",
    "get_num_qaoa_parameters",
    "make_maxcut_qaoa_kernel",
    "maxcut_cost_operator",
    "solver_capabilities",
    "solve_maxcut_qaoa",
]
