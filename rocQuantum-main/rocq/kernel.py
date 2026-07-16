from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
import threading
from concurrent.futures import Executor, ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from .backends import (
    _DENSITY_MATRIX_MAX_DENSE_OBSERVABLE_TARGETS,
    _DENSITY_MATRIX_MAX_KRAUS_TARGETS,
    _DENSITY_MATRIX_MAX_QUBITS,
    _DENSITY_MATRIX_MAX_SAMPLE_QUBITS,
    _STATEVECTOR_MAX_QUBITS_BEFORE_SIZE_OVERFLOW,
    get_backend,
    _normalize_sample_qubits,
    _validate_positive_integer,
)
from .qvec import qvec
from .results import AsyncResult, ObserveResult, SampleResult
from .target import resolve_backend_name

try:
    import rocquantum_bind
except ImportError:
    rocquantum_bind = None

_COMPILER_BINDING_MISSING_MESSAGE = (
    "rocquantum_bind is required for compiler execution. Build rocQuantum with "
    "ROCQUANTUM_BUILD_BINDINGS=ON on a ROCm host, then retry. The Python "
    "compiler path is partial and covers only the canonical core-gate MLIR subset."
)
_COMPILER_SUPPORTED_MLIR_SUBSET = (
    "Supported canonical MLIR gates: qalloc, H/X/Y/Z/S/Sdg/T/Tdg, "
    "CNOT/CZ/SWAP/CCX/MCX/CSWAP, RX/RY/RZ/P, and CRX/CRY/CRZ/CP."
    " Offline QIR emission also supports terminal MZ result operations."
)
_COMPILER_SUPPORTED_GATE_GROUPS = {
    "allocation": ("qalloc",),
    "fixed_single_qubit": ("h", "x", "y", "z", "s", "sdg", "t", "tdg"),
    "fixed_multi_qubit": ("cnot", "cz", "swap", "ccx", "mcx", "cswap"),
    "parametric_single_qubit": ("rx", "ry", "rz", "p"),
    "parametric_controlled": ("crx", "cry", "crz", "cp"),
    "terminal_measurement": ("mz",),
}
_COMPILER_SUPPORTED_BACKENDS = ("hip_statevec",)
_COMPILER_SUPPORTED_QIR_PROFILES = ("qir-v2-static", "qir-v2-base")
_COMPILER_SUPPORTED_ARTIFACT_KINDS = ("llvm-ir", "llvm-bc", "object")
_COMPILER_UNSUPPORTED_FEATURES = (
    "mid-circuit measurement and measurement-driven classical control flow",
    "native typed function arguments/results and classical SSA",
    "noise channels",
    "arbitrary unitary/matrix operations",
    "QIR control-array lowering for variadic MCX",
    "dynamic QIR qubit/result management",
    "C++/Python AST source frontend parity with nvq++",
    "ORC JIT and a runnable QIS symbol-linking runtime",
    "external pass-plugin loading ABI",
    "release-wired adjoint-generation pass pipeline",
)
_COMPILER_QIR_MISSING_MESSAGE = (
    "QIR emission requires either a rocquantum_bind build with "
    "ROCQUANTUM_ENABLE_MLIR_COMPILER=ON or the GPU-independent "
    "rocq-translate executable on PATH (or in ROCQ_TRANSLATE_EXECUTABLE). "
    "ROCm and an AMD GPU are not required for rocq-translate."
)
_COMPILER_ARTIFACT_MISSING_MESSAGE = (
    "Compiler artifact emission requires either a rocquantum_bind build with "
    "ROCQUANTUM_ENABLE_MLIR_COMPILER=ON or the GPU-independent rocq-translate "
    "executable on PATH (or in ROCQ_TRANSLATE_EXECUTABLE)."
)
_COMPILER_DIALECT_DEFINITION = {
    "active_source_tree": "rocqCompiler/",
    "legacy_scaffold_source_tree": "rocquantum/include/rocquantum/Dialect and rocquantum/src/rocqCompiler",
    "release_tablegen_ops": True,
    "release_wired": True,
    "required_llvm_mlir": "22.1.x",
    "build_option": "ROCQUANTUM_ENABLE_MLIR_COMPILER",
    "legacy_scaffold_release_linked": False,
    "note": (
        "rocqCompiler/ owns the release-wired TableGen dialect, direct Quantum-to-QIR "
        "pass, and CPU-only rocq-opt/rocq-translate tools. The older rocquantum/Dialect "
        "tree remains an excluded legacy scaffold and is not CUDA-Q compiler parity."
    ),
}
_COMPILER_TRANSFORM_PIPELINE = {
    "quantum_to_qir_v2": {
        "source_tree": "rocqCompiler/passes/QuantumToQIRPass.cpp",
        "release_wired": True,
        "native_runtime_entry_point": True,
        "profile": "qir-v2-static",
        "llvm_verified": True,
        "gpu_required": False,
        "tools": ["rocq-opt", "rocq-translate"],
    },
    "adjoint_generation": {
        "source_tree": "rocquantum/src/rocqCompiler/Transforms/AdjointGeneration.cpp",
        "legacy_scaffold_only": True,
        "release_wired": False,
        "native_runtime_entry_point": False,
    },
}


def _find_rocq_translate() -> Optional[str]:
    """Find the optional GPU-independent compiler CLI without executing it."""

    configured = os.environ.get("ROCQ_TRANSLATE_EXECUTABLE")
    if configured:
        return shutil.which(configured)
    return shutil.which("rocq-translate")


def _binding_qir_emission_available() -> bool:
    if rocquantum_bind is None or not hasattr(rocquantum_bind, "MLIRCompiler"):
        return False
    return bool(
        getattr(rocquantum_bind, "MLIR_COMPILER_QIR_EMISSION_ENABLED", True)
    )


def _binding_artifact_emission_available() -> bool:
    if not _binding_qir_emission_available():
        return False
    compiler_type = getattr(rocquantum_bind, "MLIRCompiler", None)
    return bool(
        getattr(
            rocquantum_bind,
            "MLIR_COMPILER_ARTIFACT_EMISSION_ENABLED",
            compiler_type is not None and hasattr(compiler_type, "emit_artifact"),
        )
    )


def _run_rocq_translate(
    command: List[str],
    mlir_code: str,
    operation: str,
):
    try:
        result = subprocess.run(
            command,
            input=mlir_code,
            text=True,
            capture_output=True,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(
            f"{operation} could not run rocq-translate: {exc}. "
            f"{_COMPILER_SUPPORTED_MLIR_SUBSET}"
        ) from exc
    if result.returncode != 0:
        diagnostic = result.stderr.strip() or "no diagnostic was written"
        raise RuntimeError(
            f"{operation} failed through rocq-translate "
            f"(exit {result.returncode}): {diagnostic}. "
            f"{_COMPILER_SUPPORTED_MLIR_SUBSET}"
        )
    return result


def _emit_qir_with_cli(
    mlir_code: str,
    num_qubits: int,
    executable: str,
    profile: str = "qir-v2-static",
) -> str:
    command = [
        executable,
        f"--num-qubits={num_qubits}",
        f"--profile={profile}",
        "-",
    ]
    result = _run_rocq_translate(command, mlir_code, "QIR emission")
    qir = result.stdout
    if not qir.strip() or qir.lstrip().startswith("Error:"):
        raise RuntimeError(
            "rocq-translate returned an empty or error-sentinel QIR payload. "
            f"{_COMPILER_SUPPORTED_MLIR_SUBSET}"
        )
    return qir


def _emit_artifact_with_cli(
    mlir_code: str,
    num_qubits: int,
    executable: str,
    profile: str,
    kind: str,
    optimization_level: int,
    cache_dir: Optional[Union[str, os.PathLike]] = None,
) -> bytes:
    suffix = {"llvm-ir": ".ll", "llvm-bc": ".bc", "object": ".o"}[kind]
    with tempfile.TemporaryDirectory(prefix="rocq-artifact-") as directory:
        output_path = Path(directory, "kernel" + suffix)
        command = [
            executable,
            f"--num-qubits={num_qubits}",
            f"--profile={profile}",
            f"--emit={kind}",
            f"-O{optimization_level}",
            "-o",
            str(output_path),
        ]
        if cache_dir is not None:
            command.extend(["--cache-dir", os.fspath(cache_dir)])
        command.append("-")
        _run_rocq_translate(command, mlir_code, "Compiler artifact emission")
        try:
            payload = output_path.read_bytes()
        except OSError as exc:
            raise RuntimeError(
                "rocq-translate reported success but its artifact could not be read: "
                f"{exc}"
            ) from exc
    if not payload:
        raise RuntimeError("rocq-translate returned an empty compiler artifact.")
    return payload


_RUNTIME_EXECUTION_ENTRY_POINTS = (
    "execute",
    "get_state",
    "sample",
    "observe",
    "compile_and_execute",
    "execute_async",
    "get_state_async",
    "sample_async",
    "observe_async",
    "compile_and_execute_async",
)
_RUNTIME_SUPPORTED_BACKENDS = (
    "state_vector",
    "density_matrix",
    "stabilizer",
    "tableau",
    "clifford",
)
_RUNTIME_SUPPORTED_FEATURES = (
    "canonical kernel recording through @rocq.kernel",
    "state readback through get_state()/execute()",
    "selected-qubit sampling through sample()",
    "observable evaluation through observe()",
    "host-side Future wrappers for execute/get_state/sample/observe/compile_and_execute",
    "bool-safe state-vector-only enable_fusion execution option",
    "canonical backend-name validation",
    "positive-integer direct backend size validation",
    "direct backend gate-target validation",
    "finite direct backend gate-angle validation",
    "GateFusion rotation-angle validation before native queue dispatch",
    "direct backend state readback validation before returning native results",
    "Pauli observable target validation before backend dispatch",
    "lazy statevector fallback for legacy Pauli expectation bindings",
    "dense Hermitian observable validation before native/backend dispatch",
    "dense matrix operation validation before native device upload",
    "sparse Hamiltonian observable CSR validation before native/backend dispatch",
    "density-matrix Kraus channel payload validation before native device upload",
    "density-matrix noise-model channel revalidation before backend dispatch",
    "density-matrix noise model execution",
    "experimental Clifford stabilizer Pauli propagation backend",
    "partial compiler execution entry point with compiler_capabilities() boundary metadata",
    "ContextVar-based local target selection with explicit backend override",
    "dict/float-compatible result wrappers and Future-compatible AsyncResult.get()",
    "CUDA-Q-style shots_count sampling alias with a 1000-shot default",
    "single-logical-QPU qpu_id=0 validation on asynchronous entry points",
    "typed host-specialized make_kernel builder and parameter expressions",
    "static resource estimation, drawing, and fail-closed textual translation",
    "small-system CPU reference Schrodinger/Lindblad dynamics",
)
_RUNTIME_UNSUPPORTED_FEATURES = (
    "native HIP-stream futures",
    "multi-QPU or distributed scheduler futures",
    "statevector or estimator output for dynamic control-flow trajectories",
    "one unified compiler/runtime stack across rocq and legacy python/rocq",
    "production multi-GPU parity without self-hosted ROCm artifacts",
)
_RUNTIME_ASYNC_EXECUTION = {
    "future_type": "concurrent.futures.Future",
    "wrapper_type": "rocq.AsyncResult",
    "submission": "host_threadpool",
    "preserves_submission_context": True,
    "preserves_backend_validation": True,
    "accepted_qpu_ids": [0],
    "native_hip_stream_future": False,
    "multi_qpu_scheduler": False,
    "distributed_scheduler": False,
    "device_overlap_proof": False,
}
_BUILD_LOCK = threading.RLock()
_ASYNC_EXECUTOR_LOCK = threading.Lock()
_ASYNC_EXECUTOR: Optional[ThreadPoolExecutor] = None
_MISSING_SHOTS = object()
_DEFAULT_SHOTS_COUNT = 1000


def _get_default_async_executor() -> ThreadPoolExecutor:
    global _ASYNC_EXECUTOR
    with _ASYNC_EXECUTOR_LOCK:
        if _ASYNC_EXECUTOR is None:
            _ASYNC_EXECUTOR = ThreadPoolExecutor(
                max_workers=4,
                thread_name_prefix="rocq-async",
            )
        return _ASYNC_EXECUTOR


def _submit_async(
    callback: Callable[[], object], executor: Optional[Executor] = None
) -> AsyncResult:
    submitter = executor if executor is not None else _get_default_async_executor()
    context = copy_context()
    return AsyncResult(submitter.submit(context.run, callback))


def compiler_capabilities() -> Dict[str, object]:
    """Return the supported canonical compiler subset without invoking MLIR."""

    binding_available = rocquantum_bind is not None
    mlir_runtime_available = bool(
        getattr(rocquantum_bind, "MLIR_COMPILER_ENABLED", False)
    ) if binding_available else False
    binding_qir_available = _binding_qir_emission_available()
    binding_artifact_available = _binding_artifact_emission_available()
    translator = _find_rocq_translate()
    qir_emission_available = binding_qir_available or translator is not None
    artifact_emission_available = (
        binding_artifact_available or translator is not None
    )
    gpu_execution_available = bool(
        getattr(
            rocquantum_bind,
            "MLIR_COMPILER_GPU_EXECUTION_ENABLED",
            mlir_runtime_available,
        )
    ) if binding_available else False
    if binding_qir_available:
        qir_profile = getattr(
            rocquantum_bind,
            "MLIR_COMPILER_QIR_PROFILE",
            "qir-v2-static",
        )
        qir_emission_kind = "native_binding"
    elif translator is not None:
        qir_profile = "qir-v2-static"
        qir_emission_kind = "rocq_translate_cli"
    else:
        qir_profile = None
        qir_emission_kind = "unavailable"
    if binding_available:
        mlir_runtime_kind = str(
            getattr(
                rocquantum_bind,
                "MLIR_COMPILER_RUNTIME_KIND",
                "unknown_binding_runtime",
            )
        )
    else:
        mlir_runtime_kind = "missing_binding"
    return {
        "status": "partial",
        "binding_available": binding_available,
        "mlir_runtime_available": mlir_runtime_available,
        "mlir_runtime_kind": mlir_runtime_kind,
        "qir_emission_available": qir_emission_available,
        "qir_emission_kind": qir_emission_kind,
        "artifact_emission_available": artifact_emission_available,
        "artifact_emission_kind": (
            "native_binding"
            if binding_artifact_available
            else "rocq_translate_cli"
            if translator is not None
            else "unavailable"
        ),
        "artifact_kinds": list(_COMPILER_SUPPORTED_ARTIFACT_KINDS),
        "artifact_optimization_levels": [0, 1, 2, 3],
        "artifact_cache": {
            "available": translator is not None,
            "kind": "content_addressed_cli" if translator is not None else "unavailable",
            "compiler_fingerprinted": True,
            "corruption_policy": "fail_closed",
        },
        "rocq_translate_available": translator is not None,
        "gpu_execution_available": gpu_execution_available,
        "qir_profile": qir_profile,
        "qir_profiles": list(_COMPILER_SUPPORTED_QIR_PROFILES),
        "default_backend": "hip_statevec",
        "supported_backends": list(_COMPILER_SUPPORTED_BACKENDS),
        "supported_subset": _COMPILER_SUPPORTED_MLIR_SUBSET,
        "supported_gate_groups": {
            key: list(values)
            for key, values in _COMPILER_SUPPORTED_GATE_GROUPS.items()
        },
        "unsupported_features": list(_COMPILER_UNSUPPORTED_FEATURES),
        "dialect_definition": dict(_COMPILER_DIALECT_DEFINITION),
        "transform_pipeline": {
            key: dict(value)
            for key, value in _COMPILER_TRANSFORM_PIPELINE.items()
        },
        "mlir_runtime_note": (
            "QIR emission is available in the optional LLVM/MLIR 22.1 compiler build "
            "and can run without an AMD GPU via the offline MLIRCompiler constructor "
            "or the installed rocq-translate CLI fallback. "
            "HIP compile-and-execute remains device-dependent; default Python bindings "
            "may expose a fail-fast DisabledRuntimeMLIRCompiler. LLVM bitcode and "
            "host PIC objects are offline artifacts with unresolved QIS/runtime symbols, "
            "not runnable executables."
        ),
        "python_dynamic_builder": {
            "entry_point": "rocq.make_kernel",
            "typed_arguments": True,
            "argument_expressions": True,
            "device_kernel_composition": "static_inlining",
            "adjoint_synthesis": "canonical_gate_inverse",
            "controlled_synthesis": "supported_canonical_subset",
            "terminal_measurement": True,
            "measurement_mlir": True,
            "specialization_kind": "host_gate_ir",
            "native_mlir_jit": False,
            "measurement_control_flow": False,
        },
    }


def runtime_capabilities() -> Dict[str, object]:
    """Return the canonical Python runtime contract without running a kernel."""

    return {
        "status": "partial",
        "primary_python_surface": "rocq",
        "legacy_python_surface": "python/rocq compatibility API",
        "execution_entry_points": list(_RUNTIME_EXECUTION_ENTRY_POINTS),
        "supported_backends": list(_RUNTIME_SUPPORTED_BACKENDS),
        "supported_features": list(_RUNTIME_SUPPORTED_FEATURES),
        "unsupported_features": list(_RUNTIME_UNSUPPORTED_FEATURES),
        "async_execution": dict(_RUNTIME_ASYNC_EXECUTION),
        "limits": {
            "max_statevector_qubits_before_size_overflow": _STATEVECTOR_MAX_QUBITS_BEFORE_SIZE_OVERFLOW,
            "max_density_matrix_qubits_before_dense_size_overflow": _DENSITY_MATRIX_MAX_QUBITS,
            "max_density_matrix_kraus_channel_targets": _DENSITY_MATRIX_MAX_KRAUS_TARGETS,
            "max_density_matrix_dense_observable_targets": _DENSITY_MATRIX_MAX_DENSE_OBSERVABLE_TARGETS,
            "max_density_matrix_sampled_qubits": _DENSITY_MATRIX_MAX_SAMPLE_QUBITS,
        },
        "runtime_options": {
            "enable_fusion": (
                "Optional boolean accepted by state_vector execute/get_state/sample/"
                "observe and their host-side async wrappers."
            ),
            "shots_count": (
                "CUDA-Q-style keyword alias for sample/sample_async; defaults to 1000 "
                "when neither the legacy shots argument nor shots_count is supplied."
            ),
            "qpu_id": (
                "Async entry points accept only qpu_id=0 because the local runtime "
                "does not implement a multi-QPU scheduler."
            ),
        },
        "environment_switches": {
            "ROCQ_DISABLE_GATE_FUSION": "Disables state-vector GateFusion when truthy.",
            "ROCQ_ENABLE_MOCK_BACKENDS": "Enables local CPU mock backends when native ROCm bindings are missing.",
        },
        "legacy_note": (
            "The python/rocq package remains a compatibility surface with conceptual "
            "MLIR inspection and Python circuit replay; canonical runtime work should "
            "target rocq."
        ),
        "performance_note": (
            "Local tests can prove Python dispatch contracts, but ROCm performance proof "
            "requires self-hosted ROCm CI artifacts or real AMD GPU hardware."
        ),
    }


@dataclass(frozen=True)
class GateOp:
    name: str
    targets: List[int]
    params: Dict[str, float]


class _KernelBuildContext:
    _active: Optional["_KernelBuildContext"] = None

    def __init__(self) -> None:
        self.ops: List[GateOp] = []
        self.qvecs: List[qvec] = []
        self._next_qubit_index = 0

    def register_qvec(self, reg: qvec) -> None:
        reg.qubits = list(range(self._next_qubit_index, self._next_qubit_index + reg.size))
        self._next_qubit_index += reg.size
        self.qvecs.append(reg)

    @classmethod
    def add_gate(cls, name: str, targets: List[int], params: Optional[Dict[str, float]] = None) -> None:
        if cls._active is None:
            raise RuntimeError("No active kernel context. Gate called outside @rocq.kernel.")
        if not isinstance(targets, list):
            raise TypeError("targets must be a list of qubit indices.")
        resolved = [cls._active._validate_gate_target(t) for t in targets]
        cls._active._validate_gate_arity(name, resolved)
        cls._active._validate_distinct_gate_targets(name, resolved)
        cls._active.ops.append(GateOp(name=name, targets=resolved, params=_normalize_gate_params(params)))

    def _validate_gate_target(self, target) -> int:
        if isinstance(target, bool) or not isinstance(target, Integral):
            raise ValueError("Gate targets must be integer qubit indices.")
        resolved = int(target)
        if resolved < 0 or resolved >= self._next_qubit_index:
            raise ValueError(
                f"Gate target index {resolved} is out of bounds for {self._next_qubit_index} qubits."
            )
        return resolved

    @staticmethod
    def _validate_distinct_gate_targets(name: str, targets: List[int]) -> None:
        if len(set(targets)) != len(targets):
            raise ValueError(f"Gate '{name}' target qubits must be distinct.")

    @staticmethod
    def _validate_gate_arity(name: str, targets: List[int]) -> None:
        gate = name.lower()
        fixed_arity = {
            "h": 1,
            "x": 1,
            "y": 1,
            "z": 1,
            "s": 1,
            "sdg": 1,
            "t": 1,
            "tdg": 1,
            "rx": 1,
            "ry": 1,
            "rz": 1,
            "p": 1,
            "cnot": 2,
            "cz": 2,
            "swap": 2,
            "ccx": 3,
            "cswap": 3,
            "crx": 2,
            "cry": 2,
            "crz": 2,
            "cp": 2,
        }
        if gate in fixed_arity and len(targets) != fixed_arity[gate]:
            raise ValueError(
                f"Gate '{name}' expects {fixed_arity[gate]} target(s), got {len(targets)}."
            )
        if gate == "mcx" and len(targets) < 2:
            raise ValueError(
                f"Gate '{name}' expects at least 2 target(s): one control and one target."
            )


def _normalize_gate_params(params: Optional[Dict[str, float]]) -> Dict[str, float]:
    normalized: Dict[str, float] = {}
    for key, value in (params or {}).items():
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"Gate parameter '{key}' must be a finite real number.")
        parameter = float(value)
        if not math.isfinite(parameter):
            raise ValueError(f"Gate parameter '{key}' must be finite.")
        normalized[key] = parameter
    return normalized


def _validate_boolean(value, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean.")
    return value


def _validate_qpu_id(value) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("qpu_id must be the non-negative integer 0.")
    normalized = int(value)
    if normalized != 0:
        raise ValueError(
            "The canonical local runtime exposes one logical QPU; qpu_id must be 0."
        )
    return normalized


def _normalize_sample_invocation(shots, args, shots_count):
    if shots_count is None:
        resolved_shots = (
            _DEFAULT_SHOTS_COUNT
            if shots is _MISSING_SHOTS
            else _validate_positive_integer(shots, "shots")
        )
        return resolved_shots, tuple(args)

    resolved_shots = _validate_positive_integer(shots_count, "shots_count")
    kernel_args = tuple(args)
    if shots is not _MISSING_SHOTS:
        kernel_args = (shots, *kernel_args)
    return resolved_shots, kernel_args


def _validate_compiler_backend(backend) -> str:
    if not isinstance(backend, str):
        raise ValueError(
            f"compiler_backend must be one of: {list(_COMPILER_SUPPORTED_BACKENDS)}."
        )
    if backend not in _COMPILER_SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unsupported compiler_backend '{backend}'. "
            f"Supported compiler backends are: {list(_COMPILER_SUPPORTED_BACKENDS)}."
        )
    return backend


class QuantumKernel:
    def __init__(self, func):
        self._func = func
        self.name = func.__name__
        self.num_qubits = 0
        self._last_context: Optional[_KernelBuildContext] = None

    def __call__(self, *args, **kwargs):
        """Inline this kernel while another kernel is being recorded.

        CUDA-Q kernels are composable: a parameter-vector callable can invoke a
        separately decorated ansatz kernel.  The eager rocq recorder can support
        that host-specialized subset by replaying the wrapped Python function into
        the *current* build context.  Launching a kernel from ordinary host code is
        intentionally still explicit through execute/sample/observe/get_state.
        """

        if _KernelBuildContext._active is None:
            raise RuntimeError(
                "Direct QuantumKernel calls are only valid while recording another "
                "@rocq.kernel; use execute(), sample(), observe(), or get_state() "
                "for host execution."
            )
        return self._func(*args, **kwargs)

    def build(self, *args, **kwargs) -> _KernelBuildContext:
        with _BUILD_LOCK:
            ctx = _KernelBuildContext()
            _KernelBuildContext._active = ctx
            qvec._current_kernel_context = ctx
            try:
                self._func(*args, **kwargs)
            finally:
                qvec._current_kernel_context = None
                _KernelBuildContext._active = None
            self._last_context = ctx
            self.num_qubits = ctx._next_qubit_index
            return ctx

    def _prepare_backend(
        self,
        backend: Optional[str],
        *args,
        enable_fusion: Optional[bool] = None,
        **kwargs,
    ):
        ctx = self.build(*args, **kwargs)
        if ctx._next_qubit_index == 0:
            raise ValueError("Kernel did not allocate any qubits.")
        backend_impl = get_backend(
            resolve_backend_name(backend),
            ctx._next_qubit_index,
            enable_fusion=enable_fusion,
        )
        return ctx, backend_impl

    _GATE_TO_MLIR = {
        "h": ("quantum.h", 1),
        "x": ("quantum.x", 1),
        "y": ("quantum.y", 1),
        "z": ("quantum.z", 1),
        "s": ("quantum.s", 1),
        "sdg": ("quantum.sdg", 1),
        "t": ("quantum.t", 1),
        "tdg": ("quantum.tdg", 1),
        "tdag": ("quantum.tdg", 1),
        "cnot": ("quantum.cnot", 2),
        "cx": ("quantum.cnot", 2),
        "cz": ("quantum.cz", 2),
        "swap": ("quantum.swap", 2),
        "ccx": ("quantum.ccx", 3),
        "toffoli": ("quantum.ccx", 3),
        "cswap": ("quantum.cswap", 3),
        "fredkin": ("quantum.cswap", 3),
    }
    _VARIADIC_GATE_TO_MLIR = {
        "mcx": ("quantum.mcx", 2),
    }
    _PARAM_GATE_TO_MLIR = {
        "rx": ("quantum.rx", 1),
        "ry": ("quantum.ry", 1),
        "rz": ("quantum.rz", 1),
        "p": ("quantum.p", 1),
        "phase": ("quantum.p", 1),
        "crx": ("quantum.crx", 2),
        "cry": ("quantum.cry", 2),
        "crz": ("quantum.crz", 2),
        "cp": ("quantum.cp", 2),
        "cphase": ("quantum.cp", 2),
    }

    def mlir(self, *args, **kwargs) -> str:
        """Emit minimal textual MLIR for supported core gates."""
        ctx = self.build(*args, **kwargs)
        n = ctx._next_qubit_index
        body_lines = []

        qubit_values = [f"%q{i}" for i in range(n)]
        if n == 0:
            body_lines.append(
                '    "quantum.qalloc"() '
                f'{{size = {n} : i64}} : () -> ()'
            )
        elif n == 1:
            body_lines.append(
                '    %q0 = "quantum.qalloc"() '
                f'{{size = {n} : i64}} : () -> !quantum.qubit'
            )
        else:
            lhs = ", ".join(qubit_values)
            rhs_types = ", ".join("!quantum.qubit" for _ in range(n))
            body_lines.append(
                f'    {lhs} = "quantum.qalloc"() '
                f'{{size = {n} : i64}} : () -> ({rhs_types})'
            )

        def _resolve_target_refs(targets: List[int]) -> List[str]:
            refs: List[str] = []
            for t in targets:
                if t < 0 or t >= n:
                    raise ValueError(
                        f"Gate target index {t} is out of bounds for {n} qubits."
                    )
                refs.append(qubit_values[t])
            return refs

        def _require_distinct_targets(gate_name: str, targets: List[int]) -> None:
            if len(set(targets)) != len(targets):
                raise ValueError(f"Gate '{gate_name}' target qubits must be distinct.")

        for op in ctx.ops:
            gate = op.name.lower()
            if gate in self._GATE_TO_MLIR:
                mlir_name, arity = self._GATE_TO_MLIR[gate]
                refs = _resolve_target_refs(op.targets)
                if len(refs) != arity:
                    raise ValueError(
                        f"Gate '{op.name}' expects {arity} target(s), got {len(refs)}."
                    )
                _require_distinct_targets(op.name, op.targets)
                targets = ", ".join(refs)
                operand_types = ", ".join("!quantum.qubit" for _ in range(arity))
                body_lines.append(
                    f'    "{mlir_name}"({targets}) : ({operand_types}) -> ()'
                )
            elif gate in self._VARIADIC_GATE_TO_MLIR:
                mlir_name, min_arity = self._VARIADIC_GATE_TO_MLIR[gate]
                refs = _resolve_target_refs(op.targets)
                if len(refs) < min_arity:
                    raise ValueError(
                        f"Gate '{op.name}' expects at least {min_arity} target(s), got {len(refs)}."
                    )
                _require_distinct_targets(op.name, op.targets)
                targets = ", ".join(refs)
                operand_types = ", ".join("!quantum.qubit" for _ in refs)
                body_lines.append(
                    f'    "{mlir_name}"({targets}) : ({operand_types}) -> ()'
                )
            elif gate in self._PARAM_GATE_TO_MLIR:
                refs = _resolve_target_refs(op.targets)
                mlir_name, arity = self._PARAM_GATE_TO_MLIR[gate]
                if len(refs) != arity:
                    raise ValueError(
                        f"Gate '{op.name}' expects {arity} target(s), got {len(refs)}."
                    )
                _require_distinct_targets(op.name, op.targets)
                angle = op.params.get("theta")
                if angle is None:
                    angle = op.params.get("phi")
                if angle is None and op.params:
                    angle = next(iter(op.params.values()))
                if angle is None:
                    raise ValueError(
                        f"Gate '{op.name}' requires a numeric parameter."
                    )
                targets = ", ".join(refs)
                operand_types = ", ".join("!quantum.qubit" for _ in range(arity))
                body_lines.append(
                    f'    "{mlir_name}"({targets}) '
                    f'{{angle = {float(angle):.17g} : f64}} : ({operand_types}) -> ()'
                )
            else:
                raise NotImplementedError(
                    f"MLIR emission does not yet support gate '{op.name}'. "
                    f"Extend QuantumKernel._GATE_TO_MLIR to add it."
                )
        body = "\n".join(body_lines)
        return (
            f'module {{\n'
            f'  func.func @{self.name}() {{\n'
            f'{body}\n'
            f'    return\n'
            f'  }}\n'
            f'}}'
        )

    def qir(
        self,
        *args,
        qir_profile: str = "qir-v2-static",
        **kwargs,
    ) -> str:
        if qir_profile not in _COMPILER_SUPPORTED_QIR_PROFILES:
            raise ValueError(
                f"Unsupported QIR profile '{qir_profile}'. Supported profiles are: "
                f"{list(_COMPILER_SUPPORTED_QIR_PROFILES)}."
            )
        mlir_code = self.mlir(*args, **kwargs)
        if not _binding_qir_emission_available():
            translator = _find_rocq_translate()
            if translator is None:
                raise RuntimeError(_COMPILER_QIR_MISSING_MESSAGE)
            return _emit_qir_with_cli(
                mlir_code,
                self.num_qubits,
                translator,
                qir_profile,
            )

        # QIR emission is compiler-only and must not construct a HIP backend.
        compiler = rocquantum_bind.MLIRCompiler(self.num_qubits)
        try:
            qir = (
                compiler.emit_qir(mlir_code)
                if qir_profile == "qir-v2-static"
                else compiler.emit_qir(mlir_code, qir_profile)
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "QIR emission failed through rocquantum_bind.MLIRCompiler. "
                f"{_COMPILER_SUPPORTED_MLIR_SUBSET} Original error: {exc}"
            ) from exc
        if not isinstance(qir, str) or not qir.strip():
            raise RuntimeError(
                "rocquantum_bind.MLIRCompiler.emit_qir() returned an empty or "
                "non-text payload."
            )
        if qir.lstrip().startswith("Error:"):
            raise RuntimeError(
                "QIR emission failed in rocquantum_bind.MLIRCompiler.emit_qir(): "
                f"{qir} {_COMPILER_SUPPORTED_MLIR_SUBSET}"
            )
        return qir

    def emit_artifact(
        self,
        *args,
        kind: str = "llvm-bc",
        optimization_level: int = 0,
        qir_profile: str = "qir-v2-static",
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        **kwargs,
    ) -> bytes:
        """Emit verified LLVM IR, bitcode, or a host relocatable object.

        The returned payload is always ``bytes``. Host objects intentionally
        retain unresolved QIS/runtime symbols and need a separate QIR runtime
        linker; they are not directly executable programs.
        """

        if not isinstance(kind, str) or kind not in _COMPILER_SUPPORTED_ARTIFACT_KINDS:
            raise ValueError(
                f"Unsupported compiler artifact kind '{kind}'. Supported kinds are: "
                f"{list(_COMPILER_SUPPORTED_ARTIFACT_KINDS)}."
            )
        if (
            isinstance(optimization_level, bool)
            or not isinstance(optimization_level, Integral)
            or int(optimization_level) < 0
            or int(optimization_level) > 3
        ):
            raise ValueError("optimization_level must be an integer from 0 through 3.")
        normalized_optimization_level = int(optimization_level)
        if qir_profile not in _COMPILER_SUPPORTED_QIR_PROFILES:
            raise ValueError(
                f"Unsupported QIR profile '{qir_profile}'. Supported profiles are: "
                f"{list(_COMPILER_SUPPORTED_QIR_PROFILES)}."
            )
        if (
            qir_profile == "qir-v2-base"
            and kind != "object"
            and normalized_optimization_level != 0
        ):
            raise ValueError(
                "qir-v2-base LLVM IR and bitcode require optimization_level=0 "
                "because generic LLVM optimization does not preserve the required "
                "four-block profile contract."
            )
        if cache_dir is not None:
            try:
                normalized_cache_dir = os.fspath(cache_dir)
            except TypeError as exc:
                raise TypeError("cache_dir must be a filesystem path or None.") from exc
            if not normalized_cache_dir:
                raise ValueError("cache_dir must not be an empty path.")
        else:
            normalized_cache_dir = None

        mlir_code = self.mlir(*args, **kwargs)
        if _binding_artifact_emission_available() and normalized_cache_dir is None:
            compiler = rocquantum_bind.MLIRCompiler(self.num_qubits)
            try:
                payload = compiler.emit_artifact(
                    mlir_code,
                    kind,
                    normalized_optimization_level,
                    qir_profile,
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    "Compiler artifact emission failed through "
                    "rocquantum_bind.MLIRCompiler. "
                    f"{_COMPILER_SUPPORTED_MLIR_SUBSET} Original error: {exc}"
                ) from exc
            if not isinstance(payload, (bytes, bytearray, memoryview)):
                raise RuntimeError(
                    "rocquantum_bind.MLIRCompiler.emit_artifact() returned a "
                    "non-binary payload."
                )
            result = bytes(payload)
            if not result:
                raise RuntimeError(
                    "rocquantum_bind.MLIRCompiler.emit_artifact() returned an "
                    "empty payload."
                )
            return result

        translator = _find_rocq_translate()
        if translator is None:
            raise RuntimeError(_COMPILER_ARTIFACT_MISSING_MESSAGE)
        return _emit_artifact_with_cli(
            mlir_code,
            self.num_qubits,
            translator,
            qir_profile,
            kind,
            normalized_optimization_level,
            normalized_cache_dir,
        )

    def estimate_resources(self, *args, **kwargs):
        """Estimate resources for one specialization of this kernel."""

        from .tools import estimate_resources

        return estimate_resources(self, *args, **kwargs)

    def draw(self, *args, **kwargs) -> str:
        """Return a compact text representation of this kernel."""

        from .tools import draw

        return draw(self, *args, **kwargs)

    def translate(self, format: str, *args, **kwargs) -> str:
        """Translate this kernel to a supported textual format."""

        from .tools import translate

        return translate(self, format, *args, **kwargs)

    def compile_and_execute(
        self,
        *args,
        compiler_backend: str = "hip_statevec",
        strict: bool = True,
        **kwargs,
    ):
        """Compile the supported MLIR subset and execute it through the native compiler binding."""
        strict = _validate_boolean(strict, "strict")
        compiler_backend = _validate_compiler_backend(compiler_backend)
        if rocquantum_bind is None:
            raise RuntimeError(_COMPILER_BINDING_MISSING_MESSAGE)
        mlir_code = self.mlir(*args, **kwargs)
        compiler = rocquantum_bind.MLIRCompiler(self.num_qubits, compiler_backend)
        try:
            return compiler.compile_and_execute(mlir_code, {"strict": strict})
        except RuntimeError as exc:
            raise RuntimeError(
                "compile_and_execute() failed through rocquantum_bind.MLIRCompiler. "
                f"{_COMPILER_SUPPORTED_MLIR_SUBSET} Original error: {exc}"
            ) from exc

    def compile_and_execute_async(
        self,
        *args,
        compiler_backend: str = "hip_statevec",
        strict: bool = True,
        qpu_id: int = 0,
        executor: Optional[Executor] = None,
        **kwargs,
    ) -> AsyncResult:
        """Submit compile-and-execute work to a host-side Future."""

        _validate_qpu_id(qpu_id)
        return _submit_async(
            lambda: self.compile_and_execute(
                *args,
                compiler_backend=compiler_backend,
                strict=strict,
                **kwargs,
            ),
            executor=executor,
        )

    def execute(
        self,
        *args,
        backend: Optional[str] = None,
        noise_model=None,
        enable_fusion: Optional[bool] = None,
        **kwargs,
    ):
        ctx, backend_impl = self._prepare_backend(
            backend,
            *args,
            enable_fusion=enable_fusion,
            **kwargs,
        )
        backend_impl.run_ops(ctx.ops, noise_model=noise_model)
        return backend_impl.get_state()

    def get_state(
        self,
        *args,
        backend: Optional[str] = None,
        noise_model=None,
        enable_fusion: Optional[bool] = None,
        **kwargs,
    ):
        """Return the final state through the canonical execution path."""

        return self.execute(
            *args,
            backend=backend,
            noise_model=noise_model,
            enable_fusion=enable_fusion,
            **kwargs,
        )

    def execute_async(
        self,
        *args,
        backend: Optional[str] = None,
        noise_model=None,
        enable_fusion: Optional[bool] = None,
        qpu_id: int = 0,
        executor: Optional[Executor] = None,
        **kwargs,
    ) -> AsyncResult:
        """Submit execution work to a host-side Future."""

        _validate_qpu_id(qpu_id)
        return _submit_async(
            lambda: self.execute(
                *args,
                backend=backend,
                noise_model=noise_model,
                enable_fusion=enable_fusion,
                **kwargs,
            ),
            executor=executor,
        )

    def get_state_async(
        self,
        *args,
        backend: Optional[str] = None,
        noise_model=None,
        enable_fusion: Optional[bool] = None,
        qpu_id: int = 0,
        executor: Optional[Executor] = None,
        **kwargs,
    ) -> AsyncResult:
        """Submit state readback work to a host-side Future."""

        _validate_qpu_id(qpu_id)
        return _submit_async(
            lambda: self.get_state(
                *args,
                backend=backend,
                noise_model=noise_model,
                enable_fusion=enable_fusion,
                **kwargs,
            ),
            executor=executor,
        )

    def sample(
        self,
        shots=_MISSING_SHOTS,
        *args,
        shots_count=None,
        backend: Optional[str] = None,
        qubits=None,
        noise_model=None,
        enable_fusion: Optional[bool] = None,
        **kwargs,
    ):
        resolved_shots, kernel_args = _normalize_sample_invocation(
            shots, args, shots_count
        )
        ctx = self.build(*kernel_args, **kwargs)
        if ctx._next_qubit_index == 0:
            raise ValueError("Kernel did not allocate any qubits.")
        sample_qubits = _normalize_sample_qubits(qubits, ctx._next_qubit_index)
        backend_impl = get_backend(
            resolve_backend_name(backend),
            ctx._next_qubit_index,
            enable_fusion=enable_fusion,
        )
        backend_impl.run_ops(ctx.ops, noise_model=noise_model)
        return SampleResult(
            backend_impl.sample(resolved_shots, qubits=sample_qubits)
        )

    def sample_async(
        self,
        shots=_MISSING_SHOTS,
        *args,
        shots_count=None,
        backend: Optional[str] = None,
        qubits=None,
        noise_model=None,
        enable_fusion: Optional[bool] = None,
        qpu_id: int = 0,
        executor: Optional[Executor] = None,
        **kwargs,
    ) -> AsyncResult:
        """Submit sampling work to a host-side Future."""

        _validate_qpu_id(qpu_id)
        return _submit_async(
            lambda: self.sample(
                shots,
                *args,
                shots_count=shots_count,
                backend=backend,
                qubits=qubits,
                noise_model=noise_model,
                enable_fusion=enable_fusion,
                **kwargs,
            ),
            executor=executor,
        )

    def observe(
        self,
        operator,
        *args,
        backend: Optional[str] = None,
        noise_model=None,
        enable_fusion: Optional[bool] = None,
        **kwargs,
    ):
        if operator is None:
            raise TypeError("observe() requires a quantum operator.")
        ctx, backend_impl = self._prepare_backend(
            backend,
            *args,
            enable_fusion=enable_fusion,
            **kwargs,
        )
        backend_impl.run_ops(ctx.ops, noise_model=noise_model)
        return ObserveResult(backend_impl.expectation(operator))

    def observe_async(
        self,
        operator,
        *args,
        backend: Optional[str] = None,
        noise_model=None,
        enable_fusion: Optional[bool] = None,
        qpu_id: int = 0,
        executor: Optional[Executor] = None,
        **kwargs,
    ) -> AsyncResult:
        """Submit expectation work to a host-side Future."""

        _validate_qpu_id(qpu_id)
        return _submit_async(
            lambda: self.observe(
                operator,
                *args,
                backend=backend,
                noise_model=noise_model,
                enable_fusion=enable_fusion,
                **kwargs,
            ),
            executor=executor,
        )


def kernel(func):
    return QuantumKernel(func)


def execute(
    kernel_obj: QuantumKernel,
    *args,
    backend: Optional[str] = None,
    noise_model=None,
    enable_fusion: Optional[bool] = None,
    **kwargs,
):
    if not isinstance(kernel_obj, QuantumKernel):
        raise TypeError("execute() expects a QuantumKernel instance.")
    return kernel_obj.execute(
        *args,
        backend=backend,
        noise_model=noise_model,
        enable_fusion=enable_fusion,
        **kwargs,
    )


def get_state(
    kernel_obj: QuantumKernel,
    *args,
    backend: Optional[str] = None,
    noise_model=None,
    enable_fusion: Optional[bool] = None,
    **kwargs,
):
    if not isinstance(kernel_obj, QuantumKernel):
        raise TypeError("get_state() expects a QuantumKernel instance.")
    return kernel_obj.get_state(
        *args,
        backend=backend,
        noise_model=noise_model,
        enable_fusion=enable_fusion,
        **kwargs,
    )


def compile_and_execute(
    kernel_obj: QuantumKernel,
    *args,
    compiler_backend: str = "hip_statevec",
    strict: bool = True,
    **kwargs,
):
    if not isinstance(kernel_obj, QuantumKernel):
        raise TypeError("compile_and_execute() expects a QuantumKernel instance.")
    return kernel_obj.compile_and_execute(
        *args,
        compiler_backend=compiler_backend,
        strict=strict,
        **kwargs,
    )


def compile_and_execute_async(
    kernel_obj: QuantumKernel,
    *args,
    compiler_backend: str = "hip_statevec",
    strict: bool = True,
    qpu_id: int = 0,
    executor: Optional[Executor] = None,
    **kwargs,
) -> AsyncResult:
    if not isinstance(kernel_obj, QuantumKernel):
        raise TypeError("compile_and_execute_async() expects a QuantumKernel instance.")
    return kernel_obj.compile_and_execute_async(
        *args,
        compiler_backend=compiler_backend,
        strict=strict,
        qpu_id=qpu_id,
        executor=executor,
        **kwargs,
    )


def sample(
    kernel_obj: QuantumKernel,
    shots=_MISSING_SHOTS,
    *args,
    shots_count=None,
    backend: Optional[str] = None,
    qubits=None,
    noise_model=None,
    enable_fusion: Optional[bool] = None,
    **kwargs,
):
    if not isinstance(kernel_obj, QuantumKernel):
        raise TypeError("sample() expects a QuantumKernel instance.")
    return kernel_obj.sample(
        shots,
        *args,
        shots_count=shots_count,
        backend=backend,
        qubits=qubits,
        noise_model=noise_model,
        enable_fusion=enable_fusion,
        **kwargs,
    )


def execute_async(
    kernel_obj: QuantumKernel,
    *args,
    backend: Optional[str] = None,
    noise_model=None,
    enable_fusion: Optional[bool] = None,
    qpu_id: int = 0,
    executor: Optional[Executor] = None,
    **kwargs,
) -> AsyncResult:
    if not isinstance(kernel_obj, QuantumKernel):
        raise TypeError("execute_async() expects a QuantumKernel instance.")
    return kernel_obj.execute_async(
        *args,
        backend=backend,
        noise_model=noise_model,
        enable_fusion=enable_fusion,
        qpu_id=qpu_id,
        executor=executor,
        **kwargs,
    )


def get_state_async(
    kernel_obj: QuantumKernel,
    *args,
    backend: Optional[str] = None,
    noise_model=None,
    enable_fusion: Optional[bool] = None,
    qpu_id: int = 0,
    executor: Optional[Executor] = None,
    **kwargs,
) -> AsyncResult:
    if not isinstance(kernel_obj, QuantumKernel):
        raise TypeError("get_state_async() expects a QuantumKernel instance.")
    return kernel_obj.get_state_async(
        *args,
        backend=backend,
        noise_model=noise_model,
        enable_fusion=enable_fusion,
        qpu_id=qpu_id,
        executor=executor,
        **kwargs,
    )


def sample_async(
    kernel_obj: QuantumKernel,
    shots=_MISSING_SHOTS,
    *args,
    shots_count=None,
    backend: Optional[str] = None,
    qubits=None,
    noise_model=None,
    enable_fusion: Optional[bool] = None,
    qpu_id: int = 0,
    executor: Optional[Executor] = None,
    **kwargs,
) -> AsyncResult:
    if not isinstance(kernel_obj, QuantumKernel):
        raise TypeError("sample_async() expects a QuantumKernel instance.")
    return kernel_obj.sample_async(
        shots,
        *args,
        shots_count=shots_count,
        backend=backend,
        qubits=qubits,
        noise_model=noise_model,
        enable_fusion=enable_fusion,
        qpu_id=qpu_id,
        executor=executor,
        **kwargs,
    )


def observe(
    kernel_obj: QuantumKernel,
    operator,
    *args,
    backend: Optional[str] = None,
    noise_model=None,
    enable_fusion: Optional[bool] = None,
    **kwargs,
):
    if not isinstance(kernel_obj, QuantumKernel):
        raise TypeError("observe() expects a QuantumKernel instance.")
    return kernel_obj.observe(
        operator,
        *args,
        backend=backend,
        noise_model=noise_model,
        enable_fusion=enable_fusion,
        **kwargs,
    )


def observe_async(
    kernel_obj: QuantumKernel,
    operator,
    *args,
    backend: Optional[str] = None,
    noise_model=None,
    enable_fusion: Optional[bool] = None,
    qpu_id: int = 0,
    executor: Optional[Executor] = None,
    **kwargs,
) -> AsyncResult:
    if not isinstance(kernel_obj, QuantumKernel):
        raise TypeError("observe_async() expects a QuantumKernel instance.")
    return kernel_obj.observe_async(
        operator,
        *args,
        backend=backend,
        noise_model=noise_model,
        enable_fusion=enable_fusion,
        qpu_id=qpu_id,
        executor=executor,
        **kwargs,
    )
