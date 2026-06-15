from __future__ import annotations

import math
import threading
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Callable, Dict, List, Optional

from .backends import get_backend, _normalize_sample_qubits, _validate_positive_integer
from .qvec import qvec

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
)
_COMPILER_SUPPORTED_GATE_GROUPS = {
    "allocation": ("qalloc",),
    "fixed_single_qubit": ("h", "x", "y", "z", "s", "sdg", "t", "tdg"),
    "fixed_multi_qubit": ("cnot", "cz", "swap", "ccx", "mcx", "cswap"),
    "parametric_single_qubit": ("rx", "ry", "rz", "p"),
    "parametric_controlled": ("crx", "cry", "crz", "cp"),
}
_COMPILER_SUPPORTED_BACKENDS = ("hip_statevec",)
_COMPILER_UNSUPPORTED_FEATURES = (
    "mid-circuit measurement",
    "classical control flow",
    "kernel arguments in emitted MLIR",
    "noise channels",
    "arbitrary unitary/matrix operations",
    "release-linked default MLIR runtime",
)
_RUNTIME_EXECUTION_ENTRY_POINTS = (
    "execute",
    "get_state",
    "sample",
    "observe",
    "execute_async",
    "get_state_async",
    "sample_async",
    "observe_async",
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
    "host-side Future wrappers for execute/get_state/sample/observe",
    "bool-safe state-vector-only enable_fusion execution option",
    "canonical backend-name validation",
    "positive-integer direct backend size validation",
    "direct backend gate-target validation",
    "finite direct backend gate-angle validation",
    "GateFusion rotation-angle validation before native queue dispatch",
    "Pauli observable target validation before backend dispatch",
    "lazy statevector fallback for legacy Pauli expectation bindings",
    "dense matrix operation validation before native device upload",
    "sparse Hamiltonian observable CSR validation before native/backend dispatch",
    "density-matrix Kraus channel payload validation before native device upload",
    "density-matrix noise model execution",
    "experimental Clifford stabilizer Pauli propagation backend",
)
_RUNTIME_UNSUPPORTED_FEATURES = (
    "native HIP-stream futures",
    "multi-QPU or distributed scheduler futures",
    "statevector or estimator output for dynamic control-flow trajectories",
    "one unified compiler/runtime stack across rocq and legacy python/rocq",
    "production multi-GPU parity without self-hosted ROCm artifacts",
)
_BUILD_LOCK = threading.RLock()
_ASYNC_EXECUTOR_LOCK = threading.Lock()
_ASYNC_EXECUTOR: Optional[ThreadPoolExecutor] = None


def _get_default_async_executor() -> ThreadPoolExecutor:
    global _ASYNC_EXECUTOR
    with _ASYNC_EXECUTOR_LOCK:
        if _ASYNC_EXECUTOR is None:
            _ASYNC_EXECUTOR = ThreadPoolExecutor(
                max_workers=4,
                thread_name_prefix="rocq-async",
            )
        return _ASYNC_EXECUTOR


def _submit_async(callback: Callable[[], object], executor: Optional[Executor] = None) -> Future:
    submitter = executor if executor is not None else _get_default_async_executor()
    return submitter.submit(callback)


def compiler_capabilities() -> Dict[str, object]:
    """Return the supported canonical compiler subset without invoking MLIR."""

    return {
        "status": "partial",
        "binding_available": rocquantum_bind is not None,
        "default_backend": "hip_statevec",
        "supported_backends": list(_COMPILER_SUPPORTED_BACKENDS),
        "supported_subset": _COMPILER_SUPPORTED_MLIR_SUBSET,
        "supported_gate_groups": {
            key: list(values)
            for key, values in _COMPILER_SUPPORTED_GATE_GROUPS.items()
        },
        "unsupported_features": list(_COMPILER_UNSUPPORTED_FEATURES),
        "mlir_runtime_note": (
            "Compiler execution requires rocquantum_bind.MLIRCompiler with the "
            "experimental rocqCompiler MLIR stack linked; default builds may expose "
            "a fail-fast DisabledRuntimeMLIRCompiler instead."
        ),
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
        "runtime_options": {
            "enable_fusion": (
                "Optional boolean accepted by state_vector execute/get_state/sample/"
                "observe and their host-side async wrappers."
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

    def _prepare_backend(self, backend: str, *args, enable_fusion: Optional[bool] = None, **kwargs):
        ctx = self.build(*args, **kwargs)
        if ctx._next_qubit_index == 0:
            raise ValueError("Kernel did not allocate any qubits.")
        backend_impl = get_backend(
            backend,
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

    def qir(self, *args, **kwargs) -> str:
        if rocquantum_bind is None:
            raise RuntimeError(_COMPILER_BINDING_MISSING_MESSAGE)
        mlir_code = self.mlir(*args, **kwargs)
        compiler = rocquantum_bind.MLIRCompiler(self.num_qubits, "hip_statevec")
        try:
            qir = compiler.emit_qir(mlir_code)
        except RuntimeError as exc:
            raise RuntimeError(
                "QIR emission failed through rocquantum_bind.MLIRCompiler. "
                f"{_COMPILER_SUPPORTED_MLIR_SUBSET} Original error: {exc}"
            ) from exc
        if isinstance(qir, str) and qir.startswith("Error:"):
            raise RuntimeError(
                "QIR emission failed in rocquantum_bind.MLIRCompiler.emit_qir(): "
                f"{qir} {_COMPILER_SUPPORTED_MLIR_SUBSET}"
            )
        return qir

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
        executor: Optional[Executor] = None,
        **kwargs,
    ) -> Future:
        """Submit compile-and-execute work to a host-side Future."""

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
        backend: str = "state_vector",
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
        backend: str = "state_vector",
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
        backend: str = "state_vector",
        noise_model=None,
        enable_fusion: Optional[bool] = None,
        executor: Optional[Executor] = None,
        **kwargs,
    ) -> Future:
        """Submit execution work to a host-side Future."""

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
        backend: str = "state_vector",
        noise_model=None,
        enable_fusion: Optional[bool] = None,
        executor: Optional[Executor] = None,
        **kwargs,
    ) -> Future:
        """Submit state readback work to a host-side Future."""

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
        shots: int,
        *args,
        backend: str = "state_vector",
        qubits=None,
        noise_model=None,
        enable_fusion: Optional[bool] = None,
        **kwargs,
    ):
        shots = _validate_positive_integer(shots, "shots")
        ctx = self.build(*args, **kwargs)
        if ctx._next_qubit_index == 0:
            raise ValueError("Kernel did not allocate any qubits.")
        sample_qubits = _normalize_sample_qubits(qubits, ctx._next_qubit_index)
        backend_impl = get_backend(
            backend,
            ctx._next_qubit_index,
            enable_fusion=enable_fusion,
        )
        backend_impl.run_ops(ctx.ops, noise_model=noise_model)
        return backend_impl.sample(shots, qubits=sample_qubits)

    def sample_async(
        self,
        shots: int,
        *args,
        backend: str = "state_vector",
        qubits=None,
        noise_model=None,
        enable_fusion: Optional[bool] = None,
        executor: Optional[Executor] = None,
        **kwargs,
    ) -> Future:
        """Submit sampling work to a host-side Future."""

        return _submit_async(
            lambda: self.sample(
                shots,
                *args,
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
        backend: str = "state_vector",
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
        return backend_impl.expectation(operator)

    def observe_async(
        self,
        operator,
        *args,
        backend: str = "state_vector",
        noise_model=None,
        enable_fusion: Optional[bool] = None,
        executor: Optional[Executor] = None,
        **kwargs,
    ) -> Future:
        """Submit expectation work to a host-side Future."""

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
    backend: str = "state_vector",
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
    backend: str = "state_vector",
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
    executor: Optional[Executor] = None,
    **kwargs,
) -> Future:
    if not isinstance(kernel_obj, QuantumKernel):
        raise TypeError("compile_and_execute_async() expects a QuantumKernel instance.")
    return kernel_obj.compile_and_execute_async(
        *args,
        compiler_backend=compiler_backend,
        strict=strict,
        executor=executor,
        **kwargs,
    )


def sample(
    kernel_obj: QuantumKernel,
    shots: int,
    *args,
    backend: str = "state_vector",
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
        backend=backend,
        qubits=qubits,
        noise_model=noise_model,
        enable_fusion=enable_fusion,
        **kwargs,
    )


def execute_async(
    kernel_obj: QuantumKernel,
    *args,
    backend: str = "state_vector",
    noise_model=None,
    enable_fusion: Optional[bool] = None,
    executor: Optional[Executor] = None,
    **kwargs,
) -> Future:
    if not isinstance(kernel_obj, QuantumKernel):
        raise TypeError("execute_async() expects a QuantumKernel instance.")
    return kernel_obj.execute_async(
        *args,
        backend=backend,
        noise_model=noise_model,
        enable_fusion=enable_fusion,
        executor=executor,
        **kwargs,
    )


def get_state_async(
    kernel_obj: QuantumKernel,
    *args,
    backend: str = "state_vector",
    noise_model=None,
    enable_fusion: Optional[bool] = None,
    executor: Optional[Executor] = None,
    **kwargs,
) -> Future:
    if not isinstance(kernel_obj, QuantumKernel):
        raise TypeError("get_state_async() expects a QuantumKernel instance.")
    return kernel_obj.get_state_async(
        *args,
        backend=backend,
        noise_model=noise_model,
        enable_fusion=enable_fusion,
        executor=executor,
        **kwargs,
    )


def sample_async(
    kernel_obj: QuantumKernel,
    shots: int,
    *args,
    backend: str = "state_vector",
    qubits=None,
    noise_model=None,
    enable_fusion: Optional[bool] = None,
    executor: Optional[Executor] = None,
    **kwargs,
) -> Future:
    if not isinstance(kernel_obj, QuantumKernel):
        raise TypeError("sample_async() expects a QuantumKernel instance.")
    return kernel_obj.sample_async(
        shots,
        *args,
        backend=backend,
        qubits=qubits,
        noise_model=noise_model,
        enable_fusion=enable_fusion,
        executor=executor,
        **kwargs,
    )


def observe(
    kernel_obj: QuantumKernel,
    operator,
    *args,
    backend: str = "state_vector",
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
    backend: str = "state_vector",
    noise_model=None,
    enable_fusion: Optional[bool] = None,
    executor: Optional[Executor] = None,
    **kwargs,
) -> Future:
    if not isinstance(kernel_obj, QuantumKernel):
        raise TypeError("observe_async() expects a QuantumKernel instance.")
    return kernel_obj.observe_async(
        operator,
        *args,
        backend=backend,
        noise_model=noise_model,
        enable_fusion=enable_fusion,
        executor=executor,
        **kwargs,
    )
