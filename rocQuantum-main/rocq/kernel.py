from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .backends import get_backend
from .qvec import qvec

try:
    import rocquantum_bind
except ImportError:
    rocquantum_bind = None


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
        resolved = [int(t) for t in targets]
        cls._active.ops.append(GateOp(name=name, targets=resolved, params=params or {}))


class QuantumKernel:
    def __init__(self, func):
        self._func = func
        self.name = func.__name__
        self.num_qubits = 0
        self._last_context: Optional[_KernelBuildContext] = None

    def build(self, *args, **kwargs) -> _KernelBuildContext:
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

    # Supported gate -> (MLIR op name, arity)
    _GATE_TO_MLIR = {
        "h": ("quantum.h", 1),
        "x": ("quantum.x", 1),
        "y": ("quantum.y", 1),
        "z": ("quantum.z", 1),
        "cnot": ("quantum.cnot", 2),
        "cx": ("quantum.cnot", 2),
    }
    _PARAM_GATE_TO_MLIR = {
        "rx": "quantum.rx",
        "ry": "quantum.ry",
        "rz": "quantum.rz",
    }

    def mlir(self, *args, **kwargs) -> str:
        """Emit minimal textual MLIR for supported core gates.

        Supported: H, X, Y, Z, CNOT, RX, RY, RZ.
        Unsupported gates raise ``NotImplementedError`` with the gate name.
        """
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

        for op in ctx.ops:
            gate = op.name.lower()
            if gate in self._GATE_TO_MLIR:
                mlir_name, arity = self._GATE_TO_MLIR[gate]
                refs = _resolve_target_refs(op.targets)
                if len(refs) != arity:
                    raise ValueError(
                        f"Gate '{op.name}' expects {arity} target(s), got {len(refs)}."
                    )
                targets = ", ".join(refs)
                operand_types = ", ".join("!quantum.qubit" for _ in range(arity))
                body_lines.append(
                    f'    "{mlir_name}"({targets}) : ({operand_types}) -> ()'
                )
            elif gate in self._PARAM_GATE_TO_MLIR:
                refs = _resolve_target_refs(op.targets)
                if len(refs) != 1:
                    raise ValueError(
                        f"Gate '{op.name}' expects exactly one target, got {len(refs)}."
                    )
                angle = op.params.get("theta")
                if angle is None:
                    angle = op.params.get("phi")
                if angle is None and op.params:
                    angle = next(iter(op.params.values()))
                if angle is None:
                    raise ValueError(
                        f"Gate '{op.name}' requires a numeric parameter."
                    )
                body_lines.append(
                    f'    "{self._PARAM_GATE_TO_MLIR[gate]}"({refs[0]}) '
                    f'{{angle = {float(angle):.17g} : f64}} : (!quantum.qubit) -> ()'
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
            raise RuntimeError("rocquantum_bind is required to emit QIR.")
        mlir_code = self.mlir(*args, **kwargs)
        compiler = rocquantum_bind.MLIRCompiler(self.num_qubits, "hip_statevec")
        return compiler.emit_qir(mlir_code)

    def execute(self, backend: str = "state_vector", noise_model=None, *args, **kwargs):
        ctx = self.build(*args, **kwargs)
        if ctx._next_qubit_index == 0:
            raise ValueError("Kernel did not allocate any qubits.")
        backend_impl = get_backend(backend, ctx._next_qubit_index)

        for op in ctx.ops:
            backend_impl.apply_gate(op.name, op.targets, op.params)
            if noise_model is not None:
                for channel in noise_model.get_channels():
                    if channel["op"] and channel["op"] != op.name.lower():
                        continue
                    targets = channel["qubits"] if channel["qubits"] else op.targets
                    backend_impl.apply_noise(channel["type"], targets, channel["prob"])

        return backend_impl.get_state()


def kernel(func):
    return QuantumKernel(func)


def execute(kernel_obj: QuantumKernel, backend: str = "state_vector", noise_model=None, *args, **kwargs):
    if not isinstance(kernel_obj, QuantumKernel):
        raise TypeError("execute() expects a QuantumKernel instance.")
    return kernel_obj.execute(backend=backend, noise_model=noise_model, *args, **kwargs)
