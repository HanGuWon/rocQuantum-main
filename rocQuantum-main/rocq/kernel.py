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

    def mlir(self, *args, **kwargs) -> str:
        ctx = self.build(*args, **kwargs)
        lines = [
            "// rocq kernel MLIR stub",
            f"// kernel: {self.name}",
            f"// qubits: {self.num_qubits}",
            f"// ops: {len(ctx.ops)}",
        ]
        return "\n".join(lines)

    def qir(self, *args, **kwargs) -> str:
        if rocquantum_bind is None:
            raise RuntimeError("rocquantum_bind is required to emit QIR.")
        mlir_code = self.mlir(*args, **kwargs)
        if mlir_code.startswith("// rocq kernel MLIR stub"):
            raise NotImplementedError("MLIR emission for Python kernels is not implemented yet.")

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


def execute(kernel_obj: QuantumKernel, backend: str = "state_vector", *args, **kwargs):
    if not isinstance(kernel_obj, QuantumKernel):
        raise TypeError("execute() expects a QuantumKernel instance.")
    return kernel_obj.execute(backend=backend, *args, **kwargs)
