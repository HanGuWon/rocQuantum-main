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

    # Supported gate -> MLIR op name mapping
    _GATE_TO_MLIR = {
        "h": "quantum.h", "x": "quantum.x", "y": "quantum.y",
        "z": "quantum.z", "cnot": "quantum.cnot", "cx": "quantum.cnot",
    }
    _PARAM_GATE_TO_MLIR = {
        "rx": "quantum.rx", "ry": "quantum.ry", "rz": "quantum.rz",
    }

    def mlir(self, *args, **kwargs) -> str:
        """Emit minimal textual MLIR for supported core gates.

        Supported: H, X, Y, Z, CNOT, RX, RY, RZ.
        Unsupported gates raise ``NotImplementedError`` with the gate name.
        """
        ctx = self.build(*args, **kwargs)
        n = ctx._next_qubit_index
        body_lines = []
        body_lines.append(f'    %qreg = "quantum.qalloc"() {{size = {n} : i64}} : () -> !quantum.qubit')
        for op in ctx.ops:
            gate = op.name.lower()
            if gate in self._GATE_TO_MLIR:
                targets = ", ".join(f"%qreg" for _ in op.targets)
                body_lines.append(
                    f'    "{ self._GATE_TO_MLIR[gate]}"({targets}) : '
                    f'({"!quantum.qubit, " * (len(op.targets) - 1)}!quantum.qubit) -> ()'
                )
            elif gate in self._PARAM_GATE_TO_MLIR:
                angle = list(op.params.values())[0]
                body_lines.append(
                    f'    "{ self._PARAM_GATE_TO_MLIR[gate]}"(%qreg) '
                    f'{{angle = {angle:.6e} : f64}} : (!quantum.qubit) -> ()'
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
