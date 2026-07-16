"""Static inspection and translation helpers for recorded rocq kernels."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from .kernel import QuantumKernel


@dataclass(frozen=True)
class Resources:
    """A deterministic resource estimate for one specialized kernel build."""

    num_qubits: int
    num_gates: int
    depth: int
    gate_counts: Mapping[str, int]

    def __post_init__(self):
        object.__setattr__(
            self,
            "gate_counts",
            MappingProxyType(dict(sorted(self.gate_counts.items()))),
        )

    def count(self, gate: str) -> int:
        """Return the number of occurrences of ``gate``."""

        if not isinstance(gate, str):
            raise TypeError("gate must be a string.")
        return self.gate_counts.get(gate.strip().lower(), 0)


def _require_kernel(kernel_obj) -> QuantumKernel:
    if not isinstance(kernel_obj, QuantumKernel):
        raise TypeError("Expected a QuantumKernel instance.")
    return kernel_obj


def estimate_resources(kernel_obj: QuantumKernel, *args, **kwargs) -> Resources:
    """Estimate allocation, gate counts, and dependency depth without execution."""

    kernel_obj = _require_kernel(kernel_obj)
    context = kernel_obj.build(*args, **kwargs)
    qubit_depths = [0] * context._next_qubit_index
    counts = Counter()

    for op in context.ops:
        name = op.name.strip().lower()
        counts[name] += 1
        layer = 1 + max((qubit_depths[target] for target in op.targets), default=0)
        for target in op.targets:
            qubit_depths[target] = layer

    return Resources(
        num_qubits=context._next_qubit_index,
        num_gates=len(context.ops),
        depth=max(qubit_depths, default=0),
        gate_counts=counts,
    )


def _format_angle(op) -> str:
    value = op.params.get("theta")
    if value is None:
        value = op.params.get("phi")
    if value is None and op.params:
        value = next(iter(op.params.values()))
    if value is None:
        raise ValueError(f"Gate '{op.name}' requires a numeric parameter.")
    return format(float(value), ".17g")


def draw(kernel_obj: QuantumKernel, *args, **kwargs) -> str:
    """Return a compact text circuit for one specialized kernel build."""

    kernel_obj = _require_kernel(kernel_obj)
    context = kernel_obj.build(*args, **kwargs)
    lines = [f"kernel {kernel_obj.name} ({context._next_qubit_index} qubits)"]
    for index, op in enumerate(context.ops):
        targets = ", ".join(f"q[{target}]" for target in op.targets)
        name = op.name.strip().lower()
        if op.params:
            name = f"{name}({_format_angle(op)})"
        lines.append(f"{index:>3}: {name} {targets}".rstrip())
    return "\n".join(lines)


_QASM_FIXED_GATES = {
    "h": "h",
    "x": "x",
    "y": "y",
    "z": "z",
    "s": "s",
    "sdg": "sdg",
    "t": "t",
    "tdg": "tdg",
    "tdag": "tdg",
    "cnot": "cx",
    "cx": "cx",
    "cz": "cz",
    "swap": "swap",
    "ccx": "ccx",
    "toffoli": "ccx",
}
_QASM_PARAM_GATES = {
    "rx": "rx",
    "ry": "ry",
    "rz": "rz",
    "p": "u1",
    "phase": "u1",
    "crz": "crz",
    "cp": "cu1",
    "cphase": "cu1",
}


def _translate_openqasm2(kernel_obj: QuantumKernel, *args, **kwargs) -> str:
    context = kernel_obj.build(*args, **kwargs)
    if context._next_qubit_index == 0:
        raise ValueError("OpenQASM 2 translation requires at least one qubit.")
    instructions = []
    for op in context.ops:
        name = op.name.strip().lower()
        operands = ",".join(f"q[{target}]" for target in op.targets)
        if name in _QASM_FIXED_GATES:
            instruction = f"{_QASM_FIXED_GATES[name]} {operands};"
        elif name in _QASM_PARAM_GATES:
            instruction = (
                f"{_QASM_PARAM_GATES[name]}({_format_angle(op)}) {operands};"
            )
        else:
            raise NotImplementedError(
                f"OpenQASM 2 translation does not support gate '{op.name}'."
            )
        instructions.append(instruction)

    lines = [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        f"qreg q[{context._next_qubit_index}];",
    ]
    lines.extend(instructions)
    return "\n".join(lines)


def translate(kernel_obj: QuantumKernel, format: str, *args, **kwargs) -> str:
    """Translate a specialized kernel to ``mlir``, ``openqasm2``, or ``qir``.

    Unsupported formats and operations raise instead of returning conceptual or
    partially translated output.
    """

    kernel_obj = _require_kernel(kernel_obj)
    if not isinstance(format, str) or not format.strip():
        raise ValueError("format must be one of: mlir, openqasm2, qir.")
    normalized_format = format.strip().lower().replace("_", "")
    if normalized_format == "mlir":
        return kernel_obj.mlir(*args, **kwargs)
    if normalized_format in {"openqasm2", "qasm2"}:
        return _translate_openqasm2(kernel_obj, *args, **kwargs)
    if normalized_format == "qir":
        return kernel_obj.qir(*args, **kwargs)
    raise ValueError(
        f"Unsupported translation format '{format}'. "
        "Supported formats are: mlir, openqasm2, qir."
    )


__all__ = ["Resources", "estimate_resources", "draw", "translate"]
