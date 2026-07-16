"""Typed dynamic-kernel builder for the canonical rocq runtime.

The builder mirrors the useful host-side subset of ``cudaq.make_kernel`` while
remaining explicit about its scope: argument expressions are specialized into
the existing canonical gate IR before backend execution.  It is not a native
MLIR JIT and does not implement measurement-driven classical control flow.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from numbers import Integral, Number, Real
from typing import Any, Dict, Iterator, List, Optional, Tuple, get_args, get_origin

from .kernel import GateOp, QuantumKernel, _KernelBuildContext, _normalize_gate_params


_BINARY_OPERATORS = {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "truediv": operator.truediv,
}


@dataclass(frozen=True)
class KernelExpression:
    """A symbolic classical value used by a dynamic kernel specialization."""

    operation: str
    operands: Tuple[Any, ...]
    _owner_token: Optional[object] = None

    @staticmethod
    def argument(
        index: int, owner_token: Optional[object] = None
    ) -> "KernelExpression":
        return KernelExpression("argument", (int(index),), owner_token)

    @staticmethod
    def constant(value: Any) -> "KernelExpression":
        return KernelExpression("constant", (value,))

    def _binary(self, operation: str, other: Any) -> "KernelExpression":
        return KernelExpression(operation, (self, _as_expression(other)))

    def _reflected_binary(self, operation: str, other: Any) -> "KernelExpression":
        return KernelExpression(operation, (_as_expression(other), self))

    def __add__(self, other: Any) -> "KernelExpression":
        return self._binary("add", other)

    def __radd__(self, other: Any) -> "KernelExpression":
        return self._reflected_binary("add", other)

    def __sub__(self, other: Any) -> "KernelExpression":
        return self._binary("sub", other)

    def __rsub__(self, other: Any) -> "KernelExpression":
        return self._reflected_binary("sub", other)

    def __mul__(self, other: Any) -> "KernelExpression":
        return self._binary("mul", other)

    def __rmul__(self, other: Any) -> "KernelExpression":
        return self._reflected_binary("mul", other)

    def __truediv__(self, other: Any) -> "KernelExpression":
        return self._binary("truediv", other)

    def __rtruediv__(self, other: Any) -> "KernelExpression":
        return self._reflected_binary("truediv", other)

    def __neg__(self) -> "KernelExpression":
        return KernelExpression("neg", (self,))

    def __getitem__(self, index: Any) -> "KernelExpression":
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise TypeError("Kernel argument indices must be integers.")
        return KernelExpression("index", (self, int(index)))

    def resolve(self, arguments: Tuple[Any, ...]) -> Any:
        if self.operation == "argument":
            return arguments[self.operands[0]]
        if self.operation == "constant":
            return self.operands[0]
        if self.operation == "index":
            container = self.operands[0].resolve(arguments)
            index = self.operands[1]
            try:
                return container[index]
            except (IndexError, KeyError, TypeError) as exc:
                raise ValueError(
                    f"Kernel argument index {index} cannot be resolved for this specialization."
                ) from exc
        if self.operation == "neg":
            return -self.operands[0].resolve(arguments)
        if self.operation in _BINARY_OPERATORS:
            left = self.operands[0].resolve(arguments)
            right = self.operands[1].resolve(arguments)
            try:
                return _BINARY_OPERATORS[self.operation](left, right)
            except (ArithmeticError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Kernel expression '{self.operation}' failed during specialization."
                ) from exc
        raise RuntimeError(f"Unknown kernel expression operation '{self.operation}'.")


def _as_expression(value: Any) -> KernelExpression:
    return value if isinstance(value, KernelExpression) else KernelExpression.constant(value)


def _substitute_expression(
    expression: KernelExpression,
    arguments: Dict[int, KernelExpression],
) -> KernelExpression:
    """Substitute callee argument nodes while preserving symbolic arithmetic."""

    if expression.operation == "argument":
        argument_index = expression.operands[0]
        try:
            return arguments[argument_index]
        except KeyError as exc:
            raise RuntimeError(
                f"Classical kernel argument {argument_index} was not bound at the call site."
            ) from exc
    if expression.operation == "constant":
        return expression
    return KernelExpression(
        expression.operation,
        tuple(
            _substitute_expression(operand, arguments)
            if isinstance(operand, KernelExpression)
            else operand
            for operand in expression.operands
        ),
        expression._owner_token,
    )


def _expression_owner_tokens(expression: KernelExpression) -> set[object]:
    tokens: set[object] = set()
    if expression.operation == "argument" and expression._owner_token is not None:
        tokens.add(expression._owner_token)
    for operand in expression.operands:
        if isinstance(operand, KernelExpression):
            tokens.update(_expression_owner_tokens(operand))
    return tokens


@dataclass(frozen=True)
class QuakeValue:
    """A qubit handle owned by one :class:`KernelBuilder`."""

    index: int
    _owner_token: object
    _argument_index: Optional[int] = None


@dataclass(frozen=True)
class MeasurementHandle:
    """A terminal builder measurement declaration.

    The canonical runtime currently samples terminal qubits as a launch-time
    operation rather than recording a classical SSA value in ``GateOp``.  This
    handle therefore identifies a terminal measurement selection; it cannot be
    coerced to ``bool`` for measurement-driven control flow.
    """

    qubits: Tuple[int, ...]
    basis: str
    register_name: str
    _owner_token: object

    def __bool__(self) -> bool:
        raise TypeError(
            "MeasurementHandle cannot be used as a boolean because mid-circuit "
            "classical control is not implemented."
        )


class BuilderQVector:
    """A fixed-size sequence of builder-owned qubit handles."""

    def __init__(self, start: int, size: int, owner_token: object):
        self._start = int(start)
        self._size = int(size)
        self._owner_token = owner_token

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[QuakeValue]:
        for offset in range(self._size):
            yield QuakeValue(self._start + offset, self._owner_token)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return list(self)[index]
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise TypeError("Quantum register indices must be integers or slices.")
        normalized = int(index)
        if normalized < 0:
            normalized += self._size
        if normalized < 0 or normalized >= self._size:
            raise IndexError("Quantum register index is out of range.")
        return QuakeValue(self._start + normalized, self._owner_token)


@dataclass(frozen=True)
class _BuilderOperation:
    name: str
    targets: Tuple[QuakeValue, ...]
    parameters: Tuple[Tuple[str, KernelExpression], ...] = ()


def _validate_argument_type(argument_type: Any) -> None:
    origin = get_origin(argument_type)
    if argument_type is QuakeValue:
        return
    if argument_type in {float, int, bool, complex}:
        return
    if argument_type in {list, tuple} or origin in {list, tuple}:
        element_types = get_args(argument_type)
        declared_element_types = tuple(
            element_type for element_type in element_types if element_type is not Ellipsis
        )
        if declared_element_types and any(
            element_type not in {float, int, bool, complex}
            for element_type in declared_element_types
        ):
            raise TypeError(
                "Dynamic kernel sequence arguments support only float, int, bool, "
                "or complex elements."
            )
        if len(set(declared_element_types)) > 1:
            raise TypeError("Dynamic kernel tuple arguments must use one homogeneous element type.")
        return
    raise TypeError(
        "Dynamic kernel argument types must be QuakeValue, float, int, bool, complex, "
        "list[T], or tuple[T]."
    )


def _normalize_scalar_argument(value: Any, expected_type: type, index: int) -> Any:
    label = f"kernel argument {index}"
    if expected_type is bool:
        if not isinstance(value, bool):
            raise TypeError(f"{label} must be bool.")
        return value
    if expected_type is int:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{label} must be int.")
        return int(value)
    if expected_type is float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{label} must be a finite real number.")
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError(f"{label} must be finite.")
        return normalized
    if expected_type is complex:
        if isinstance(value, bool) or not isinstance(value, Number):
            raise TypeError(f"{label} must be a finite number.")
        normalized = complex(value)
        if not math.isfinite(normalized.real) or not math.isfinite(normalized.imag):
            raise ValueError(f"{label} must be finite.")
        return normalized
    raise RuntimeError(f"Unsupported normalized argument type {expected_type!r}.")


def _normalize_argument(value: Any, expected_type: Any, index: int) -> Any:
    origin = get_origin(expected_type)
    if expected_type in {float, int, bool, complex}:
        return _normalize_scalar_argument(value, expected_type, index)

    if isinstance(value, (str, bytes)):
        raise TypeError(f"kernel argument {index} must be a numeric sequence.")
    try:
        values = list(value)
    except TypeError as exc:
        raise TypeError(f"kernel argument {index} must be a numeric sequence.") from exc

    element_types = get_args(expected_type)
    if element_types:
        element_type = element_types[0]
        values = [
            _normalize_scalar_argument(item, element_type, index)
            for item in values
        ]
    else:
        normalized_values = []
        for item in values:
            if isinstance(item, bool):
                normalized_values.append(item)
            elif isinstance(item, Integral):
                normalized_values.append(int(item))
            elif isinstance(item, Real):
                normalized_values.append(
                    _normalize_scalar_argument(item, float, index)
                )
            elif isinstance(item, Number):
                normalized_values.append(
                    _normalize_scalar_argument(item, complex, index)
                )
            else:
                raise TypeError(
                    f"kernel argument {index} must contain only numeric values."
                )
        values = normalized_values
    if expected_type is tuple or origin is tuple:
        return tuple(values)
    return values


class KernelBuilder(QuantumKernel):
    """Programmatic, typed kernel specialized onto the canonical gate runtime."""

    _SINGLE_QUBIT_GATES = ("h", "x", "y", "z", "s", "sdg", "t", "tdg")
    _PARAMETRIC_SINGLE_QUBIT_GATES = ("rx", "ry", "rz", "p")

    def __init__(self, argument_types: Tuple[Any, ...] = (), name: str = "builder_kernel"):
        for argument_type in argument_types:
            _validate_argument_type(argument_type)
        super().__init__(lambda: None)
        self.name = name
        self._argument_types = tuple(argument_types)
        self._owner_token = object()
        self._arguments = tuple(
            QuakeValue(-1, self._owner_token, index)
            if argument_type is QuakeValue
            else KernelExpression.argument(index, self._owner_token)
            for index, argument_type in enumerate(self._argument_types)
        )
        self._builder_num_qubits = 0
        self._operations: List[_BuilderOperation] = []
        self._measurements: List[MeasurementHandle] = []

    @property
    def arguments(self) -> Tuple[Any, ...]:
        return self._arguments

    @property
    def argument_types(self) -> Tuple[Any, ...]:
        return self._argument_types

    @property
    def measurement_handles(self) -> Tuple[MeasurementHandle, ...]:
        return tuple(self._measurements)

    def qalloc(self, size: int = 1) -> BuilderQVector:
        if self._measurements:
            raise RuntimeError(
                "Qubits cannot be allocated after a terminal measurement declaration."
            )
        if isinstance(size, bool) or not isinstance(size, Integral) or int(size) <= 0:
            raise ValueError("qalloc size must be a positive integer.")
        register = BuilderQVector(self._builder_num_qubits, int(size), self._owner_token)
        self._builder_num_qubits += int(size)
        self.num_qubits = self._builder_num_qubits
        return register

    def _target(self, value: Any) -> QuakeValue:
        if isinstance(value, BuilderQVector):
            if len(value) != 1:
                raise ValueError(
                    "A multi-qubit register cannot be used as a single gate target."
                )
            value = value[0]
        if not isinstance(value, QuakeValue):
            raise TypeError("Builder gate targets must be qubit handles from qalloc().")
        if value._owner_token is not self._owner_token:
            raise ValueError("A qubit handle cannot be shared across kernel builders.")
        if value._argument_index is not None:
            return value
        if value.index < 0 or value.index >= self._builder_num_qubits:
            raise ValueError("Builder gate target is outside the allocated register.")
        return value

    def _record(self, name: str, targets, **parameters: Any) -> None:
        if self._measurements:
            raise RuntimeError(
                "Quantum gates cannot be appended after a terminal measurement declaration."
            )
        self._append_operation(name, targets, **parameters)

    def _append_operation(self, name: str, targets, **parameters: Any) -> None:
        resolved_targets = tuple(self._target(target) for target in targets)
        target_keys = [self._qubit_key(target) for target in resolved_targets]
        if len(set(target_keys)) != len(target_keys):
            raise ValueError(f"Gate '{name}' target qubits must be distinct.")
        resolved_parameters = tuple(
            (key, _as_expression(value)) for key, value in parameters.items()
        )
        if any(
            token is not self._owner_token
            for _, expression in resolved_parameters
            for token in _expression_owner_tokens(expression)
        ):
            raise ValueError(
                "A symbolic classical expression cannot be shared across kernel builders."
            )
        self._operations.append(
            _BuilderOperation(
                name=name,
                targets=resolved_targets,
                parameters=resolved_parameters,
            )
        )

    @staticmethod
    def _broadcast_targets(target: Any):
        if isinstance(target, BuilderQVector):
            return list(target)
        return [target]

    def _single(self, name: str, target: Any) -> None:
        for qubit in self._broadcast_targets(target):
            self._record(name, [qubit])

    def _parametric_single(self, name: str, angle: Any, target: Any) -> None:
        key = "phi" if name in {"rz", "p"} else "theta"
        for qubit in self._broadcast_targets(target):
            self._record(name, [qubit], **{key: angle})

    def h(self, target): self._single("h", target)
    def x(self, target): self._single("x", target)
    def y(self, target): self._single("y", target)
    def z(self, target): self._single("z", target)
    def s(self, target): self._single("s", target)
    def sdg(self, target): self._single("sdg", target)
    def t(self, target): self._single("t", target)
    def tdg(self, target): self._single("tdg", target)
    def rx(self, angle, target): self._parametric_single("rx", angle, target)
    def ry(self, angle, target): self._parametric_single("ry", angle, target)
    def rz(self, angle, target): self._parametric_single("rz", angle, target)
    def p(self, angle, target): self._parametric_single("p", angle, target)
    def cx(self, control, target): self._record("cnot", [control, target])
    def cnot(self, control, target): self.cx(control, target)
    def cz(self, control, target): self._record("cz", [control, target])
    def swap(self, first, second): self._record("swap", [first, second])
    def ccx(self, first, second, target): self._record("ccx", [first, second, target])
    def cswap(self, control, first, second): self._record("cswap", [control, first, second])
    def crx(self, angle, control, target): self._record("crx", [control, target], theta=angle)
    def cry(self, angle, control, target): self._record("cry", [control, target], theta=angle)
    def crz(self, angle, control, target): self._record("crz", [control, target], phi=angle)
    def cp(self, angle, control, target): self._record("cp", [control, target], phi=angle)

    def mcx(self, controls, target) -> None:
        if isinstance(controls, BuilderQVector):
            control_values = list(controls)
        else:
            try:
                control_values = list(controls)
            except TypeError:
                control_values = [controls]
        if not control_values:
            raise ValueError("mcx requires at least one control qubit.")
        self._record("mcx", control_values + [target])

    @staticmethod
    def _qubit_key(value: QuakeValue) -> Tuple[str, int]:
        if value._argument_index is not None:
            return ("argument", value._argument_index)
        return ("local", value.index)

    @staticmethod
    def _operation(
        name: str,
        targets: Tuple[QuakeValue, ...],
        **parameters: Any,
    ) -> _BuilderOperation:
        return _BuilderOperation(
            name=name,
            targets=targets,
            parameters=tuple(
                (key, _as_expression(value)) for key, value in parameters.items()
            ),
        )

    def _bind_call_arguments(
        self,
        target: "KernelBuilder",
        arguments: Tuple[Any, ...],
    ) -> Tuple[Dict[int, QuakeValue], Dict[int, KernelExpression]]:
        if not isinstance(target, KernelBuilder):
            raise TypeError("Kernel composition requires another KernelBuilder.")
        if target is self:
            raise ValueError("A KernelBuilder cannot recursively call itself.")
        if target._measurements:
            raise NotImplementedError(
                "Composing kernels that declare measurements requires measurement-aware "
                "classical IR, which is not implemented."
            )
        if len(arguments) != len(target.argument_types):
            raise TypeError(
                f"Called kernel expects {len(target.argument_types)} argument(s), "
                f"received {len(arguments)}."
            )

        quantum_arguments: Dict[int, QuakeValue] = {}
        classical_arguments: Dict[int, KernelExpression] = {}
        for index, (argument, expected_type) in enumerate(
            zip(arguments, target.argument_types)
        ):
            if expected_type is QuakeValue:
                quantum_arguments[index] = self._target(argument)
                continue
            if isinstance(argument, (QuakeValue, BuilderQVector)):
                raise TypeError(f"called kernel argument {index} must be classical.")
            if isinstance(argument, KernelExpression):
                if any(
                    token is not self._owner_token
                    for token in _expression_owner_tokens(argument)
                ):
                    raise ValueError(
                        "A symbolic classical expression cannot be shared across "
                        "kernel builders."
                    )
                classical_arguments[index] = argument
                continue
            classical_arguments[index] = KernelExpression.constant(
                _normalize_argument(argument, expected_type, index)
            )
        return quantum_arguments, classical_arguments

    def _inline_operations(
        self,
        target: "KernelBuilder",
        arguments: Tuple[Any, ...],
    ) -> Tuple[List[_BuilderOperation], int]:
        quantum_arguments, classical_arguments = self._bind_call_arguments(
            target, arguments
        )
        local_qubits = {
            index: QuakeValue(
                self._builder_num_qubits + index,
                self._owner_token,
            )
            for index in range(target._builder_num_qubits)
        }

        def map_target(value: QuakeValue) -> QuakeValue:
            if value._owner_token is not target._owner_token:
                raise RuntimeError("Called kernel contains a foreign qubit handle.")
            if value._argument_index is not None:
                try:
                    return quantum_arguments[value._argument_index]
                except KeyError as exc:
                    raise RuntimeError(
                        f"Quantum kernel argument {value._argument_index} was not bound."
                    ) from exc
            try:
                return local_qubits[value.index]
            except KeyError as exc:
                raise RuntimeError(
                    f"Called kernel references unallocated local qubit {value.index}."
                ) from exc

        operations: List[_BuilderOperation] = []
        for operation_spec in target._operations:
            operations.append(
                _BuilderOperation(
                    name=operation_spec.name,
                    targets=tuple(map_target(value) for value in operation_spec.targets),
                    parameters=tuple(
                        (
                            key,
                            _substitute_expression(expression, classical_arguments),
                        )
                        for key, expression in operation_spec.parameters
                    ),
                )
            )
        return operations, target._builder_num_qubits

    def _commit_inlined(
        self,
        operations: List[_BuilderOperation],
        allocated_qubits: int,
    ) -> None:
        if self._measurements:
            raise RuntimeError(
                "Kernel calls cannot be appended after a terminal measurement declaration."
            )
        for operation_spec in operations:
            target_keys = [
                self._qubit_key(target) for target in operation_spec.targets
            ]
            if len(set(target_keys)) != len(target_keys):
                raise ValueError(
                    f"Gate '{operation_spec.name}' target qubits must be distinct "
                    "after kernel argument binding."
                )
            _KernelBuildContext._validate_gate_arity(
                operation_spec.name,
                list(range(len(operation_spec.targets))),
            )
            if any(
                token is not self._owner_token
                for _, expression in operation_spec.parameters
                for token in _expression_owner_tokens(expression)
            ):
                raise ValueError(
                    "Inlined operations contain a foreign symbolic classical expression."
                )
        self._operations.extend(operations)
        self._builder_num_qubits += allocated_qubits
        self.num_qubits = self._builder_num_qubits

    def apply_call(self, target: "KernelBuilder", *arguments: Any) -> None:
        """Inline a device-style builder kernel at this call site."""

        operations, allocated_qubits = self._inline_operations(target, arguments)
        self._commit_inlined(operations, allocated_qubits)

    def call(self, target: "KernelBuilder", *arguments: Any) -> None:
        """Alias for :meth:`apply_call`, matching CUDA-Q dynamic-kernel examples."""

        self.apply_call(target, *arguments)

    @staticmethod
    def _adjoint_operations(
        operations: List[_BuilderOperation],
    ) -> List[_BuilderOperation]:
        self_adjoint = {"h", "x", "y", "z", "cnot", "cz", "swap", "ccx", "cswap", "mcx"}
        inverse_names = {"s": "sdg", "sdg": "s", "t": "tdg", "tdg": "t"}
        parametric = {"rx", "ry", "rz", "p", "crx", "cry", "crz", "cp"}
        adjoint: List[_BuilderOperation] = []
        for operation_spec in reversed(operations):
            name = operation_spec.name.lower()
            if name in self_adjoint:
                adjoint.append(operation_spec)
                continue
            if name in inverse_names:
                adjoint.append(
                    _BuilderOperation(
                        inverse_names[name],
                        operation_spec.targets,
                        operation_spec.parameters,
                    )
                )
                continue
            if name in parametric:
                adjoint.append(
                    _BuilderOperation(
                        name,
                        operation_spec.targets,
                        tuple(
                            (key, -expression)
                            for key, expression in operation_spec.parameters
                        ),
                    )
                )
                continue
            raise NotImplementedError(
                f"Adjoint synthesis does not support gate '{operation_spec.name}'."
            )
        return adjoint

    def adjoint(self, target: "KernelBuilder", *arguments: Any) -> None:
        """Inline the inverse of ``target`` using exact gate-level synthesis."""

        operations, allocated_qubits = self._inline_operations(target, arguments)
        self._commit_inlined(
            self._adjoint_operations(operations),
            allocated_qubits,
        )

    def _control_values(self, control: Any) -> Tuple[QuakeValue, ...]:
        if isinstance(control, BuilderQVector):
            values = list(control)
        elif isinstance(control, QuakeValue):
            values = [control]
        else:
            try:
                values = list(control)
            except TypeError as exc:
                raise TypeError(
                    "Kernel controls must be a qubit handle or a sequence of handles."
                ) from exc
        if not values:
            raise ValueError("Controlled kernel calls require at least one control qubit.")
        resolved = tuple(self._target(value) for value in values)
        keys = [self._qubit_key(value) for value in resolved]
        if len(set(keys)) != len(keys):
            raise ValueError("Controlled kernel call controls must be distinct.")
        return resolved

    @classmethod
    def _controlled_x(
        cls,
        controls: Tuple[QuakeValue, ...],
        target: QuakeValue,
    ) -> _BuilderOperation:
        if len(controls) == 1:
            return cls._operation("cnot", controls + (target,))
        if len(controls) == 2:
            return cls._operation("ccx", controls + (target,))
        return cls._operation("mcx", controls + (target,))

    @classmethod
    def _controlled_operation(
        cls,
        operation_spec: _BuilderOperation,
        controls: Tuple[QuakeValue, ...],
    ) -> List[_BuilderOperation]:
        target_keys = {cls._qubit_key(value) for value in operation_spec.targets}
        if any(cls._qubit_key(control) in target_keys for control in controls):
            raise ValueError(
                "A controlled kernel call cannot use an operation target as an "
                "additional control."
            )

        name = operation_spec.name.lower()
        targets = operation_spec.targets
        parameters = dict(operation_spec.parameters)
        if name == "x":
            return [cls._controlled_x(controls, targets[0])]
        if name == "y":
            return [
                cls._operation("sdg", (targets[0],)),
                cls._controlled_x(controls, targets[0]),
                cls._operation("s", (targets[0],)),
            ]
        if name == "z":
            return [
                cls._operation("h", (targets[0],)),
                cls._controlled_x(controls, targets[0]),
                cls._operation("h", (targets[0],)),
            ]
        if name == "h":
            return [
                cls._operation("ry", (targets[0],), theta=math.pi / 4.0),
                cls._controlled_x(controls, targets[0]),
                cls._operation("ry", (targets[0],), theta=-math.pi / 4.0),
            ]
        if name in {"s", "sdg", "t", "tdg", "p"}:
            if len(controls) != 1:
                raise NotImplementedError(
                    f"Multi-controlled phase synthesis is not available for gate '{name}'."
                )
            phases = {
                "s": KernelExpression.constant(math.pi / 2.0),
                "sdg": KernelExpression.constant(-math.pi / 2.0),
                "t": KernelExpression.constant(math.pi / 4.0),
                "tdg": KernelExpression.constant(-math.pi / 4.0),
            }
            angle = parameters.get("phi", phases.get(name))
            if angle is None:
                raise RuntimeError(f"Gate '{name}' is missing its phase parameter.")
            return [cls._operation("cp", controls + targets, phi=angle)]
        if name in {"rx", "ry", "rz"}:
            if len(controls) != 1:
                raise NotImplementedError(
                    f"Multi-controlled rotation synthesis is not available for gate '{name}'."
                )
            parameter_name = "phi" if name == "rz" else "theta"
            angle = parameters.get(parameter_name)
            if angle is None:
                raise RuntimeError(f"Gate '{name}' is missing its rotation parameter.")
            return [
                cls._operation(
                    f"c{name}",
                    controls + targets,
                    **{parameter_name: angle},
                )
            ]
        if name in {"cnot", "ccx", "mcx"}:
            combined_controls = controls + targets[:-1]
            if len({cls._qubit_key(value) for value in combined_controls}) != len(
                combined_controls
            ):
                raise ValueError("Controlled-X synthesis requires distinct control qubits.")
            return [cls._controlled_x(combined_controls, targets[-1])]
        if name == "cz":
            combined_controls = controls + (targets[0],)
            return [
                cls._operation("h", (targets[1],)),
                cls._controlled_x(combined_controls, targets[1]),
                cls._operation("h", (targets[1],)),
            ]
        if name == "swap" and len(controls) == 1:
            return [cls._operation("cswap", controls + targets)]
        raise NotImplementedError(
            f"Controlled-kernel synthesis does not support gate '{operation_spec.name}' "
            "with the requested control arity."
        )

    def control(
        self,
        target: "KernelBuilder",
        control: Any,
        *arguments: Any,
    ) -> None:
        """Inline a controlled form of ``target`` over supported canonical gates."""

        control_values = self._control_values(control)
        operations, allocated_qubits = self._inline_operations(target, arguments)
        controlled: List[_BuilderOperation] = []
        for operation_spec in operations:
            controlled.extend(
                self._controlled_operation(operation_spec, control_values)
            )
        self._commit_inlined(controlled, allocated_qubits)

    def _measurement_targets(self, target: Any) -> Tuple[QuakeValue, ...]:
        if isinstance(target, BuilderQVector):
            values = list(target)
        elif isinstance(target, QuakeValue):
            values = [target]
        else:
            try:
                values = list(target)
            except TypeError as exc:
                raise TypeError(
                    "Measurement targets must be a qubit handle or sequence of handles."
                ) from exc
        if not values:
            raise ValueError("Measurement requires at least one target qubit.")
        resolved = tuple(self._target(value) for value in values)
        if any(value._argument_index is not None for value in resolved):
            raise NotImplementedError(
                "Measurement of a quantum kernel argument requires measurement-aware "
                "call lowering, which is not implemented."
            )
        keys = [self._qubit_key(value) for value in resolved]
        if len(set(keys)) != len(keys):
            raise ValueError("Measurement target qubits must be distinct.")
        already_measured = {
            qubit
            for measurement in self._measurements
            for qubit in measurement.qubits
        }
        if any(value.index in already_measured for value in resolved):
            raise ValueError("A qubit cannot be declared for terminal measurement twice.")
        return resolved

    def _measure(
        self,
        basis: str,
        target: Any,
        register_name: str,
    ) -> MeasurementHandle:
        if not isinstance(register_name, str):
            raise TypeError("Measurement register_name must be a string.")
        if "\x00" in register_name:
            raise ValueError("Measurement register_name cannot contain a NUL byte.")
        if register_name and any(
            handle.register_name == register_name for handle in self._measurements
        ):
            raise ValueError(f"Measurement register name '{register_name}' is already used.")
        targets = self._measurement_targets(target)
        if basis == "x":
            for value in targets:
                self._append_operation("h", [value])
        elif basis == "y":
            for value in targets:
                self._append_operation("sdg", [value])
                self._append_operation("h", [value])
        handle = MeasurementHandle(
            tuple(value.index for value in targets),
            basis,
            register_name,
            self._owner_token,
        )
        self._measurements.append(handle)
        return handle

    def mz(self, target: Any, register_name: str = "") -> MeasurementHandle:
        """Declare terminal Z-basis measurement used by :meth:`sample`."""

        return self._measure("z", target, register_name)

    def mx(self, target: Any, register_name: str = "") -> MeasurementHandle:
        """Declare terminal X-basis measurement through an exact basis change."""

        return self._measure("x", target, register_name)

    def my(self, target: Any, register_name: str = "") -> MeasurementHandle:
        """Declare terminal Y-basis measurement through an exact basis change."""

        return self._measure("y", target, register_name)

    def exp_pauli(self, angle, targets, word: str) -> None:
        try:
            target_values = list(targets)
        except TypeError as exc:
            raise TypeError("exp_pauli targets must be a quantum register or sequence.") from exc
        if not isinstance(word, str) or not word:
            raise ValueError("exp_pauli word must be a non-empty I/X/Y/Z string.")
        normalized_word = word.upper()
        if len(normalized_word) != len(target_values):
            raise ValueError("exp_pauli word length must match the target count.")
        if any(pauli not in "IXYZ" for pauli in normalized_word):
            raise ValueError("exp_pauli word may contain only I, X, Y, and Z.")
        active = [
            (pauli, self._target(target))
            for pauli, target in zip(normalized_word, target_values)
            if pauli != "I"
        ]
        if not active:
            return
        if len({self._qubit_key(target) for _, target in active}) != len(active):
            raise ValueError("exp_pauli targets must be distinct.")
        for pauli, target in active:
            if pauli == "X":
                self.h(target)
            elif pauli == "Y":
                self.rx(math.pi / 2.0, target)
        for (_, control), (_, target) in zip(active, active[1:]):
            self.cx(control, target)
        self.rz(-2.0 * _as_expression(angle), active[-1][1])
        for (_, control), (_, target) in reversed(list(zip(active, active[1:]))):
            self.cx(control, target)
        for pauli, target in reversed(active):
            if pauli == "X":
                self.h(target)
            elif pauli == "Y":
                self.rx(-math.pi / 2.0, target)

    def build(self, *args, **kwargs) -> _KernelBuildContext:
        if kwargs:
            raise TypeError("Dynamic kernel specializations accept positional arguments only.")
        if any(argument_type is QuakeValue for argument_type in self._argument_types):
            raise RuntimeError(
                "A kernel with QuakeValue arguments is a device-style kernel and cannot "
                "be launched directly; compose it with apply_call(), control(), or adjoint()."
            )
        if len(args) != len(self._argument_types):
            raise TypeError(
                f"Kernel specialization expects {len(self._argument_types)} argument(s), "
                f"received {len(args)}."
            )
        normalized_arguments = tuple(
            _normalize_argument(value, expected_type, index)
            for index, (value, expected_type) in enumerate(zip(args, self._argument_types))
        )
        context = _KernelBuildContext()
        context._next_qubit_index = self._builder_num_qubits
        for operation_spec in self._operations:
            targets = [target.index for target in operation_spec.targets]
            context._validate_gate_arity(operation_spec.name, targets)
            context._validate_distinct_gate_targets(operation_spec.name, targets)
            parameters = _normalize_gate_params(
                {
                    key: expression.resolve(normalized_arguments)
                    for key, expression in operation_spec.parameters
                }
            )
            context.ops.append(GateOp(operation_spec.name, targets, parameters))
        self._last_context = context
        self.num_qubits = self._builder_num_qubits
        return context

    def _prepare_backend(self, backend, *args, **kwargs):
        if self._measurements:
            raise NotImplementedError(
                "State and expectation execution do not model measurement collapse; "
                "use sample() for a kernel with terminal measurements."
            )
        return super()._prepare_backend(backend, *args, **kwargs)

    def sample(self, *args, **kwargs):
        """Sample declared terminal measurements, or all qubits when none exist."""

        if self._measurements:
            if kwargs.get("qubits") is not None:
                raise ValueError(
                    "sample(qubits=...) cannot override builder terminal measurements."
                )
            kwargs = dict(kwargs)
            kwargs["qubits"] = [
                qubit
                for measurement in self._measurements
                for qubit in measurement.qubits
            ]
        return super().sample(*args, **kwargs)

    def mlir(self, *args, **kwargs) -> str:
        mlir = super().mlir(*args, **kwargs)
        if not self._measurements:
            return mlir

        explicit_labels = []
        for measurement in self._measurements:
            if not measurement.register_name:
                continue
            if len(measurement.qubits) == 1:
                explicit_labels.append(measurement.register_name)
            else:
                explicit_labels.extend(
                    f"{measurement.register_name}[{offset}]"
                    for offset in range(len(measurement.qubits))
                )
        if len(set(explicit_labels)) != len(explicit_labels):
            raise ValueError(
                "Measurement register names expand to duplicate QIR result labels."
            )

        used_labels = set(explicit_labels)
        next_generated_label = 0
        result_index = 0
        measurement_lines = []
        for measurement in self._measurements:
            for offset, qubit in enumerate(measurement.qubits):
                if measurement.register_name:
                    label = (
                        measurement.register_name
                        if len(measurement.qubits) == 1
                        else f"{measurement.register_name}[{offset}]"
                    )
                else:
                    while f"result[{next_generated_label}]" in used_labels:
                        next_generated_label += 1
                    label = f"result[{next_generated_label}]"
                    next_generated_label += 1
                    used_labels.add(label)
                escaped_label = []
                for byte in label.encode("utf-8"):
                    if 0x20 <= byte <= 0x7E and byte not in {0x22, 0x5C}:
                        escaped_label.append(chr(byte))
                    elif byte in {0x22, 0x5C}:
                        escaped_label.append("\\" + chr(byte))
                    else:
                        escaped_label.append(f"\\{byte:02X}")
                measurement_lines.append(
                    f'    %r{result_index} = "quantum.mz"(%q{qubit}) '
                    f'{{registerName = "{"".join(escaped_label)}"}} : '
                    "(!quantum.qubit) -> !quantum.result"
                )
                result_index += 1

        return_marker = "\n    return\n  }\n}"
        marker_index = mlir.rfind(return_marker)
        if marker_index < 0:
            raise RuntimeError("Generated kernel MLIR is missing its terminal return.")
        return (
            mlir[:marker_index]
            + "\n"
            + "\n".join(measurement_lines)
            + mlir[marker_index:]
        )

    def draw(self, *args, **kwargs) -> str:
        drawing = super().draw(*args, **kwargs)
        if not self._measurements:
            return drawing
        lines = [drawing]
        operation_index = len(self._operations)
        for measurement in self._measurements:
            targets = ", ".join(f"q[{qubit}]" for qubit in measurement.qubits)
            register = (
                f" -> {measurement.register_name}"
                if measurement.register_name
                else ""
            )
            lines.append(
                f"{operation_index:>3}: m{measurement.basis} {targets}{register}"
            )
            operation_index += 1
        return "\n".join(lines)

    def translate(self, format: str, *args, **kwargs) -> str:
        if self._measurements:
            raise NotImplementedError(
                "Translation of terminal measurement declarations requires "
                "measurement/result operations in the target IR."
            )
        return super().translate(format, *args, **kwargs)


def make_kernel(*argument_types: Any):
    """Create a typed dynamic kernel and CUDA-Q-style argument placeholders.

    With no argument types only the builder is returned.  Otherwise the return
    value is ``(builder, arg0, ...)``.
    """

    builder = KernelBuilder(tuple(argument_types))
    if not argument_types:
        return builder
    return (builder, *builder.arguments)


__all__ = [
    "BuilderQVector",
    "KernelBuilder",
    "KernelExpression",
    "MeasurementHandle",
    "QuakeValue",
    "make_kernel",
]
