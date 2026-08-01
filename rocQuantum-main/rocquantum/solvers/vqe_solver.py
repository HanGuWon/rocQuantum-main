# Copyright (c) 2025-2026, rocQuantum Developers.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
High-level Variational Quantum Eigensolver (VQE) using rocQuantum primitives.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
import inspect
from numbers import Integral, Real
from typing import Callable, Dict, Any, Optional
import warnings
import numpy as np
from abc import ABC, abstractmethod

# --- rocQuantum Imports ---
try:
    import rocq
    from rocq.operator import PauliOperator, QuantumOperator, iter_pauli_terms
    from rocq.kernel import QuantumKernel, observe
except ImportError:
    # Fallback: allow module to be imported for inspection even if rocq
    # is not installed in the current environment.
    rocq = None  # type: ignore
    PauliOperator = None  # type: ignore
    QuantumOperator = None  # type: ignore
    iter_pauli_terms = None  # type: ignore
    QuantumKernel = None  # type: ignore
    observe = None  # type: ignore

# --- Optional Third-party Imports ---
try:
    from scipy.optimize import minimize, OptimizeResult
except ImportError:  # pragma: no cover - exercised in minimal CI environments.
    class OptimizeResult(dict):
        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError as exc:
                raise AttributeError(key) from exc

    def minimize(*args, **kwargs):
        raise RuntimeError("SciPy is required to use SciPyOptimizer.minimize().")

# --- Type Hinting Placeholders ---
AnsatzKernel = Callable[..., None]  # An ansatz is a kernel function
_FALLBACK_SUPPORTED_BACKENDS = (
    "state_vector",
    "density_matrix",
    "stabilizer",
    "tableau",
    "clifford",
)

_VECTOR_PARAMETER_NAMES = {
    "params",
    "parameters",
    "thetas",
    "angles",
    "parameter_vector",
    "parameter_values",
    "values",
}

_SCIPY_POSITIVE_INTEGER_OPTIONS = {
    "maxiter",
    "maxfev",
    "maxfun",
    "maxls",
    "maxcg",
    "maxcor",
    "maxcv",
}
_SCIPY_POSITIVE_REAL_OPTIONS = {
    "tol",
    "ftol",
    "gtol",
    "xtol",
    "xatol",
    "fatol",
    "eps",
    "rhobeg",
    "catol",
    "initial_tr_radius",
    "final_tr_radius",
}


class ObserveExecutionType(Enum):
    """Classify an expectation evaluation performed during optimization."""

    function = "function"
    gradient = "gradient"


@dataclass(frozen=True)
class ObserveIteration:
    """Immutable record of one objective or gradient expectation evaluation."""

    parameters: tuple[float, ...]
    result: float
    type: ObserveExecutionType

    def __post_init__(self):
        parameters = tuple(
            float(value) for value in _parameter_vector(self.parameters, "parameters")
        )
        result = _finite_real_scalar(self.result, "result")
        if not isinstance(self.type, ObserveExecutionType):
            raise ValueError("type must be an ObserveExecutionType value.")
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "result", result)


def _parameter_prefers_vector(parameter: inspect.Parameter) -> bool:
    if parameter.name.lower() in _VECTOR_PARAMETER_NAMES:
        return True

    annotation = parameter.annotation
    if annotation is inspect.Parameter.empty:
        return False

    annotation_text = getattr(annotation, "__name__", str(annotation)).lower()
    return any(token in annotation_text for token in ("ndarray", "array", "sequence", "list", "tuple"))


def _parameter_vector(params, label: str = "parameters") -> np.ndarray:
    if isinstance(params, (str, bytes)) or isinstance(params, (bool, np.bool_)):
        raise ValueError(f"{label} must be finite numeric values.")
    try:
        raw_values = np.asarray(params, dtype=object).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite numeric values.") from exc

    values = []
    for value in raw_values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(f"{label} must be finite numeric values.")
        normalized = float(value)
        if not np.isfinite(normalized):
            raise ValueError(f"{label} must be finite.")
        values.append(normalized)
    vector = np.asarray(values, dtype=float)
    return vector


def _positive_integer(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer.")
    integer = int(value)
    if integer <= 0:
        raise ValueError(f"{name} must be positive.")
    return integer


def _finite_real_scalar(value, label: str) -> float:
    try:
        scalar = np.asarray(value).reshape(())
    except ValueError as exc:
        raise ValueError(f"{label} must be a finite real scalar.") from exc

    raw_value = scalar.item()
    if isinstance(raw_value, (bool, np.bool_)):
        raise ValueError(f"{label} must be a finite real scalar.")
    if isinstance(raw_value, (complex, np.complexfloating)):
        real_part = float(np.real(raw_value))
        imag_part = float(np.imag(raw_value))
        if not np.isfinite(real_part) or not np.isfinite(imag_part):
            raise ValueError(f"{label} must be finite.")
        if abs(imag_part) > 1.0e-9:
            raise ValueError(f"{label} must be real.")
        return real_part
    if not isinstance(raw_value, Real):
        raise ValueError(f"{label} must be a finite real scalar.")
    real_value = float(raw_value)
    if not np.isfinite(real_value):
        raise ValueError(f"{label} must be finite.")
    return real_value


def _positive_integer_option(value, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be a positive integer.")
    integer = int(value)
    if integer <= 0:
        raise ValueError(f"{label} must be positive.")
    return integer


def _positive_real_option(value, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a positive finite real number.")
    real_value = float(value)
    if not np.isfinite(real_value) or real_value <= 0.0:
        raise ValueError(f"{label} must be positive and finite.")
    return real_value


def _validate_scipy_nested_options(options) -> dict:
    if not isinstance(options, Mapping):
        raise ValueError("SciPyOptimizer 'options' must be a mapping.")

    normalized = {}
    for key, value in options.items():
        if not isinstance(key, str):
            raise ValueError("SciPyOptimizer 'options' keys must be strings.")
        if key in _SCIPY_POSITIVE_INTEGER_OPTIONS:
            normalized[key] = _positive_integer_option(
                value,
                f"SciPyOptimizer options['{key}']",
            )
        elif key in _SCIPY_POSITIVE_REAL_OPTIONS:
            normalized[key] = _positive_real_option(
                value,
                f"SciPyOptimizer options['{key}']",
            )
        else:
            normalized[key] = value
    return normalized


def _validate_scipy_minimize_kwargs(options) -> dict:
    if not isinstance(options, Mapping):
        raise ValueError("SciPyOptimizer options must be a mapping.")

    normalized = {}
    for key, value in options.items():
        if not isinstance(key, str):
            raise ValueError("SciPyOptimizer option keys must be strings.")
        if key == "method":
            if not isinstance(value, str) and not callable(value):
                raise ValueError("SciPyOptimizer method option must be a string or callable.")
            if isinstance(value, str) and not value:
                raise ValueError("SciPyOptimizer method option must be non-empty.")
            normalized[key] = value
        elif key == "tol":
            normalized[key] = _positive_real_option(value, "SciPyOptimizer tol option")
        elif key == "callback":
            if value is not None and not callable(value):
                raise ValueError("SciPyOptimizer callback option must be callable or None.")
            normalized[key] = value
        elif key == "jac":
            if isinstance(value, str):
                if value not in {"2-point", "3-point", "cs"}:
                    raise ValueError(
                        "SciPyOptimizer jac string must be '2-point', '3-point', or 'cs'."
                    )
            elif value is True or (
                value is not None and value is not False and not callable(value)
            ):
                raise ValueError(
                    "SciPyOptimizer jac must be a finite-difference string, callable, "
                    "False, or None."
                )
            normalized[key] = value
        elif key == "options":
            normalized[key] = _validate_scipy_nested_options(value)
        else:
            normalized[key] = value
    return normalized


def _supported_backend_names() -> tuple[str, ...]:
    if rocq is None or not hasattr(rocq, "runtime_capabilities"):
        return _FALLBACK_SUPPORTED_BACKENDS
    capabilities = rocq.runtime_capabilities()
    return tuple(capabilities.get("supported_backends", _FALLBACK_SUPPORTED_BACKENDS))


def _validate_backend_name(backend: str) -> str:
    supported = _supported_backend_names()
    if not isinstance(backend, str) or backend not in supported:
        raise ValueError(f"backend must be one of: {list(supported)}.")
    return backend


def _validate_boolean_option(value, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean.")
    return value


def _optimizer_result_attribute(result, attribute: str):
    try:
        return getattr(result, attribute)
    except AttributeError as exc:
        raise ValueError(f"optimizer result must provide '{attribute}'.") from exc


def _validate_hamiltonian(hamiltonian):
    if QuantumOperator is None:
        return hamiltonian
    if not isinstance(hamiltonian, QuantumOperator):
        raise ValueError("hamiltonian must be a rocq.operator.QuantumOperator.")
    return hamiltonian


def _validate_ansatz_kernel(ansatz_kernel):
    if QuantumKernel is not None and isinstance(ansatz_kernel, QuantumKernel):
        return ansatz_kernel
    if not callable(ansatz_kernel):
        raise ValueError("ansatz_kernel must be a callable or rocq.kernel.QuantumKernel.")
    return ansatz_kernel


def _coerce_ansatz_kernel(ansatz_kernel):
    """Adapt a CUDA-QX-style callable ansatz to the canonical rocq kernel type."""

    ansatz_kernel = _validate_ansatz_kernel(ansatz_kernel)
    if QuantumKernel is not None and isinstance(ansatz_kernel, QuantumKernel):
        return ansatz_kernel
    if rocq is None or not hasattr(rocq, "kernel"):
        raise RuntimeError(
            "Canonical 'rocq' package is required to adapt a callable ansatz."
        )

    original_callable = ansatz_kernel

    @rocq.kernel
    def callable_ansatz_adapter(parameters):
        # CUDA-QX functional VQE defines this boundary as one parameter vector;
        # the user's local variable name and annotation must not affect dispatch.
        original_callable(_parameter_vector(parameters).copy())

    callable_name = getattr(original_callable, "__name__", "callable")
    callable_ansatz_adapter.name = f"{callable_name}_adapter"
    return callable_ansatz_adapter


def _ansatz_parameter_args(params: np.ndarray, ansatz_kernel: AnsatzKernel):
    params = _parameter_vector(params)
    ansatz_kernel = _validate_ansatz_kernel(ansatz_kernel)
    underlying = getattr(ansatz_kernel, "_func", ansatz_kernel)
    try:
        signature = inspect.signature(underlying)
    except (TypeError, ValueError):
        return tuple(float(value) for value in params)

    required_keyword_only = [
        parameter.name
        for parameter in signature.parameters.values()
        if parameter.kind == inspect.Parameter.KEYWORD_ONLY
        and parameter.default is inspect.Parameter.empty
    ]
    if required_keyword_only:
        raise ValueError("ansatz_kernel must not require keyword-only parameters.")

    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }
        and parameter.default is inspect.Parameter.empty
    ]
    optional_positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }
        and parameter.default is not inspect.Parameter.empty
    ]
    has_varargs = any(
        parameter.kind == inspect.Parameter.VAR_POSITIONAL
        for parameter in signature.parameters.values()
    )
    if len(positional) == 1 and not optional_positional and not has_varargs:
        if _parameter_prefers_vector(positional[0]):
            return (params.copy(),)

    minimum_parameters = len(positional)
    maximum_parameters = None if has_varargs else minimum_parameters + len(optional_positional)
    if params.size < minimum_parameters:
        raise ValueError(
            f"ansatz_kernel expects at least {minimum_parameters} parameter value(s); "
            f"got {params.size}."
        )
    if maximum_parameters is not None and params.size > maximum_parameters:
        raise ValueError(
            f"ansatz_kernel expects at most {maximum_parameters} parameter value(s); "
            f"got {params.size}."
        )
    return tuple(float(value) for value in params)


_EXACT_TWO_POINT_SHIFT_GATES = frozenset({"rx", "ry", "rz", "p"})
_MIN_STABLE_HOST_GRADIENT_STEP = 5.0e-3


def _context_gate_signature(context):
    return tuple(
        (operation.name, tuple(operation.targets), tuple(sorted(operation.params)))
        for operation in context.ops
    )


def _context_parameter_values(context):
    return tuple(
        float(operation.params[name])
        for operation in context.ops
        for name in sorted(operation.params)
    )


def _supports_exact_parameter_shift(
    ansatz_kernel,
    params: np.ndarray,
    parameter_index: int,
) -> bool:
    """Prove the eager circuit fits the exact two-evaluation shift rule.

    A host parameter is eligible only when the circuit structure is invariant and
    it controls at most one RX/RY/RZ/P angle with affine coefficient +1 or -1.
    Shared, scaled, nonlinear, controlled, or otherwise opaque parameterizations
    deliberately fall back to a numerical derivative.
    """

    if QuantumKernel is None or not isinstance(ansatz_kernel, QuantumKernel):
        # Preserve the legacy VQE_Solver contract for opaque callables.  The
        # CUDA-QX functional vqe() entry point adapts callables to QuantumKernel
        # first, so real recorded circuits still receive the proof/fallback path.
        return True

    local_probe = 1.0e-6
    exact_shift = np.pi / 2.0
    offsets = (0.0, local_probe, -local_probe, exact_shift, -exact_shift)
    contexts = []
    try:
        for offset in offsets:
            probe_parameters = params.copy()
            probe_parameters[parameter_index] += offset
            contexts.append(
                ansatz_kernel.build(
                    *_ansatz_parameter_args(probe_parameters, ansatz_kernel)
                )
            )
    except (TypeError, ValueError, RuntimeError):
        return False

    signatures = [_context_gate_signature(context) for context in contexts]
    qubit_counts = [context._next_qubit_index for context in contexts]
    if any(signature != signatures[0] for signature in signatures[1:]):
        return False
    if any(count != qubit_counts[0] for count in qubit_counts[1:]):
        return False

    parameter_values = [_context_parameter_values(context) for context in contexts]
    if any(len(values) != len(parameter_values[0]) for values in parameter_values[1:]):
        return False

    base_values, plus_local, minus_local, plus_exact, minus_exact = parameter_values
    changed_indices = [
        index
        for index, base_value in enumerate(base_values)
        if not (
            np.isclose(plus_local[index], base_value, rtol=0.0, atol=1.0e-10)
            and np.isclose(minus_local[index], base_value, rtol=0.0, atol=1.0e-10)
            and np.isclose(plus_exact[index], base_value, rtol=0.0, atol=1.0e-10)
            and np.isclose(minus_exact[index], base_value, rtol=0.0, atol=1.0e-10)
        )
    ]

    # A structurally invariant circuit with no parameter-dependent gate has a
    # constant objective; the two-point rule returns the exact zero derivative.
    if not changed_indices:
        return True
    if len(changed_indices) != 1:
        return False

    changed_index = changed_indices[0]
    flat_gate_names = [
        operation.name
        for operation in contexts[0].ops
        for _ in sorted(operation.params)
    ]
    if flat_gate_names[changed_index] not in _EXACT_TWO_POINT_SHIFT_GATES:
        return False

    slope = (
        plus_local[changed_index] - minus_local[changed_index]
    ) / (2.0 * local_probe)
    if not np.isclose(abs(slope), 1.0, rtol=1.0e-7, atol=1.0e-7):
        return False
    signed_slope = 1.0 if slope > 0.0 else -1.0
    base_value = base_values[changed_index]
    return bool(
        np.isclose(
            plus_exact[changed_index],
            base_value + signed_slope * exact_shift,
            rtol=1.0e-9,
            atol=1.0e-9,
        )
        and np.isclose(
            minus_exact[changed_index],
            base_value - signed_slope * exact_shift,
            rtol=1.0e-9,
            atol=1.0e-9,
        )
    )

# --- Optimizer Strategy Pattern Definition ---

class Optimizer(ABC):
    """
    Abstract base class for classical optimizers (Strategy Pattern).

    This interface allows for different optimization algorithms to be seamlessly
    plugged into the VQE_Solver. To add a new optimizer, create a concrete
    class that inherits from this one and implement the `minimize` method.
    """
    @abstractmethod
    def minimize(
        self,
        fun: Callable[[np.ndarray], float],
        x0: np.ndarray,
        args: tuple = ()
    ) -> OptimizeResult:
        """
        Executes the minimization routine.

        Args:
            fun (Callable): The objective function to minimize.
            x0 (np.ndarray): The initial guess for the parameters.
            args (tuple): Extra arguments to pass to the objective function.

        Returns:
            OptimizeResult: The result of the optimization.
        """
        pass

class SciPyOptimizer(Optimizer):
    """
    A concrete implementation of the Optimizer strategy that wraps
    `scipy.optimize.minimize`.
    """
    def __init__(self, options: Dict[str, Any] = None):
        """
        Initializes the SciPyOptimizer.

        Args:
            options (Dict[str, Any], optional): A dictionary of options
                (e.g., {'method': 'BFGS', 'tol': 1e-6}) to be passed to
                `scipy.optimize.minimize`. Defaults to a standard configuration.
        """
        if options is None:
            self.options = {'method': 'COBYLA', 'tol': 1e-6}
        else:
            self.options = _validate_scipy_minimize_kwargs(options)

    def minimize(
        self,
        fun: Callable[[np.ndarray], float],
        x0: np.ndarray,
        args: tuple = ()
    ) -> OptimizeResult:
        """
        Minimizes the objective function using scipy.optimize.minimize.
        """
        return minimize(
            fun=fun,
            x0=x0,
            args=args,
            **self.options
        )


class _SciPyCallableOptimizer(Optimizer):
    """Adapt a SciPy-compatible ``minimize`` callable to ``Optimizer``."""

    def __init__(self, minimize_callable, options: Optional[Mapping] = None):
        if inspect.isclass(minimize_callable) or not callable(minimize_callable):
            raise ValueError("optimizer callable must follow scipy.optimize.minimize.")
        self.minimize_callable = minimize_callable
        self.options = _validate_scipy_minimize_kwargs(options or {})

    def minimize(
        self,
        fun: Callable[[np.ndarray], float],
        x0: np.ndarray,
        args: tuple = (),
    ) -> OptimizeResult:
        return self.minimize_callable(fun=fun, x0=x0, args=args, **self.options)


def _normalize_optimizer_options(options) -> dict:
    if options is None:
        return {}
    if not isinstance(options, Mapping):
        raise ValueError("optimizer_options must be a mapping or None.")
    if any(not isinstance(key, str) for key in options):
        raise ValueError("optimizer_options keys must be strings.")
    return dict(options)


def _merge_forwarded_scipy_options(
    optimizer_options,
    *,
    method=None,
    jac=None,
    callback=None,
    options=None,
):
    """Merge the CUDA-QX documented SciPy keywords with conflict checks."""

    merged = _normalize_optimizer_options(optimizer_options)
    forwarded = {
        "method": method,
        "jac": jac,
        "callback": callback,
        "options": options,
    }
    for name, value in forwarded.items():
        if value is None:
            continue
        if name in merged:
            raise ValueError(
                f"top-level {name} conflicts with optimizer_options['{name}']."
            )
        merged[name] = value
    return merged or None


def _registered_scipy_options(
    method: str,
    *,
    tol: float,
    max_iterations: Optional[int],
    optimizer_options,
) -> dict:
    options = _normalize_optimizer_options(optimizer_options)
    if "method" in options or "tol" in options:
        raise ValueError(
            "optimizer_options must not override the registered optimizer method or tol."
        )

    nested = options.pop("options", {})
    if not isinstance(nested, Mapping):
        raise ValueError("optimizer_options['options'] must be a mapping.")
    nested = dict(nested)
    if max_iterations is not None:
        if "maxiter" in nested:
            raise ValueError(
                "max_iterations conflicts with optimizer_options['options']['maxiter']."
            )
        nested["maxiter"] = _positive_integer(max_iterations, "max_iterations")

    options["method"] = method
    options["tol"] = _positive_real_option(tol, "tol")
    if nested:
        options["options"] = nested
    return options


def _normalize_optimizer(
    optimizer="cobyla",
    *,
    tol: float = 1.0e-6,
    max_iterations: Optional[int] = None,
    optimizer_options=None,
) -> Optimizer:
    """Normalize registered names, SciPy callables, or optimizer objects."""

    if optimizer is None:
        optimizer = "cobyla"

    if isinstance(optimizer, str):
        normalized_name = optimizer.strip().lower()
        methods = {"cobyla": "COBYLA", "lbfgs": "L-BFGS-B"}
        if normalized_name not in methods:
            raise ValueError("optimizer must be 'cobyla', 'lbfgs', or a SciPy-compatible optimizer.")
        return SciPyOptimizer(
            _registered_scipy_options(
                methods[normalized_name],
                tol=tol,
                max_iterations=max_iterations,
                optimizer_options=optimizer_options,
            )
        )

    minimize_method = getattr(optimizer, "minimize", None)
    if callable(minimize_method):
        if optimizer_options is not None or max_iterations is not None:
            raise ValueError(
                "optimizer_options and max_iterations are only configurable for "
                "registered names or SciPy minimize callables."
            )
        _positive_real_option(tol, "tol")
        return optimizer

    if callable(optimizer) and not inspect.isclass(optimizer):
        callable_options = _normalize_optimizer_options(optimizer_options)
        if "tol" in callable_options:
            raise ValueError("optimizer_options must not override tol.")
        callable_options["tol"] = _positive_real_option(tol, "tol")
        nested = callable_options.get("options", {})
        if not isinstance(nested, Mapping):
            raise ValueError("optimizer_options['options'] must be a mapping.")
        nested = dict(nested)
        if max_iterations is not None:
            if "maxiter" in nested:
                raise ValueError(
                    "max_iterations conflicts with optimizer_options['options']['maxiter']."
                )
            nested["maxiter"] = _positive_integer(max_iterations, "max_iterations")
        if nested:
            callable_options["options"] = nested
        return _SciPyCallableOptimizer(optimizer, callable_options)

    raise ValueError(
        "optimizer must be 'cobyla', 'lbfgs', a scipy.optimize.minimize callable, "
        "or an object with minimize()."
    )

# --- VQE Solver ---

class VQE_Solver:
    """
    A high-level solver for the Variational Quantum Eigensolver (VQE) algorithm.

    This class uses a Strategy Pattern for its optimizer, allowing for easy
    extensibility with different classical optimization routines.
    """

    def __init__(
        self,
        optimizer: Optimizer = None,
        backend: str = "state_vector",
        verbose: bool = False,
    ):
        """
        Initializes the VQE_Solver.

        Args:
            optimizer (Optimizer, optional): A concrete optimizer instance that
                adheres to the Optimizer interface. If None, a default
                `SciPyOptimizer` is used.
        """
        if optimizer is None:
            self.optimizer = SciPyOptimizer()
        else:
            minimize_fn = getattr(optimizer, "minimize", None)
            if not callable(minimize_fn):
                raise ValueError("optimizer must define a callable minimize method.")
            self.optimizer = optimizer
        self.backend = _validate_backend_name(backend)
        self.verbose = _validate_boolean_option(verbose, "verbose")

        self._intermediate_results = []
        self._parameter_shift_fallback_warned = False

    def _objective_function(
        self,
        params: np.ndarray,
        hamiltonian: QuantumOperator,
        ansatz_kernel: AnsatzKernel,
        num_qubits: int,
        record_intermediate: bool = True,
        execution_type: ObserveExecutionType = ObserveExecutionType.function,
    ) -> float:
        """
        Internal objective function evaluated by the classical optimizer.
        """
        params = _parameter_vector(params)
        if not isinstance(execution_type, ObserveExecutionType):
            raise ValueError("execution_type must be an ObserveExecutionType value.")
        _positive_integer(num_qubits, "num_qubits")
        if observe is None:
            raise RuntimeError(
                "Canonical 'rocq' package is not available. Install the Python package "
                "and retry."
            )
        hamiltonian = _validate_hamiltonian(hamiltonian)
        ansatz_kernel = _validate_ansatz_kernel(ansatz_kernel)
        energy = observe(
            ansatz_kernel,
            hamiltonian,
            *_ansatz_parameter_args(params, ansatz_kernel),
            backend=self.backend,
        )
        energy = _finite_real_scalar(energy, "observed energy")
        if record_intermediate:
            self._intermediate_results.append({
                "parameters": params.copy(),
                "energy": energy,
                "type": execution_type,
            })
        return energy

    def evaluate_energy(
        self,
        hamiltonian: QuantumOperator,
        ansatz_kernel: AnsatzKernel,
        num_qubits: int,
        parameters: np.ndarray,
        record_intermediate: bool = False,
    ) -> float:
        """Evaluate the VQE objective once without invoking the optimizer."""
        return self._objective_function(
            parameters,
            hamiltonian,
            ansatz_kernel,
            num_qubits,
            record_intermediate=record_intermediate,
        )

    def estimate_gradient(
        self,
        params: np.ndarray,
        hamiltonian: QuantumOperator,
        ansatz_kernel: AnsatzKernel,
        num_qubits: int,
        method: str = "parameter_shift",
        step: float = 1e-5,
        record_intermediate: bool = False,
    ) -> np.ndarray:
        """Estimate the VQE objective gradient for the supported experimental subset."""
        params = _parameter_vector(params)
        _positive_integer(num_qubits, "num_qubits")
        gradient = np.zeros_like(params, dtype=float)
        if not isinstance(record_intermediate, bool):
            raise ValueError("record_intermediate must be a boolean.")
        if not isinstance(method, str):
            raise ValueError(
                "method must be 'parameter_shift', 'central_difference', or "
                "'forward_difference'."
            )
        method = method.lower()

        if method == "parameter_shift":
            if isinstance(step, (bool, np.bool_)) or not isinstance(step, Real):
                raise ValueError("finite_diff step must be a positive finite real number.")
            fallback_shift = float(step)
            if not np.isfinite(fallback_shift) or fallback_shift <= 0:
                raise ValueError("finite_diff step must be positive and finite.")
            shift = np.pi / 2.0
            scale = 0.5
            is_forward = False
        elif method in {
            "finite_diff",
            "finite_difference",
            "central_difference",
        }:
            if isinstance(step, (bool, np.bool_)) or not isinstance(step, Real):
                raise ValueError("finite_diff step must be a positive finite real number.")
            shift = float(step)
            if not np.isfinite(shift) or shift <= 0:
                raise ValueError("finite_diff step must be positive and finite.")
            scale = 1.0 / (2.0 * shift)
            is_forward = False
        elif method == "forward_difference":
            if isinstance(step, (bool, np.bool_)) or not isinstance(step, Real):
                raise ValueError("finite_diff step must be a positive finite real number.")
            shift = float(step)
            if not np.isfinite(shift) or shift <= 0:
                raise ValueError("finite_diff step must be positive and finite.")
            scale = 1.0 / shift
            is_forward = True
        else:
            raise ValueError(
                "method must be 'parameter_shift', 'central_difference', or "
                "'forward_difference'."
            )

        hamiltonian = _validate_hamiltonian(hamiltonian)
        ansatz_kernel = _validate_ansatz_kernel(ansatz_kernel)
        baseline = None
        if is_forward:
            baseline = self._objective_function(
                params,
                hamiltonian,
                ansatz_kernel,
                num_qubits,
                record_intermediate=record_intermediate,
                execution_type=ObserveExecutionType.gradient,
            )
        for idx in range(params.size):
            parameter_shift_fallback = (
                method == "parameter_shift"
                and not _supports_exact_parameter_shift(ansatz_kernel, params, idx)
            )
            if parameter_shift_fallback:
                # The local mock and common native state-vector ABI use
                # complex64 amplitudes.  A four-point centered stencil with a
                # precision-aware floor avoids catastrophic cancellation while
                # retaining O(h^4) truncation error.
                shift = max(fallback_shift, _MIN_STABLE_HOST_GRADIENT_STEP)
                if not self._parameter_shift_fallback_warned:
                    warnings.warn(
                        "The eager ansatz recorder could not prove the exact two-point "
                        "parameter-shift rule for every host parameter; using a "
                        "central-difference derivative for scaled, shared, nonlinear, "
                        "controlled, or opaque parameterization.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    self._parameter_shift_fallback_warned = True
            elif method == "parameter_shift":
                shift = np.pi / 2.0
                scale = 0.5
            plus = params.copy()
            plus[idx] += shift
            f_plus = self._objective_function(
                plus,
                hamiltonian,
                ansatz_kernel,
                num_qubits,
                record_intermediate=record_intermediate,
                execution_type=ObserveExecutionType.gradient,
            )
            if is_forward:
                gradient[idx] = scale * (f_plus - baseline)
                continue

            minus = params.copy()
            minus[idx] -= shift
            f_minus = self._objective_function(
                minus,
                hamiltonian,
                ansatz_kernel,
                num_qubits,
                record_intermediate=record_intermediate,
                execution_type=ObserveExecutionType.gradient,
            )
            if parameter_shift_fallback:
                plus_two = params.copy()
                plus_two[idx] += 2.0 * shift
                minus_two = params.copy()
                minus_two[idx] -= 2.0 * shift
                f_plus_two = self._objective_function(
                    plus_two,
                    hamiltonian,
                    ansatz_kernel,
                    num_qubits,
                    record_intermediate=record_intermediate,
                    execution_type=ObserveExecutionType.gradient,
                )
                f_minus_two = self._objective_function(
                    minus_two,
                    hamiltonian,
                    ansatz_kernel,
                    num_qubits,
                    record_intermediate=record_intermediate,
                    execution_type=ObserveExecutionType.gradient,
                )
                gradient[idx] = (
                    -f_plus_two + 8.0 * f_plus - 8.0 * f_minus + f_minus_two
                ) / (12.0 * shift)
                continue
            gradient[idx] = scale * (f_plus - f_minus)
        return gradient

    def solve(
        self,
        hamiltonian: QuantumOperator,
        ansatz_kernel: AnsatzKernel,
        num_qubits: int,
        initial_params: np.ndarray
    ) -> Dict[str, Any]:
        """
        Executes the VQE algorithm.
        """
        num_qubits = _positive_integer(num_qubits, "num_qubits")
        if self.verbose:
            print("Starting VQE optimization...")
        self._intermediate_results = []
        initial_parameter_vector = _parameter_vector(initial_params, label="initial_params")
        expected_parameter_count = initial_parameter_vector.size
        hamiltonian = _validate_hamiltonian(hamiltonian)
        ansatz_kernel = _validate_ansatz_kernel(ansatz_kernel)

        result = self.optimizer.minimize(
            fun=self._objective_function,
            x0=initial_parameter_vector,
            args=(hamiltonian, ansatz_kernel, num_qubits)
        )
        optimal_energy = _finite_real_scalar(
            _optimizer_result_attribute(result, "fun"),
            "optimizer result energy",
        )
        optimal_parameters = _parameter_vector(
            _optimizer_result_attribute(result, "x"),
            label="optimizer result parameters",
        )
        if optimal_parameters.size != expected_parameter_count:
            raise ValueError(
                "optimizer result parameters must contain "
                f"{expected_parameter_count} value(s)."
            )

        if self.verbose:
            print("VQE optimization finished.")

        solution = {
            'optimal_energy': optimal_energy,
            'optimal_parameters': optimal_parameters,
            'optimizer_result': result,
            'intermediate_results': self._intermediate_results
        }
        return solution


def _operator_qubit_count(operator) -> int:
    operator = _validate_hamiltonian(operator)
    if iter_pauli_terms is None:
        return 0
    maximum_index = -1
    try:
        terms = iter_pauli_terms(operator)
    except NotImplementedError:
        return 0
    for _, paulis in terms:
        for _, qubit in paulis:
            maximum_index = max(maximum_index, int(qubit))
    return maximum_index + 1


def _resolve_num_qubits(
    ansatz_kernel,
    parameters: np.ndarray,
    hamiltonian,
    explicit_num_qubits=None,
) -> int:
    operator_qubits = _operator_qubit_count(hamiltonian)
    kernel_qubits = 0
    if QuantumKernel is not None and isinstance(ansatz_kernel, QuantumKernel):
        ansatz_kernel.build(*_ansatz_parameter_args(parameters, ansatz_kernel))
        kernel_qubits = int(ansatz_kernel.num_qubits)

    if explicit_num_qubits is not None:
        resolved = _positive_integer(explicit_num_qubits, "num_qubits")
        if operator_qubits > resolved:
            raise ValueError("num_qubits is too small for the Hamiltonian qubit indices.")
        if kernel_qubits and resolved != kernel_qubits:
            raise ValueError("num_qubits must match the qubits allocated by the kernel.")
        return resolved

    if kernel_qubits and operator_qubits > kernel_qubits:
        raise ValueError("the Hamiltonian references qubits not allocated by the kernel.")

    resolved = max(operator_qubits, kernel_qubits)
    if resolved <= 0:
        raise ValueError(
            "num_qubits could not be inferred; provide num_qubits for an identity-only "
            "Hamiltonian or a non-rocq kernel."
        )
    return resolved


def _normalize_gradient_name(gradient) -> Optional[str]:
    if gradient is None:
        return None
    if not isinstance(gradient, str):
        raise ValueError(
            "gradient must be None, 'parameter_shift', 'central_difference', or "
            "'forward_difference'."
        )
    normalized = gradient.strip().lower()
    aliases = {
        "parameter_shift": "parameter_shift",
        "finite_diff": "central_difference",
        "finite_difference": "central_difference",
        "central_difference": "central_difference",
        "forward_difference": "forward_difference",
    }
    if normalized not in aliases:
        raise ValueError(
            "gradient must be None, 'parameter_shift', 'central_difference', or "
            "'forward_difference'."
        )
    return aliases[normalized]


def vqe(
    kernel,
    spin_op,
    initial_parameters,
    *,
    optimizer="cobyla",
    gradient=None,
    shots=None,
    max_iterations=None,
    verbose: bool = False,
    tol: float = 1.0e-6,
    backend: str = "state_vector",
    num_qubits=None,
    optimizer_options=None,
    callback=None,
    method=None,
    jac=None,
    options=None,
):
    """Run the CUDA-QX-style functional VQE contract on the host runtime.

    Shot-based expectation estimation is intentionally fail-closed until the
    canonical ``rocq.observe`` API exposes a sampled expectation contract.
    """

    if shots is not None:
        raise NotImplementedError(
            "shots-based VQE is not supported by the current rocq.observe contract."
        )

    parameters = _parameter_vector(initial_parameters, "initial_parameters")
    hamiltonian = _validate_hamiltonian(spin_op)
    ansatz_kernel = _coerce_ansatz_kernel(kernel)
    resolved_num_qubits = _resolve_num_qubits(
        ansatz_kernel,
        parameters,
        hamiltonian,
        explicit_num_qubits=num_qubits,
    )
    forwarded_optimizer_options = _merge_forwarded_scipy_options(
        optimizer_options,
        method=method,
        jac=jac,
        callback=callback,
        options=options,
    )
    normalized_optimizer = _normalize_optimizer(
        optimizer,
        tol=tol,
        max_iterations=max_iterations,
        optimizer_options=forwarded_optimizer_options,
    )
    solver = VQE_Solver(
        optimizer=normalized_optimizer,
        backend=backend,
        verbose=verbose,
    )

    gradient_name = _normalize_gradient_name(gradient)
    if gradient_name is not None:
        if not isinstance(normalized_optimizer, (SciPyOptimizer, _SciPyCallableOptimizer)):
            raise ValueError(
                "gradient is only supported with a registered optimizer or a "
                "SciPy minimize callable."
            )
        method_name = normalized_optimizer.options.get("method")
        if isinstance(method_name, str) and method_name.upper() == "COBYLA":
            raise ValueError("optimizer='cobyla' does not accept an explicit gradient.")
        if "jac" in normalized_optimizer.options:
            raise ValueError("gradient conflicts with optimizer_options['jac'].")

        def jacobian(current_parameters, objective, ansatz, qubits):
            return solver.estimate_gradient(
                current_parameters,
                objective,
                ansatz,
                qubits,
                method=gradient_name,
                record_intermediate=True,
            )

        normalized_optimizer.options["jac"] = jacobian

    solution = solver.solve(
        hamiltonian,
        ansatz_kernel,
        resolved_num_qubits,
        initial_params=parameters,
    )
    trace = [
        ObserveIteration(
            parameters=tuple(float(value) for value in entry["parameters"]),
            result=entry["energy"],
            type=entry.get("type", ObserveExecutionType.function),
        )
        for entry in solution["intermediate_results"]
    ]
    return (
        float(solution["optimal_energy"]),
        solution["optimal_parameters"].copy(),
        trace,
    )

if __name__ == '__main__':
    print("rocQuantum VQE example (experimental canonical API).")
    hamiltonian = PauliOperator("Z0")

    @rocq.kernel
    def simple_ansatz(theta: float):
        q = rocq.qvec(1)
        rocq.rx(theta, q[0])

    initial_parameters = np.array([0.5])
    num_qubits_for_problem = 1

    scipy_optimizer = SciPyOptimizer(options={'method': 'COBYLA', 'tol': 1e-6})
    vqe_solver = VQE_Solver(optimizer=scipy_optimizer, verbose=True)

    vqe_result = vqe_solver.solve(
        hamiltonian=hamiltonian,
        ansatz_kernel=simple_ansatz,
        num_qubits=num_qubits_for_problem,
        initial_params=initial_parameters
    )

    # 4. Print the Results
    print("\n--- VQE Results ---")
    print(f"Optimal Energy: {vqe_result['optimal_energy']:.8f}")
    print(f"Optimal Parameters: {vqe_result['optimal_parameters']}")
    print("-------------------")
