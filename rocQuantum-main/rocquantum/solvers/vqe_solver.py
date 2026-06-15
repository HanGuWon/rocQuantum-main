# Copyright (c) 2025-2026, rocQuantum Developers.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
High-level Variational Quantum Eigensolver (VQE) using rocQuantum primitives.
"""

from collections.abc import Mapping
import inspect
from numbers import Integral, Real
from typing import Callable, List, Dict, Any
import numpy as np
from abc import ABC, abstractmethod

# --- rocQuantum Imports ---
try:
    import rocq
    from rocq.operator import PauliOperator, QuantumOperator
    from rocq.kernel import QuantumKernel, observe
except ImportError:
    # Fallback: allow module to be imported for inspection even if rocq
    # is not installed in the current environment.
    rocq = None  # type: ignore
    PauliOperator = None  # type: ignore
    QuantumOperator = None  # type: ignore
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
    "parameter_vector",
    "parameter_values",
    "values",
}


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


def _optimizer_result_attribute(result, attribute: str):
    try:
        return getattr(result, attribute)
    except AttributeError as exc:
        raise ValueError(f"optimizer result must provide '{attribute}'.") from exc


def _ansatz_parameter_args(params: np.ndarray, ansatz_kernel: AnsatzKernel):
    params = _parameter_vector(params)
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
            if not isinstance(options, Mapping):
                raise ValueError("SciPyOptimizer options must be a mapping.")
            if any(not isinstance(key, str) for key in options):
                raise ValueError("SciPyOptimizer option keys must be strings.")
            self.options = dict(options)

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
        self.verbose = bool(verbose)

        self._intermediate_results = []

    def _objective_function(
        self,
        params: np.ndarray,
        hamiltonian: QuantumOperator,
        ansatz_kernel: AnsatzKernel,
        num_qubits: int,
        record_intermediate: bool = True,
    ) -> float:
        """
        Internal objective function evaluated by the classical optimizer.
        """
        params = _parameter_vector(params)
        _positive_integer(num_qubits, "num_qubits")
        if observe is None:
            raise RuntimeError(
                "Canonical 'rocq' package is not available. Install the Python package "
                "and retry."
            )
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
    ) -> np.ndarray:
        """Estimate the VQE objective gradient for the supported experimental subset."""
        params = _parameter_vector(params)
        _positive_integer(num_qubits, "num_qubits")
        gradient = np.zeros_like(params, dtype=float)
        if not isinstance(method, str):
            raise ValueError("method must be 'parameter_shift' or 'finite_diff'.")
        method = method.lower()

        if method == "parameter_shift":
            shift = np.pi / 2.0
            scale = 0.5
        elif method in {"finite_diff", "finite_difference"}:
            if isinstance(step, (bool, np.bool_)) or not isinstance(step, Real):
                raise ValueError("finite_diff step must be a positive finite real number.")
            shift = float(step)
            if not np.isfinite(shift) or shift <= 0:
                raise ValueError("finite_diff step must be positive and finite.")
            scale = 1.0 / (2.0 * shift)
        else:
            raise ValueError("method must be 'parameter_shift' or 'finite_diff'.")

        for idx in range(params.size):
            plus = params.copy()
            minus = params.copy()
            plus[idx] += shift
            minus[idx] -= shift
            f_plus = self._objective_function(
                plus,
                hamiltonian,
                ansatz_kernel,
                num_qubits,
                record_intermediate=False,
            )
            f_minus = self._objective_function(
                minus,
                hamiltonian,
                ansatz_kernel,
                num_qubits,
                record_intermediate=False,
            )
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
