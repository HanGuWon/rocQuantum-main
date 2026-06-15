# Copyright (c) 2025-2026, rocQuantum Developers.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
High-level Variational Quantum Eigensolver (VQE) using rocQuantum primitives.
"""

import inspect
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


def _ansatz_parameter_args(params: np.ndarray, ansatz_kernel: AnsatzKernel):
    params = np.asarray(params, dtype=float).reshape(-1)
    underlying = getattr(ansatz_kernel, "_func", ansatz_kernel)
    try:
        signature = inspect.signature(underlying)
    except (TypeError, ValueError):
        return tuple(float(value) for value in params)

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
    if len(positional) == 1 and (params.size != 1 or _parameter_prefers_vector(positional[0])):
        return (params.copy(),)
    return tuple(float(value) for value in params)


def _parameter_vector(params) -> np.ndarray:
    return np.asarray(params, dtype=float).reshape(-1)

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
            self.options = options

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
        self.optimizer = optimizer if optimizer is not None else SciPyOptimizer()
        self.backend = backend
        self.verbose = bool(verbose)

        self._intermediate_results = []

    def _objective_function(
        self,
        params: np.ndarray,
        hamiltonian: QuantumOperator,
        ansatz_kernel: AnsatzKernel,
        num_qubits: int
    ) -> float:
        """
        Internal objective function evaluated by the classical optimizer.
        """
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
        energy = float(np.real(energy))
        self._intermediate_results.append({
            "parameters": np.asarray(params, dtype=float).copy(),
            "energy": energy,
        })
        return energy

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
        gradient = np.zeros_like(params, dtype=float)
        method = method.lower()

        if method == "parameter_shift":
            shift = np.pi / 2.0
            scale = 0.5
        elif method in {"finite_diff", "finite_difference"}:
            shift = float(step)
            scale = 1.0 / (2.0 * shift)
        else:
            raise ValueError("method must be 'parameter_shift' or 'finite_diff'.")

        for idx in range(params.size):
            plus = params.copy()
            minus = params.copy()
            plus[idx] += shift
            minus[idx] -= shift
            f_plus = self._objective_function(plus, hamiltonian, ansatz_kernel, num_qubits)
            f_minus = self._objective_function(minus, hamiltonian, ansatz_kernel, num_qubits)
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
        if self.verbose:
            print("Starting VQE optimization...")
        self._intermediate_results = []

        result = self.optimizer.minimize(
            fun=self._objective_function,
            x0=_parameter_vector(initial_params),
            args=(hamiltonian, ansatz_kernel, num_qubits)
        )

        if self.verbose:
            print("VQE optimization finished.")

        solution = {
            'optimal_energy': result.fun,
            'optimal_parameters': result.x,
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
