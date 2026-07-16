# Experimental Solver Helpers

This package is an experimental, minimal higher-level layer over the canonical
`rocq` runtime. It is not a CUDA-QX parity claim.

Current supported subset:

- `vqe()` provides the CUDA-QX-style functional return contract
  `(energy, optimal_parameters, observe_trace)`.  Trace entries are immutable
  `ObserveIteration` values classified by `ObserveExecutionType`.  Registered
  optimizer names are exactly `cobyla` and `lbfgs`; a SciPy-compatible
  `minimize` callable or object with `minimize()` may also be supplied.  A
  CUDA-QX-style callable that accepts one parameter vector is adapted to a
  composable `QuantumKernel` independently of that argument's Python name.
  When `optimizer=scipy.optimize.minimize`, the documented top-level `method`,
  `jac`, `callback`, and `options` keywords are validated and forwarded. A
  non-`None` `shots` value fails closed because canonical `rocq.observe()` does
  not yet expose sampled expectation values.
- `qaoa()` accepts arbitrary real Pauli-sum problem and reference Hamiltonians,
  uses the default transverse-X mixer when none is supplied, implements shared
  `2 * p`, per-term full, and optional counterdiabatic-RY parameterizations,
  and returns a tuple-unpackable `QAOAResult` whose final configuration is a
  canonical `rocq.SampleResult`.  Its Pauli evolution uses
  the canonical CUDA-Q convention `exp_pauli(theta, P) = exp(+i theta P)`.
- `get_operator_pool("qaoa", num_qubits=n)` generates the CUDA-QX QAOA pool:
  all one-qubit X/Y terms and all two-qubit XX, YY, YZ, ZY, XY, YX, XZ, ZX
  terms.
- `adapt_vqe()` is a single-process host reference implementation.  It selects
  the largest pool gradient, grows the ansatz dynamically, supports warm/cold
  starts and convergence controls, and deliberately uses central finite
  differences so weighted or multi-term generators are not assigned an
  invalid two-point parameter-shift rule.  Shot-based and MQPU/MPI execution
  remain unsupported.
- `jordan_wigner()` transforms precomputed one- and two-body integrals and
  accepts the official `tol` spelling as an alias for `tolerance`;
  `MolecularHamiltonian.from_integrals()` retains immutable integral metadata.
  Geometry/XYZ/PySCF construction fails explicitly through `create_molecule()`;
  it is not represented as implemented.

- `rocquantum.solvers.solver_capabilities()` and the package-level
  `capabilities()` alias expose the experimental supported/unsupported solver
  contract, entry points, optional SciPy dependency, execution scope,
  hardware-evidence boundary, docs path, and ROCm validation limit for CUDA-QX
  comparisons. The execution scope is a host Python optimizer loop over
  canonical `rocq.observe()` calls, not native adjoint, distributed solver, or
  hybrid-workflow scheduler execution.
- `VQE_Solver.evaluate_energy()` and `VQE_Solver.solve()` evaluate canonical `rocq.operator.QuantumOperator` objectives through `rocq.observe()`, including supported Pauli, dense Hermitian, scaled/divided composite sums, and full-state CSR sparse observables on the state-vector backend or density-matrix correctness fallback. Hamiltonians must be canonical `QuantumOperator` instances, ansatz kernels must be `rocq.kernel.QuantumKernel` objects or callables, and both are rejected before backend or optimizer use.
- `VQE_Solver.solve()` is quiet by default for library and batch use; pass
  `verbose=True` to print start/finish progress messages. The `verbose`
  option must be a boolean.
- `VQE_Solver.estimate_gradient()` supports `parameter_shift` and finite
  differences. Exact two-evaluation parameter shift is used only when circuit
  recording proves that a host parameter controls one RX/RY/RZ/P angle with
  affine coefficient +1 or -1. Scaled, shared, nonlinear, controlled, or opaque
  recorded parameterizations emit a warning and use a precision-safe four-point
  centered derivative instead;
  scalar single-parameter inputs are normalized to one-element vectors for
  gradient and optimizer entry points, and gradient probes do not mutate the
  optimizer `intermediate_results` trace. VQE objective, optimizer initial
  parameters, and gradient parameters must be finite real values; boolean or
  string parameters are rejected instead of being coerced to numeric values.
  Gradient methods must be supported method-name strings.
  Solver backends must match the canonical runtime supported backend names.
  Ansatz positional parameter counts are validated before backend use. Observed
  energies and optimizer results must provide finite real `fun` energy
  and finite real `x` parameter values matching the initial parameter count;
  finite-difference steps must be positive finite real values.
  Custom optimizer objects must expose a callable `minimize()` method.
  `SciPyOptimizer` options must be a string-keyed mapping and are copied at
  construction so later caller-side mutation cannot silently change solver
  configuration. SciPy `method`, `tol`, `callback`, and nested `options`
  payloads are validated before `scipy.optimize.minimize()` is invoked,
  including positive numeric checks for common tolerance and iteration-limit
  fields.
- `make_maxcut_qaoa_kernel()` builds a MaxCut-style QAOA ansatz using H, CNOT,
  RZ, and RX gates. The cost phase uses a CNOT-RZ-CNOT block with angle
  `-gamma * w`, matching the non-global phase of `0.5 * w * (I - Zi Zj)`.
- `get_num_qaoa_parameters()` exposes the flat QAOA parameter count for the
  supported gamma/beta ansatz, returning `2 * layers` after validating the
  layer count and optional canonical cost operator.
- `maxcut_cost_operator()` builds the weighted MaxCut cost operator as
  `0.5 * w * (I - Zi Zj)` for each edge, accepting either `(u, v, weight)`
  edge entries or `{(u, v): weight}` mappings and aggregating duplicate or
  reversed undirected edges before emitting ansatz cost phases or cost terms.
  MaxCut helpers reject non-integer endpoints, non-positive `num_qubits` /
  `layers`, non-iterable edge containers, malformed edge entries, malformed
  edge-weight mapping keys, self-loops, out-of-range endpoints, and non-finite
  or non-real weights instead of silently truncating or propagating invalid
  problem data. QAOA
  ansatz runtime parameters and `solve_maxcut_qaoa()` initial parameters must
  also be finite real values.
- `solve_maxcut_qaoa()` wires that ansatz into `VQE_Solver` by minimizing the
  negated cost operator, so the reported `optimal_cut_value` maximizes the
  weighted MaxCut objective while preserving the positive `cost_operator` for
  inspection.
- `VQE_Solver` passes vectors as one ansatz argument when the target kernel has a single vector-style parameter, including one-element vectors, so the QAOA helper and vector-parameter ansatzes can be evaluated directly by the VQE objective path.

Install `rocquantum[solvers]` when using the default `SciPyOptimizer`; the
base package keeps SciPy optional for users that provide their own optimizer.

Minimal VQE example:

```python
import numpy as np
import rocq
from rocq.operator import PauliOperator
from rocquantum.solvers import VQE_Solver


@rocq.kernel
def ansatz(theta):
    q = rocq.qvec(1)
    rocq.rx(theta, q[0])


solver = VQE_Solver(backend="state_vector")
energy = solver.evaluate_energy(
    PauliOperator("Z0"),
    ansatz,
    num_qubits=1,
    parameters=np.array([0.25]),
)
gradient = solver.estimate_gradient(np.array([0.25]), PauliOperator("Z0"), ansatz, 1)
```

Minimal MaxCut QAOA wrapper example:

```python
import numpy as np
from rocquantum.solvers import get_num_qaoa_parameters, solve_maxcut_qaoa


parameter_count = get_num_qaoa_parameters(layers=1)
result = solve_maxcut_qaoa(
    num_qubits=2,
    edges=[(0, 1, 1.0)],
    layers=1,
    initial_params=np.zeros(parameter_count),
)
print(result["optimal_energy"], result["optimal_parameters"])
print(result["optimal_cut_value"])
```

For production-grade workflows, this layer still needs native adjoint
differentiation, ROCm hardware validation, geometry/PySCF and Bravyi-Kitaev
chemistry paths, UCC state preparation, and distributed MQPU/MPI execution.
