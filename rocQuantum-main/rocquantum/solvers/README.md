# Experimental Solver Helpers

This package is an experimental, minimal higher-level layer over the canonical
`rocq` runtime. It is not a CUDA-QX parity claim.

Current supported subset:

- `VQE_Solver.evaluate_energy()` and `VQE_Solver.solve()` evaluate canonical `rocq.operator.QuantumOperator` objectives through `rocq.observe()`, including supported Pauli, dense Hermitian, scaled/divided composite sums, and full-state CSR sparse observables on the state-vector backend or density-matrix correctness fallback.
- `VQE_Solver.solve()` is quiet by default for library and batch use; pass
  `verbose=True` to print start/finish progress messages.
- `VQE_Solver.estimate_gradient()` supports `parameter_shift` and `finite_diff`;
  scalar single-parameter inputs are normalized to one-element vectors for
  gradient and optimizer entry points, and gradient probes do not mutate the
  optimizer `intermediate_results` trace. VQE objective, optimizer initial
  parameters, and gradient parameters must be finite real values; boolean or
  string parameters are rejected instead of being coerced to numeric values.
  Observed energies and optimizer result energies/parameters must also be finite
  real values; finite-difference steps must be positive finite real values.
- `make_maxcut_qaoa_kernel()` builds a MaxCut-style QAOA ansatz using H, CNOT,
  RZ, and RX gates. The cost phase uses a CNOT-RZ-CNOT block with angle
  `-gamma * w`, matching the non-global phase of `0.5 * w * (I - Zi Zj)`.
- `maxcut_cost_operator()` builds the weighted MaxCut cost operator as
  `0.5 * w * (I - Zi Zj)` for each edge, aggregating duplicate or reversed
  undirected edges before emitting ansatz cost phases or cost terms. MaxCut
  helpers reject non-integer endpoints, non-positive `num_qubits` / `layers`,
  self-loops, out-of-range endpoints, and non-finite or non-real weights
  instead of silently truncating or propagating invalid problem data. QAOA
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
from rocquantum.solvers import solve_maxcut_qaoa


result = solve_maxcut_qaoa(
    num_qubits=2,
    edges=[(0, 1, 1.0)],
    layers=1,
    initial_params=np.array([0.2, 0.4]),
)
print(result["optimal_energy"], result["optimal_parameters"])
print(result["optimal_cut_value"])
```

For production-grade workflows, this layer still needs richer operator algebra,
native adjoint differentiation, optimizer integration tests on ROCm runners,
and broader algorithm coverage.
