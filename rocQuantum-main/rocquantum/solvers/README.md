# Experimental Solver Helpers

This package is an experimental, minimal higher-level layer over the canonical
`rocq` runtime. It is not a CUDA-QX parity claim.

Current supported subset:

- `VQE_Solver` evaluates canonical `rocq.operator.QuantumOperator` objectives through `rocq.observe()`, including supported Pauli, dense Hermitian, scaled/divided composite sums, and full-state CSR sparse observables on the state-vector backend or density-matrix correctness fallback.
- `VQE_Solver.estimate_gradient()` supports `parameter_shift` and `finite_diff`; scalar single-parameter inputs are normalized to one-element vectors for gradient and optimizer entry points.
- `make_maxcut_qaoa_kernel()` builds a MaxCut-style QAOA ansatz using H, CNOT,
  RZ, and RX gates. The cost phase uses a CNOT-RZ-CNOT block with angle
  `-gamma * w`, matching the non-global phase of `0.5 * w * (I - Zi Zj)`.
- `maxcut_cost_operator()` builds the weighted MaxCut cost operator as
  `0.5 * w * (I - Zi Zj)` for each edge, aggregating duplicate or reversed
  undirected edges before emitting ansatz cost phases or cost terms.
- `VQE_Solver` passes vectors as one ansatz argument when the target kernel has a single vector-style parameter, including one-element vectors, so the QAOA helper and vector-parameter ansatzes can be evaluated directly by the VQE objective path.

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
energy = solver._objective_function(
    np.array([0.25]),
    PauliOperator("Z0"),
    ansatz,
    num_qubits=1,
)
gradient = solver.estimate_gradient(np.array([0.25]), PauliOperator("Z0"), ansatz, 1)
```

For production-grade workflows, this layer still needs richer operator algebra,
native adjoint differentiation, optimizer integration tests on ROCm runners,
and broader algorithm coverage.
