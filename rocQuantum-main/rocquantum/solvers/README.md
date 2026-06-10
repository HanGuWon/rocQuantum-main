# Experimental Solver Helpers

This package is an experimental, minimal higher-level layer over the canonical
`rocq` runtime. It is not a CUDA-QX parity claim.

Current supported subset:

- `VQE_Solver` evaluates Pauli-observable objectives through `rocq.observe()`.
- `VQE_Solver.estimate_gradient()` supports `parameter_shift` and `finite_diff`.
- `make_maxcut_qaoa_kernel()` builds a MaxCut-style QAOA ansatz using H, CNOT,
  RZ, and RX gates.
- `maxcut_cost_operator()` builds a Pauli-Z cost operator for that helper.

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

For production-grade workflows, this layer still needs richer operators,
optimizer integration tests on ROCm runners, and broader algorithm coverage.
