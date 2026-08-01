"""Educational two-qubit H2-style VQE on the canonical runtime.

The Hamiltonian coefficients are precomputed; rocQuantum does not currently
provide a chemistry Hamiltonian builder.
"""

import numpy as np

import rocq
from rocquantum.solvers import VQE_Solver

from _common import GridSearchOptimizer


H2_HAMILTONIAN = (
    -1.0523732
    + 0.3979374 * rocq.spin.z(0)
    - 0.3979374 * rocq.spin.z(1)
    - 0.0112801 * (rocq.spin.z(0) * rocq.spin.z(1))
    + 0.1809312 * (rocq.spin.x(0) * rocq.spin.x(1))
)


@rocq.kernel
def h2_ansatz(theta):
    q = rocq.qvec(2)
    rocq.x(q[0])
    rocq.ry(theta, q[1])
    rocq.cnot(q[1], q[0])


def main():
    solver = VQE_Solver(
        optimizer=GridSearchOptimizer(points=41),
        backend="state_vector",
    )
    result = solver.solve(
        H2_HAMILTONIAN,
        h2_ansatz,
        num_qubits=2,
        initial_params=np.array([0.0]),
    )
    assert np.isfinite(result["optimal_energy"])
    print("Reduced H2-style energy:", result["optimal_energy"])
    print("Parameter:", result["optimal_parameters"])


if __name__ == "__main__":
    main()
