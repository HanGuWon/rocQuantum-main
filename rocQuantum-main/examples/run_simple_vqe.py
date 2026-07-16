"""Run the experimental VQE solver with a dependency-free grid optimizer."""

import numpy as np

import rocq
from rocquantum.solvers import VQE_Solver

from _common import GridSearchOptimizer


@rocq.kernel
def ansatz(theta):
    q = rocq.qvec(1)
    rocq.ry(theta, q[0])


def main():
    solver = VQE_Solver(
        optimizer=GridSearchOptimizer(points=33),
        backend="state_vector",
    )
    result = solver.solve(
        hamiltonian=rocq.PauliOperator("Z0"),
        ansatz_kernel=ansatz,
        num_qubits=1,
        initial_params=np.array([0.0]),
    )
    np.testing.assert_allclose(result["optimal_energy"], -1.0, atol=1e-6)
    print("Optimal energy:", result["optimal_energy"])
    print("Optimal parameter:", result["optimal_parameters"])


if __name__ == "__main__":
    main()
