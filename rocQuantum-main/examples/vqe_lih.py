"""Educational four-qubit LiH-style VQE using precomputed toy coefficients.

This exercises a larger canonical kernel, not chemistry generation or the
unreleased TensorNet slicing Python API.
"""

import numpy as np

import rocq
from rocquantum.solvers import VQE_Solver

from _common import GridSearchOptimizer


LIH_STYLE_HAMILTONIAN = (
    -7.8
    + 0.10 * rocq.spin.z(0)
    + 0.10 * rocq.spin.z(1)
    + 0.30 * rocq.spin.z(2)
    + 0.30 * rocq.spin.z(3)
    + 0.15 * (rocq.spin.z(0) * rocq.spin.z(1))
    + 0.02 * (rocq.spin.x(2) * rocq.spin.x(3))
)


@rocq.kernel
def lih_style_ansatz(theta):
    q = rocq.qvec(4)
    rocq.x(q[0])
    rocq.x(q[1])
    rocq.ry(theta, q[2])
    rocq.ry(theta, q[3])
    rocq.cnot(q[0], q[1])
    rocq.cnot(q[1], q[2])
    rocq.cnot(q[2], q[3])


def main():
    solver = VQE_Solver(
        optimizer=GridSearchOptimizer(points=33),
        backend="state_vector",
    )
    result = solver.solve(
        LIH_STYLE_HAMILTONIAN,
        lih_style_ansatz,
        num_qubits=4,
        initial_params=np.array([0.0]),
    )
    assert np.isfinite(result["optimal_energy"])
    print("Reduced LiH-style energy:", result["optimal_energy"])
    print("TensorNet slicing was not used; it has no canonical Python API yet.")


if __name__ == "__main__":
    main()
