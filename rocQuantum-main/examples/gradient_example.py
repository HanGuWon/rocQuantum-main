"""Estimate a parameter-shift gradient through the experimental solver layer."""

import numpy as np

import rocq
from rocquantum.solvers import VQE_Solver


@rocq.kernel
def ansatz(theta):
    q = rocq.qvec(1)
    rocq.rx(theta, q[0])


def main():
    theta = np.pi / 4
    solver = VQE_Solver(backend="state_vector")
    gradient = solver.estimate_gradient(
        np.array([theta]),
        rocq.PauliOperator("Z0"),
        ansatz,
        num_qubits=1,
        method="parameter_shift",
    )
    np.testing.assert_allclose(gradient, [-np.sin(theta)], atol=1e-6)
    print("Parameter-shift gradient:", gradient)


if __name__ == "__main__":
    main()
