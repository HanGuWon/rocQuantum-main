"""Evaluate a reduced H2-style objective with density-matrix noise."""

import numpy as np

import rocq


IDENTITY = np.eye(2, dtype=np.complex128)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
PAULI_Z = np.diag([1, -1]).astype(np.complex128)
H2_MATRIX = (
    -1.0523732 * np.eye(4)
    + 0.3979374 * np.kron(PAULI_Z, IDENTITY)
    - 0.3979374 * np.kron(IDENTITY, PAULI_Z)
    - 0.0112801 * np.kron(PAULI_Z, PAULI_Z)
    + 0.1809312 * np.kron(PAULI_X, PAULI_X)
)
H2_HAMILTONIAN = rocq.HermitianOperator(H2_MATRIX, targets=[0, 1])


@rocq.kernel
def h2_ansatz(theta):
    q = rocq.qvec(2)
    rocq.x(q[0])
    rocq.ry(theta, q[1])
    rocq.cnot(q[1], q[0])


def main():
    noise = rocq.NoiseModel()
    noise.add_channel("depolarizing", 0.01, on_qubits=[0, 1])

    evaluations = []
    for theta in np.linspace(-np.pi, np.pi, 25):
        energy = rocq.observe(
            h2_ansatz,
            H2_HAMILTONIAN,
            theta,
            backend="density_matrix",
            noise_model=noise,
        )
        evaluations.append((float(energy), theta))

    energy, theta = min(evaluations, key=lambda item: item[0])
    assert np.isfinite(energy)
    print("Noisy reduced H2-style energy:", energy)
    print("Parameter:", theta)


if __name__ == "__main__":
    main()
