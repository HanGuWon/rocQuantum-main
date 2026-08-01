"""Run a Bell state through the canonical density-matrix API."""

import numpy as np

import rocq


@rocq.kernel
def bell_state():
    q = rocq.qvec(2)
    rocq.h(q[0])
    rocq.cnot(q[0], q[1])


def main():
    ideal = rocq.get_state(bell_state, backend="density_matrix")

    noise = rocq.NoiseModel()
    noise.add_channel("bit_flip", 0.1, on_qubits=[0], after_op="cnot")
    noisy = rocq.get_state(
        bell_state,
        backend="density_matrix",
        noise_model=noise,
    )

    np.testing.assert_allclose(np.trace(ideal), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.trace(noisy), 1.0, atol=1e-6)
    print("Ideal Bell density matrix:\n", ideal)
    print("Noisy Bell density matrix:\n", noisy)


if __name__ == "__main__":
    main()
