"""Run parameterized gates through synchronous and host-async APIs."""

import numpy as np

import rocq


@rocq.kernel
def parameterized_bell(theta):
    q = rocq.qvec(2)
    rocq.ry(theta, q[0])
    rocq.cnot(q[0], q[1])


def main():
    theta = np.pi / 2
    state = rocq.get_state(parameterized_bell, theta, backend="state_vector")
    future = rocq.sample_async(
        parameterized_bell,
        128,
        theta,
        backend="state_vector",
    )
    counts = future.result()

    expected = np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2)
    np.testing.assert_allclose(state, expected, atol=1e-6)
    assert sum(counts.values()) == 128
    print("Final state:", state)
    print("Host-Future sample:", counts)


if __name__ == "__main__":
    main()
