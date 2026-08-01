"""Demonstrate deferred correction while reporting the dynamic-circuit boundary."""

import numpy as np

import rocq


@rocq.kernel
def coherent_teleportation(theta):
    q = rocq.qvec(3)
    rocq.ry(theta, q[0])
    rocq.h(q[1])
    rocq.cnot(q[1], q[2])
    rocq.cnot(q[0], q[1])
    rocq.h(q[0])
    # Deferred-measurement equivalents of classically controlled X and Z.
    rocq.cnot(q[1], q[2])
    rocq.cz(q[0], q[2])


def main():
    unsupported = rocq.compiler_capabilities()["unsupported_features"]
    print("Mid-circuit status:", next(item for item in unsupported if "mid-circuit" in item))

    theta = np.pi / 3
    value = rocq.observe(
        coherent_teleportation,
        rocq.PauliOperator("Z2"),
        theta,
        backend="state_vector",
    )
    np.testing.assert_allclose(value, np.cos(theta), atol=1e-6)
    print("Deferred-correction <Z2>:", value)


if __name__ == "__main__":
    main()
