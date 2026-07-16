"""Sample and observe a Bell state with the canonical rocq runtime."""

import numpy as np

import rocq


@rocq.kernel
def bell_state():
    q = rocq.qvec(2)
    rocq.h(q[0])
    rocq.cnot(q[0], q[1])


def main():
    counts = rocq.sample(bell_state, 256, backend="state_vector")
    zz = rocq.observe(
        bell_state,
        rocq.PauliOperator("Z0 Z1"),
        backend="state_vector",
    )
    assert sum(counts.values()) == 256
    assert set(counts).issubset({"00", "11"})
    np.testing.assert_allclose(zz, 1.0, atol=1e-6)
    print("Counts:", counts)
    print("<Z0 Z1>:", zz)


if __name__ == "__main__":
    main()
