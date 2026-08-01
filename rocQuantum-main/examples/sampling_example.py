"""Demonstrate full-register and selected-qubit sampling."""

import rocq


@rocq.kernel
def ghz_state():
    q = rocq.qvec(3)
    rocq.h(q[0])
    rocq.cnot(q[0], q[1])
    rocq.cnot(q[0], q[2])


def main():
    full = rocq.sample(ghz_state, 256, backend="state_vector")
    selected = rocq.sample(
        ghz_state,
        128,
        backend="state_vector",
        qubits=[0, 2],
    )
    assert sum(full.values()) == 256
    assert sum(selected.values()) == 128
    assert set(full).issubset({"000", "111"})
    assert set(selected).issubset({"00", "11"})
    print("All qubits:", full)
    print("Qubits [0, 2]:", selected)


if __name__ == "__main__":
    main()
