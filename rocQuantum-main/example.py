"""Small canonical rocq Bell-state example.

Set ``ROCQ_ENABLE_MOCK_BACKENDS=1`` to use the CPU correctness fallback when
the native ROCm bindings are not installed.
"""

import rocq


@rocq.kernel
def bell_state():
    q = rocq.qvec(2)
    rocq.h(q[0])
    rocq.cnot(q[0], q[1])


def main():
    counts = rocq.sample(bell_state, 100, backend="state_vector")
    print("Bell-state counts:", counts)
    assert sum(counts.values()) == 100
    assert set(counts).issubset({"00", "11"})


if __name__ == "__main__":
    main()
