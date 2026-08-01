"""Verify canonical multi-controlled X behavior with basis states."""

import numpy as np

import rocq


@rocq.kernel
def apply_mcx(initial_state):
    q = rocq.qvec(3)
    for qubit in range(3):
        if (initial_state >> qubit) & 1:
            rocq.x(q[qubit])
    rocq.mcx([q[0], q[1]], q[2])


def main():
    expected = {3: 7, 5: 5, 6: 6, 7: 3}
    for initial, final in expected.items():
        state = rocq.get_state(apply_mcx, initial, backend="state_vector")
        measured = int(np.argmax(np.abs(state)))
        assert measured == final
        print(f"|{initial:03b}> -> |{measured:03b}>")


if __name__ == "__main__":
    main()
