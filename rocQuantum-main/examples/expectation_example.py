"""Evaluate Pauli observables on a canonical GHZ kernel."""

import numpy as np

import rocq


@rocq.kernel
def ghz_state():
    q = rocq.qvec(3)
    rocq.h(q[0])
    rocq.cnot(q[0], q[1])
    rocq.cnot(q[0], q[2])


def main():
    observables = {
        "<Z0 Z1>": (rocq.PauliOperator("Z0 Z1"), 1.0),
        "<X0 X1 X2>": (rocq.PauliOperator("X0 X1 X2"), 1.0),
        "<Z0>": (rocq.PauliOperator("Z0"), 0.0),
    }
    for label, (operator, expected) in observables.items():
        value = rocq.observe(ghz_state, operator, backend="state_vector")
        np.testing.assert_allclose(value, expected, atol=1e-6)
        print(f"{label} = {value:.6f}")


if __name__ == "__main__":
    main()
