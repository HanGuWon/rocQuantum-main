"""Show the current adjoint boundary and an explicit inverse circuit."""

import numpy as np

import rocq


@rocq.kernel
def gate_then_explicit_inverse():
    q = rocq.qvec(1)
    rocq.s(q[0])
    rocq.h(q[0])
    # The release-wired generic adjoint transform is not available yet.
    rocq.h(q[0])
    rocq.sdg(q[0])


def main():
    capabilities = rocq.compiler_capabilities()
    adjoint = capabilities["transform_pipeline"]["adjoint_generation"]
    print("Release-wired adjoint generation:", adjoint["release_wired"])
    print(gate_then_explicit_inverse.mlir())

    state = rocq.get_state(gate_then_explicit_inverse, backend="state_vector")
    np.testing.assert_allclose(state, np.array([1.0, 0.0]), atol=1e-6)
    print("The explicit inverse returned the register to |0>.")


if __name__ == "__main__":
    main()
