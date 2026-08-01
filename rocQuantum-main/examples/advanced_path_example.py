"""Inspect canonical MLIR for advanced controlled-gate recording."""

import numpy as np

import rocq


@rocq.kernel
def advanced_kernel(theta):
    q = rocq.qvec(4)
    rocq.h(q[0])
    rocq.crx(theta, q[0], q[1])
    rocq.mcx([q[0], q[1], q[2]], q[3])


def main():
    theta = np.pi / 5
    print(advanced_kernel.mlir(theta))
    groups = rocq.compiler_capabilities()["supported_gate_groups"]
    print("Compiler-advertised controlled gates:", groups["parametric_controlled"])


if __name__ == "__main__":
    main()
