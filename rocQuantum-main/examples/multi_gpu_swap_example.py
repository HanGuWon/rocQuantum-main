"""Exercise SWAP locally and report the separate multi-GPU evidence boundary."""

import numpy as np

import rocq


@rocq.kernel
def local_swap():
    q = rocq.qvec(3)
    rocq.x(q[0])
    rocq.swap(q[0], q[2])


def main():
    capabilities = rocq.distributed_capabilities()
    print("Distributed scope:", capabilities["execution_scope"])
    print("Hardware probe performed:", capabilities["hardware_evidence"]["probe_performed"])

    state = rocq.get_state(local_swap, backend="state_vector")
    assert int(np.argmax(np.abs(state))) == 4
    print("Local SWAP verified; this is not multi-GPU runtime proof.")


if __name__ == "__main__":
    main()
