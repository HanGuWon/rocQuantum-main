"""Report the current public-API boundary for TensorNet slicing."""

import rocq


def main():
    capabilities = rocq.runtime_capabilities()
    print("Canonical Python surface:", capabilities["primary_python_surface"])
    print("Tensor-network slicing public API: unavailable")
    print(
        "hipTensorNet has native source-level slicing work, but the canonical rocq "
        "package does not yet expose a TensorNetwork/plan/workspace contract."
    )


if __name__ == "__main__":
    main()
