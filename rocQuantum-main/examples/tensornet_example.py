"""Report the honest canonical boundary for hipTensorNet."""

import rocq


def main():
    print("rocq runtime status:", rocq.runtime_capabilities()["status"])
    print("Canonical TensorNetwork constructor: unavailable")
    print(
        "Use the native hipTensorNet C++ tests for source-level development. "
        "A stable Python TensorNetwork API has not been released yet."
    )


if __name__ == "__main__":
    main()
