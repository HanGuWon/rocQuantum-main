"""Run the supported three-qubit repetition-code subset."""

from rocquantum.qec import run_repetition_code_single_round


def main():
    result = run_repetition_code_single_round(
        initial_bits=[0, 0, 0],
        error_qubit=1,
        shots=32,
        backend="state_vector",
    )
    assert result["syndrome"] == [1, 1]
    assert "X1" in result["correction_applied"]
    print("Syndrome:", result["syndrome"])
    print("Correction:", result["correction_applied"])
    print("Scope:", result["experimental_supported_subset"])


if __name__ == "__main__":
    main()
