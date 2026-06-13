"""CUDA-Q-style Pauli spin-operator factories."""

from __future__ import annotations

from .operator import PauliOperator


def _target_index(target: int) -> int:
    index = int(target)
    if index < 0:
        raise ValueError("Spin-operator target must be non-negative.")
    return index


def i(target: int | None = None) -> PauliOperator:
    """Return an identity Pauli operator."""

    if target is None:
        return PauliOperator("I")
    return PauliOperator(f"I{_target_index(target)}")


def x(target: int) -> PauliOperator:
    """Return an X Pauli operator on ``target``."""

    return PauliOperator(f"X{_target_index(target)}")


def y(target: int) -> PauliOperator:
    """Return a Y Pauli operator on ``target``."""

    return PauliOperator(f"Y{_target_index(target)}")


def z(target: int) -> PauliOperator:
    """Return a Z Pauli operator on ``target``."""

    return PauliOperator(f"Z{_target_index(target)}")


identity = i


__all__ = ["i", "identity", "x", "y", "z"]
