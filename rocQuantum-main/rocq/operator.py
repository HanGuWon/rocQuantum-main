# Task 3: Foundation for Operator Algebra
from __future__ import annotations
from abc import ABC, abstractmethod
from numbers import Number
from .kernel import execute, QuantumKernel

class QuantumOperator(ABC):
    """Abstract base class for all quantum operators.

    This class defines the basic interface for operators, including support
    for scalar multiplication and addition to form more complex operators.

    Args:
        coefficient (Number): A scalar coefficient for the operator.
            Defaults to 1.0.
    """
    def __init__(self, coefficient: Number = 1.0):
        self.coefficient = complex(coefficient)

    def __mul__(self, other):
        if isinstance(other, Number):
            # Create a new instance of the derived class with the new coefficient
            new_op = self.__class__.__new__(self.__class__)
            new_op.__dict__.update(self.__dict__)
            new_op.coefficient = self.coefficient * other
            return new_op
        raise NotImplementedError(f"Cannot multiply QuantumOperator by {type(other)}")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        if isinstance(other, QuantumOperator):
            return SumOperator([self, other])
        raise NotImplementedError(f"Cannot add QuantumOperator to {type(other)}")

    @abstractmethod
    def to_string(self) -> str:
        """Returns a user-friendly string representation of the operator."""
        pass

class PauliOperator(QuantumOperator):
    """Represents a single Pauli string operator (e.g., 0.5 * X0 Y1 Z2).

    This is a fundamental building block for creating Hamiltonians for many
    physics and chemistry problems.

    Args:
        pauli_string (str): A string defining the Pauli operators and their
            target qubits, e.g., "X0 Y1" or "Z2".
        coefficient (Number): A scalar coefficient for the operator.
            Defaults to 1.0.
    """
    def __init__(self, pauli_string: str, coefficient: Number = 1.0):
        super().__init__(coefficient)
        self.pauli_string = pauli_string

    def to_string(self) -> str:
        return f"{self.coefficient} * {self.pauli_string}"

class HermitianOperator(QuantumOperator):
    """Represents an operator defined by a Hermitian matrix."""
    def __init__(self, matrix, coefficient: Number = 1.0):
        super().__init__(coefficient)
        self.matrix = matrix

    def to_string(self) -> str:
        return f"{self.coefficient} * Hermitian(matrix)"

class SumOperator(QuantumOperator):
    """Represents a sum of QuantumOperators, typically a Hamiltonian."""
    def __init__(self, operators: list[QuantumOperator], coefficient: Number = 1.0):
        super().__init__(coefficient)
        self.terms = operators

    def __add__(self, other):
        if isinstance(other, SumOperator):
            return SumOperator(list(self.terms) + list(other.terms))
        elif isinstance(other, QuantumOperator):
            return SumOperator(list(self.terms) + [other])
        else:
            raise NotImplementedError

    def to_string(self) -> str:
        return " + ".join([f"({term.to_string()})" for term in self.terms])


def get_expectation_value(kernel: QuantumKernel, operator: QuantumOperator, backend: str, **kwargs):
    """Computes the expectation value of an operator.

    .. note::
        This function is not yet connected to a backend implementation.
        Use the hipStateVec C++ API directly for expectation values.

    Raises:
        NotImplementedError: Always, until a backend integration is available.
    """
    raise NotImplementedError(
        "Expectation value computation is not yet connected to a backend. "
        "Use the hipStateVec C++ API directly for expectation values."
    )
