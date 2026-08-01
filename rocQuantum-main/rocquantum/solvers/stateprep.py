"""CPU-reference state preparation helpers for chemistry ansatzes.

The UCCSD excitation ordering and Jordan-Wigner generators in this module
follow CUDA-QX 0.6.0.  The implementation records ordinary ``rocq`` gates, so
it remains backend-independent and can be exercised with the CPU state-vector
backend.
"""

from __future__ import annotations

import math
from numbers import Integral, Real

try:
    import rocq
    from rocq.operator import (
        PauliOperator,
        QuantumOperator,
        SumOperator,
        iter_pauli_terms,
    )
except ImportError:  # pragma: no cover - import-only environments.
    rocq = None  # type: ignore
    PauliOperator = None  # type: ignore
    QuantumOperator = None  # type: ignore
    SumOperator = None  # type: ignore
    iter_pauli_terms = None  # type: ignore


def _nonnegative_integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a non-negative integer.")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return normalized


def _positive_integer(value, name: str) -> int:
    normalized = _nonnegative_integer(value, name)
    if normalized == 0:
        raise ValueError(f"{name} must be a positive integer.")
    return normalized


def _finite_real(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number.")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    return normalized


def _system_occupations(num_electrons, num_qubits, spin=0):
    electrons = _nonnegative_integer(num_electrons, "num_electrons")
    qubits = _positive_integer(num_qubits, "num_qubits")
    total_spin = _nonnegative_integer(spin, "spin")

    if qubits % 2:
        raise ValueError("num_qubits must be even for interleaved spin orbitals.")
    if electrons > qubits:
        raise ValueError("num_electrons cannot exceed num_qubits.")
    if total_spin > electrons or (electrons - total_spin) % 2:
        raise ValueError(
            "spin must be no greater than num_electrons and have the same parity."
        )

    spatial_orbitals = qubits // 2
    occupied_beta_count = (electrons - total_spin) // 2
    occupied_alpha_count = electrons - occupied_beta_count
    if (
        occupied_alpha_count > spatial_orbitals
        or occupied_beta_count > spatial_orbitals
    ):
        raise ValueError(
            "num_electrons and spin do not fit in the available spatial orbitals."
        )

    occupied_alpha = [2 * index for index in range(occupied_alpha_count)]
    occupied_beta = [2 * index + 1 for index in range(occupied_beta_count)]
    virtual_alpha = [
        2 * index for index in range(occupied_alpha_count, spatial_orbitals)
    ]
    virtual_beta = [
        2 * index + 1 for index in range(occupied_beta_count, spatial_orbitals)
    ]
    return (
        qubits,
        occupied_alpha,
        virtual_alpha,
        occupied_beta,
        virtual_beta,
    )


def get_uccsd_excitations(num_electrons, num_qubits, spin=0):
    """Return CUDA-QX-ordered UCCSD single and double excitations.

    Spin orbitals are interleaved ``alpha_0, beta_0, alpha_1, beta_1, ...``.
    The result is ``(singles_alpha, singles_beta, doubles_mixed,
    doubles_alpha, doubles_beta)``.
    """

    (
        _,
        occupied_alpha,
        virtual_alpha,
        occupied_beta,
        virtual_beta,
    ) = _system_occupations(num_electrons, num_qubits, spin)

    singles_alpha = [
        [occupied, virtual] for occupied in occupied_alpha for virtual in virtual_alpha
    ]
    singles_beta = [
        [occupied, virtual] for occupied in occupied_beta for virtual in virtual_beta
    ]
    doubles_mixed = [
        [occupied_alpha_mode, occupied_beta_mode, virtual_beta_mode, virtual_alpha_mode]
        for occupied_alpha_mode in occupied_alpha
        for occupied_beta_mode in occupied_beta
        for virtual_beta_mode in virtual_beta
        for virtual_alpha_mode in virtual_alpha
    ]
    doubles_alpha = [
        [
            occupied_alpha[left],
            occupied_alpha[right],
            virtual_alpha[first],
            virtual_alpha[second],
        ]
        for left in range(len(occupied_alpha) - 1)
        for right in range(left + 1, len(occupied_alpha))
        for first in range(len(virtual_alpha) - 1)
        for second in range(first + 1, len(virtual_alpha))
    ]
    doubles_beta = [
        [
            occupied_beta[left],
            occupied_beta[right],
            virtual_beta[first],
            virtual_beta[second],
        ]
        for left in range(len(occupied_beta) - 1)
        for right in range(left + 1, len(occupied_beta))
        for first in range(len(virtual_beta) - 1)
        for second in range(first + 1, len(virtual_beta))
    ]
    return (
        singles_alpha,
        singles_beta,
        doubles_mixed,
        doubles_alpha,
        doubles_beta,
    )


def get_num_uccsd_parameters(num_electrons, num_qubits, spin=0) -> int:
    """Return the parameter count for :func:`uccsd`."""

    return sum(
        len(excitation_group)
        for excitation_group in get_uccsd_excitations(num_electrons, num_qubits, spin)
    )


def _pauli_term(assignments, coefficient: float, num_qubits: int):
    word = " ".join(
        f"{pauli}{qubit}"
        for qubit, pauli in sorted(assignments.items())
        if pauli != "I"
    )
    return PauliOperator(
        word or "I",
        coefficient=coefficient,
        num_qubits=num_qubits,
    )


def _single_excitation_generator(p_occ: int, q_virt: int, num_qubits: int):
    parity = {index: "Z" for index in range(p_occ + 1, q_virt)}
    positive = dict(parity)
    positive[p_occ] = "Y"
    positive[q_virt] = "X"
    negative = dict(parity)
    negative[p_occ] = "X"
    negative[q_virt] = "Y"
    return SumOperator(
        [
            _pauli_term(positive, 0.5, num_qubits),
            _pauli_term(negative, -0.5, num_qubits),
        ]
    )


def _canonical_double_indices(p_occ, q_occ, r_virt, s_virt):
    occupied_reversed = p_occ > q_occ
    virtual_reversed = r_virt > s_virt
    i_occ, j_occ = sorted((p_occ, q_occ))
    a_virt, b_virt = sorted((r_virt, s_virt))
    orientation = -1.0 if occupied_reversed != virtual_reversed else 1.0
    return i_occ, j_occ, a_virt, b_virt, orientation


def _double_excitation_generator(
    p_occ: int,
    q_occ: int,
    r_virt: int,
    s_virt: int,
    num_qubits: int,
):
    i_occ, j_occ, a_virt, b_virt, _ = _canonical_double_indices(
        p_occ, q_occ, r_virt, s_virt
    )
    parity = {
        index: "Z"
        for index in (
            *range(i_occ + 1, j_occ),
            *range(a_virt + 1, b_virt),
        )
    }
    pauli_patterns = (
        ("X", "X", "X", "Y", 0.125),
        ("X", "X", "Y", "X", 0.125),
        ("X", "Y", "Y", "Y", 0.125),
        ("Y", "X", "Y", "Y", 0.125),
        ("X", "Y", "X", "X", -0.125),
        ("Y", "X", "X", "X", -0.125),
        ("Y", "Y", "X", "Y", -0.125),
        ("Y", "Y", "Y", "X", -0.125),
    )
    terms = []
    for first, second, third, fourth, coefficient in pauli_patterns:
        assignments = dict(parity)
        assignments[i_occ] = first
        assignments[j_occ] = second
        assignments[a_virt] = third
        assignments[b_virt] = fourth
        terms.append(_pauli_term(assignments, coefficient, num_qubits))
    return SumOperator(terms)


def get_uccsd_operator_pool(num_electrons, num_qubits, spin=0):
    """Generate the CUDA-QX 0.6.0 UCCSD ADAPT operator pool."""

    if QuantumOperator is None:
        raise RuntimeError(
            "Canonical 'rocq' package is required to build the UCCSD operator pool."
        )
    excitations = get_uccsd_excitations(num_electrons, num_qubits, spin)
    singles_alpha, singles_beta, doubles_mixed, doubles_alpha, doubles_beta = (
        excitations
    )
    pool = [
        _single_excitation_generator(p_occ, q_virt, int(num_qubits))
        for p_occ, q_virt in (*singles_alpha, *singles_beta)
    ]
    pool.extend(
        _double_excitation_generator(*excitation, int(num_qubits))
        for excitation in (*doubles_mixed, *doubles_alpha, *doubles_beta)
    )
    return pool


def _register_size(qubits) -> int:
    if isinstance(qubits, (str, bytes)):
        raise ValueError("qubits must be a non-empty rocq quantum register.")
    try:
        size = len(qubits)
    except TypeError as exc:
        raise ValueError("qubits must be a non-empty rocq quantum register.") from exc
    return _positive_integer(size, "qubit register size")


def _validate_excitation_indices(indices, num_qubits: int):
    normalized = tuple(
        _nonnegative_integer(index, "excitation index") for index in indices
    )
    if len(set(normalized)) != len(normalized):
        raise ValueError("excitation indices must be distinct.")
    if any(index >= num_qubits for index in normalized):
        raise ValueError("excitation index is outside the qubit register.")
    return normalized


def _apply_generator(qubits, theta: float, generator):
    if rocq is None or iter_pauli_terms is None:
        raise RuntimeError(
            "Canonical 'rocq' package is required for UCCSD state preparation."
        )
    num_qubits = len(qubits)
    for coefficient, paulis in iter_pauli_terms(generator):
        coefficient = complex(coefficient)
        if (
            not math.isfinite(coefficient.real)
            or not math.isfinite(coefficient.imag)
            or abs(coefficient.imag) > 1.0e-12
        ):
            raise ValueError(
                "UCCSD excitation generators must have finite real coefficients."
            )
        word = ["I"] * num_qubits
        for pauli, qubit in paulis:
            word[int(qubit)] = str(pauli).upper()
        rocq.exp_pauli(theta * coefficient.real, qubits, "".join(word))


def single_excitation(qubits, theta, p_occ, q_virt):
    """Record one spin-preserving UCC single excitation."""

    num_qubits = _register_size(qubits)
    theta = _finite_real(theta, "theta")
    p_occ, q_virt = _validate_excitation_indices((p_occ, q_virt), num_qubits)
    if p_occ >= q_virt:
        raise ValueError("single excitation requires p_occ < q_virt.")
    _apply_generator(
        qubits,
        # CUDA-QX's direct gate kernel uses RZ(+/-0.5 * theta). With rocq's
        # exp(+i angle P) convention this is exp(-i theta * G / 2), where G is
        # the Hermitian pool generator returned above.
        -0.5 * theta,
        _single_excitation_generator(p_occ, q_virt, num_qubits),
    )


def double_excitation(qubits, theta, p_occ, q_occ, r_virt, s_virt):
    """Record one UCC double excitation.

    The orientation sign matches CUDA-QX's direct state-preparation kernel when
    either the occupied or virtual input pair, but not both, is reversed.
    """

    num_qubits = _register_size(qubits)
    theta = _finite_real(theta, "theta")
    p_occ, q_occ, r_virt, s_virt = _validate_excitation_indices(
        (p_occ, q_occ, r_virt, s_virt), num_qubits
    )
    i_occ, j_occ, a_virt, b_virt, orientation = _canonical_double_indices(
        p_occ, q_occ, r_virt, s_virt
    )
    if j_occ >= a_virt:
        raise ValueError(
            "double excitation requires occupied indices below virtual indices."
        )
    generator = _double_excitation_generator(i_occ, j_occ, a_virt, b_virt, num_qubits)
    # CUDA-QX decomposes each +/-0.125 double-generator term with an
    # RZ(+/-0.125 * theta), giving the same -theta/2 scale as the single kernel.
    _apply_generator(qubits, -0.5 * orientation * theta, generator)


def uccsd(qubits, thetas, num_electrons, spin=0):
    """Record the CUDA-QX-ordered UCCSD reference ansatz on ``qubits``."""

    num_qubits = _register_size(qubits)
    excitations = get_uccsd_excitations(num_electrons, num_qubits, spin)
    expected = sum(len(group) for group in excitations)
    if isinstance(thetas, (str, bytes)):
        raise ValueError("thetas must be a finite real parameter sequence.")
    try:
        parameters = [_finite_real(value, "UCCSD parameter") for value in thetas]
    except TypeError as exc:
        raise ValueError("thetas must be a finite real parameter sequence.") from exc
    if len(parameters) != expected:
        raise ValueError(f"thetas must contain exactly {expected} UCCSD parameters.")

    parameter_index = 0
    singles_alpha, singles_beta, doubles_mixed, doubles_alpha, doubles_beta = (
        excitations
    )
    for p_occ, q_virt in (*singles_alpha, *singles_beta):
        single_excitation(qubits, parameters[parameter_index], p_occ, q_virt)
        parameter_index += 1
    for excitation in (*doubles_mixed, *doubles_alpha, *doubles_beta):
        double_excitation(qubits, parameters[parameter_index], *excitation)
        parameter_index += 1


__all__ = [
    "double_excitation",
    "get_num_uccsd_parameters",
    "get_uccsd_excitations",
    "get_uccsd_operator_pool",
    "single_excitation",
    "uccsd",
]
