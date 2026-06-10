from __future__ import annotations

import numpy as np


def _observable_terms(observable):
    try:
        from qiskit.quantum_info import Pauli, SparsePauliOp
    except ImportError as exc:
        raise ImportError("Qiskit quantum_info is required for native expectation estimation.") from exc

    if isinstance(observable, SparsePauliOp):
        return [(complex(coeff), pauli.to_label()) for coeff, pauli in zip(observable.coeffs, observable.paulis)]

    if isinstance(observable, Pauli):
        return [(1.0 + 0.0j, observable.to_label())]

    if isinstance(observable, str):
        return [(1.0 + 0.0j, observable)]

    if isinstance(observable, tuple) and len(observable) == 2 and isinstance(observable[0], str):
        label, coeff = observable
        return [(complex(coeff), label)]

    if isinstance(observable, list):
        return [(complex(coeff), str(label)) for label, coeff in observable]

    if hasattr(observable, "to_sparse_pauli_op"):
        return _observable_terms(observable.to_sparse_pauli_op())

    raise TypeError(f"Unsupported Qiskit observable type: {type(observable)!r}")


def _label_to_runtime_term(label: str, num_qubits: int):
    if len(label) > num_qubits:
        raise ValueError("Observable acts on more qubits than the circuit.")

    # Qiskit labels are big-endian strings; rocQuantum targets are qubit indices.
    padded = "I" * (num_qubits - len(label)) + label.upper()
    paulis = []
    targets = []
    for qubit, pauli in enumerate(reversed(padded)):
        if pauli == "I":
            continue
        if pauli not in {"X", "Y", "Z"}:
            raise ValueError("Pauli labels may only contain I, X, Y, or Z.")
        paulis.append(pauli)
        targets.append(qubit)
    return "".join(paulis), targets


def estimate_pauli_observable(runtime, observable, num_qubits: int) -> float:
    result = 0.0 + 0.0j
    for coeff, label in _observable_terms(observable):
        pauli_string, targets = _label_to_runtime_term(label, int(num_qubits))
        if not targets:
            result += coeff
        else:
            result += coeff * runtime.expectation_pauli_string(pauli_string, targets)

    real_result = np.real_if_close(result)
    if np.iscomplexobj(real_result):
        raise ValueError("Observable expectation has a non-negligible imaginary component.")
    return float(real_result)
