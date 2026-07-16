# Defines the gate operations available in the rocq programming model.
# When a kernel is being "recorded", these functions do not execute;
# they merely register themselves and their arguments in the kernel's context.

import math
from numbers import Real

from .kernel import _KernelBuildContext

# Gate functions
def h(target):
    _KernelBuildContext.add_gate("h", [target])

def x(target):
    _KernelBuildContext.add_gate("x", [target])

def y(target):
    _KernelBuildContext.add_gate("y", [target])

def z(target):
    _KernelBuildContext.add_gate("z", [target])

def s(target):
    _KernelBuildContext.add_gate("s", [target])

def sdg(target):
    _KernelBuildContext.add_gate("sdg", [target])

def t(target):
    _KernelBuildContext.add_gate("t", [target])

def tdg(target):
    _KernelBuildContext.add_gate("tdg", [target])

def tdag(target):
    _KernelBuildContext.add_gate("tdg", [target])

def rx(angle, target):
    _KernelBuildContext.add_gate("rx", [target], params={"theta": angle})

def ry(angle, target):
    _KernelBuildContext.add_gate("ry", [target], params={"theta": angle})

def rz(angle, target):
    _KernelBuildContext.add_gate("rz", [target], params={"phi": angle})

def p(angle, target):
    _KernelBuildContext.add_gate("p", [target], params={"phi": angle})

def phase(angle, target):
    _KernelBuildContext.add_gate("p", [target], params={"phi": angle})

def cnot(control, target):
    _KernelBuildContext.add_gate("cnot", [control, target])

def cx(control, target):
    _KernelBuildContext.add_gate("cnot", [control, target])

def cz(control, target):
    _KernelBuildContext.add_gate("cz", [control, target])

def swap(qubit_a, qubit_b):
    _KernelBuildContext.add_gate("swap", [qubit_a, qubit_b])

def mcx(controls, target):
    try:
        control_list = list(controls)
    except TypeError:
        control_list = [controls]
    _KernelBuildContext.add_gate("mcx", control_list + [target])

def ccx(control_a, control_b, target):
    _KernelBuildContext.add_gate("ccx", [control_a, control_b, target])

def toffoli(control_a, control_b, target):
    _KernelBuildContext.add_gate("ccx", [control_a, control_b, target])

def cswap(control, target_a, target_b):
    _KernelBuildContext.add_gate("cswap", [control, target_a, target_b])

def fredkin(control, target_a, target_b):
    _KernelBuildContext.add_gate("cswap", [control, target_a, target_b])

def crx(angle, control, target):
    _KernelBuildContext.add_gate("crx", [control, target], params={"theta": angle})

def cry(angle, control, target):
    _KernelBuildContext.add_gate("cry", [control, target], params={"theta": angle})

def crz(angle, control, target):
    _KernelBuildContext.add_gate("crz", [control, target], params={"phi": angle})

def cp(angle, control, target):
    _KernelBuildContext.add_gate("cp", [control, target], params={"phi": angle})

def cphase(angle, control, target):
    _KernelBuildContext.add_gate("cp", [control, target], params={"phi": angle})


def exp_pauli(angle, targets_or_word, word_or_target=None, *additional_targets):
    """Apply the CUDA-Q-style Pauli-word exponential ``exp(+i angle P)``.

    Both CUDA-Q call forms are accepted::

        exp_pauli(theta, qvector, "XYZ")
        exp_pauli(theta, "XZ", qvector[0], qvector[2])

    The operation is decomposed into the canonical H/RX/CNOT/RZ gate subset,
    with ``RZ(-2 * angle)`` matching CUDA-Q's positive-exponent convention.
    An all-identity word is omitted because it contributes only an unobservable
    global phase to sampling and expectation values.
    """

    if isinstance(angle, bool) or not isinstance(angle, Real):
        raise ValueError("exp_pauli angle must be a finite real number.")
    theta = float(angle)
    if not math.isfinite(theta):
        raise ValueError("exp_pauli angle must be finite.")

    if isinstance(targets_or_word, str):
        word = targets_or_word
        raw_targets = (word_or_target,) + tuple(additional_targets)
    else:
        word = word_or_target
        if additional_targets:
            raise TypeError(
                "exp_pauli register form accepts exactly one target register and one Pauli word."
            )
        try:
            raw_targets = tuple(targets_or_word)
        except TypeError as exc:
            raise TypeError(
                "exp_pauli targets must be a quantum register or explicit qubits."
            ) from exc

    if not isinstance(word, str) or not word:
        raise ValueError("exp_pauli Pauli word must be a non-empty I/X/Y/Z string.")
    normalized_word = word.upper()
    if any(pauli not in "IXYZ" for pauli in normalized_word):
        raise ValueError("exp_pauli Pauli word may contain only I, X, Y, and Z.")
    if len(raw_targets) != len(normalized_word):
        raise ValueError("exp_pauli Pauli-word length must match the number of targets.")

    context = _KernelBuildContext._active
    if context is None:
        raise RuntimeError("No active kernel context. exp_pauli called outside @rocq.kernel.")
    resolved_targets = [context._validate_gate_target(target) for target in raw_targets]
    if len(set(resolved_targets)) != len(resolved_targets):
        raise ValueError("exp_pauli target qubits must be distinct.")

    active = [
        (pauli, target)
        for pauli, target in zip(normalized_word, resolved_targets)
        if pauli != "I"
    ]
    if not active:
        return

    for pauli, target in active:
        if pauli == "X":
            h(target)
        elif pauli == "Y":
            rx(math.pi / 2.0, target)

    for (_, control), (_, target) in zip(active, active[1:]):
        cnot(control, target)
    rz(-2.0 * theta, active[-1][1])
    for (_, control), (_, target) in reversed(list(zip(active, active[1:]))):
        cnot(control, target)

    for pauli, target in reversed(active):
        if pauli == "X":
            h(target)
        elif pauli == "Y":
            rx(-math.pi / 2.0, target)
