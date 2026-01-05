# Defines the gate operations available in the rocq programming model.
# When a kernel is being "recorded", these functions do not execute;
# they merely register themselves and their arguments in the kernel's context.

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

def rx(angle, target):
    _KernelBuildContext.add_gate("rx", [target], params={"theta": angle})

def ry(angle, target):
    _KernelBuildContext.add_gate("ry", [target], params={"theta": angle})

def rz(angle, target):
    _KernelBuildContext.add_gate("rz", [target], params={"phi": angle})

def cnot(control, target):
    _KernelBuildContext.add_gate("cnot", [control, target])

def cx(control, target):
    _KernelBuildContext.add_gate("cnot", [control, target])

def cz(control, target):
    _KernelBuildContext.add_gate("cz", [control, target])

def swap(qubit_a, qubit_b):
    _KernelBuildContext.add_gate("swap", [qubit_a, qubit_b])

def crx(angle, control, target):
    _KernelBuildContext.add_gate("crx", [control, target], params={"theta": angle})

def cry(angle, control, target):
    _KernelBuildContext.add_gate("cry", [control, target], params={"theta": angle})

def crz(angle, control, target):
    _KernelBuildContext.add_gate("crz", [control, target], params={"phi": angle})
