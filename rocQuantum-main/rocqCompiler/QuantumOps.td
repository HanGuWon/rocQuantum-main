#ifndef QUANTUM_OPS_TD
#define QUANTUM_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Quantum_Dialect : Dialect {
  let name = "quantum";
  let cppNamespace = "rocq::mlir::quantum";
  let summary = "A dialect for quantum computing.";
}

// Base class for Quantum dialect operations
class Quantum_Op<string mnemonic, list<Trait> traits = []> :
    Op<Quantum_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

def QubitType : DialectType<Quantum_Dialect, "qubit", "a single qubit">;

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

def QallocOp : Quantum_Op<"qalloc", []> {
  let summary = "Allocate a register of qubits";
  let arguments = (ins I64Attr:$size);
  let results = (outs Variadic<QubitType>:$qubits);
}

def HOp : Quantum_Op<"h", [NoSideEffect]> {
  let arguments = (ins QubitType:$target);
}
def XOp : Quantum_Op<"x", [NoSideEffect]> {
  let arguments = (ins QubitType:$target);
}
def YOp : Quantum_Op<"y", [NoSideEffect]> {
  let arguments = (ins QubitType:$target);
}
def ZOp : Quantum_Op<"z", [NoSideEffect]> {
  let arguments = (ins QubitType:$target);
}
def CnotOp : Quantum_Op<"cnot", [NoSideEffect]> {
  let arguments = (ins QubitType:$control, QubitType:$target);
}
def RxOp : Quantum_Op<"rx", [NoSideEffect]> {
  let arguments = (ins F64Attr:$angle, QubitType:$target);
}
def RyOp : Quantum_Op<"ry", [NoSideEffect]> {
  let arguments = (ins F64Attr:$angle, QubitType:$target);
}
def RzOp : Quantum_Op<"rz", [NoSideEffect]> {
  let arguments = (ins F64Attr:$angle, QubitType:$target);
}

// New conditional operation
def IfOp : Quantum_Op<"if", [RecursiveSideEffects]> {
    let summary = "A conditional operation with then and else regions.";
    let arguments = (ins I1:$condition);
    let regions = (region SizedRegion<1>:$thenRegion,
                   region SizedRegion<1>:$elseRegion);
}

#endif // QUANTUM_OPS_TD
