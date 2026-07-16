#ifndef ROCQ_COMPILER_QUANTUM_OPS_TD
#define ROCQ_COMPILER_QUANTUM_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Quantum_Dialect : Dialect {
  let name = "quantum";
  // Do not place generated declarations below rocq::mlir.  MLIR 22's ODS
  // output contains intentionally relative `mlir::Attribute` spellings,
  // which would otherwise resolve to rocq::mlir instead of ::mlir.
  let cppNamespace = "::rocq::quantum";
  let summary = "The canonical rocQuantum quantum-program dialect";
  let useDefaultTypePrinterParser = 1;
}

class Quantum_Op<string mnemonic, list<Trait> traits = []>
    : Op<Quantum_Dialect, mnemonic, traits>;

// A qubit is a handle into mutable quantum state.  It is intentionally not a
// pure SSA value: gate operations below carry read/write memory effects.
def Quantum_QubitType : TypeDef<Quantum_Dialect, "Qubit"> {
  let mnemonic = "qubit";
  let summary = "A handle to one logical qubit";
}

// Measurement results are resource handles as well.  Under the static QIR v2
// profiles they lower to opaque pointers whose integer payload is assigned by
// the compiler; they are not host booleans and must not be read implicitly.
def Quantum_ResultType : TypeDef<Quantum_Dialect, "Result"> {
  let mnemonic = "result";
  let summary = "A handle to one logical measurement result";
}

def QallocOp : Quantum_Op<"qalloc", [MemoryEffects<[MemAlloc, MemWrite]>]> {
  let summary = "Allocate a statically sized register of qubits";
  let arguments = (ins I64Attr:$size);
  let results = (outs Variadic<Quantum_QubitType>:$qubits);
}

class Quantum_StateOp<string mnemonic>
    : Quantum_Op<mnemonic, [MemoryEffects<[MemRead, MemWrite]>]>;

def HOp : Quantum_StateOp<"h"> {
  let arguments = (ins Quantum_QubitType:$target);
}
def XOp : Quantum_StateOp<"x"> {
  let arguments = (ins Quantum_QubitType:$target);
}
def YOp : Quantum_StateOp<"y"> {
  let arguments = (ins Quantum_QubitType:$target);
}
def ZOp : Quantum_StateOp<"z"> {
  let arguments = (ins Quantum_QubitType:$target);
}
def SOp : Quantum_StateOp<"s"> {
  let arguments = (ins Quantum_QubitType:$target);
}
def SdgOp : Quantum_StateOp<"sdg"> {
  let arguments = (ins Quantum_QubitType:$target);
}
def TOp : Quantum_StateOp<"t"> {
  let arguments = (ins Quantum_QubitType:$target);
}
def TdgOp : Quantum_StateOp<"tdg"> {
  let arguments = (ins Quantum_QubitType:$target);
}
def CnotOp : Quantum_StateOp<"cnot"> {
  let arguments = (ins Quantum_QubitType:$control,
                       Quantum_QubitType:$target);
}
def CzOp : Quantum_StateOp<"cz"> {
  let arguments = (ins Quantum_QubitType:$control,
                       Quantum_QubitType:$target);
}
def SwapOp : Quantum_StateOp<"swap"> {
  let arguments = (ins Quantum_QubitType:$lhs, Quantum_QubitType:$rhs);
}
def CcxOp : Quantum_StateOp<"ccx"> {
  let arguments = (ins Quantum_QubitType:$control0,
                       Quantum_QubitType:$control1,
                       Quantum_QubitType:$target);
}
def McxOp : Quantum_StateOp<"mcx"> {
  let arguments = (ins Variadic<Quantum_QubitType>:$qubits);
}
def CswapOp : Quantum_StateOp<"cswap"> {
  let arguments = (ins Quantum_QubitType:$control,
                       Quantum_QubitType:$lhs,
                       Quantum_QubitType:$rhs);
}
def RxOp : Quantum_StateOp<"rx"> {
  let arguments = (ins F64Attr:$angle, Quantum_QubitType:$target);
}
def RyOp : Quantum_StateOp<"ry"> {
  let arguments = (ins F64Attr:$angle, Quantum_QubitType:$target);
}
def RzOp : Quantum_StateOp<"rz"> {
  let arguments = (ins F64Attr:$angle, Quantum_QubitType:$target);
}
def POp : Quantum_StateOp<"p"> {
  let arguments = (ins F64Attr:$angle, Quantum_QubitType:$target);
}
def CrxOp : Quantum_StateOp<"crx"> {
  let arguments = (ins F64Attr:$angle,
                       Quantum_QubitType:$control,
                       Quantum_QubitType:$target);
}
def CryOp : Quantum_StateOp<"cry"> {
  let arguments = (ins F64Attr:$angle,
                       Quantum_QubitType:$control,
                       Quantum_QubitType:$target);
}
def CrzOp : Quantum_StateOp<"crz"> {
  let arguments = (ins F64Attr:$angle,
                       Quantum_QubitType:$control,
                       Quantum_QubitType:$target);
}
def CpOp : Quantum_StateOp<"cp"> {
  let arguments = (ins F64Attr:$angle,
                       Quantum_QubitType:$control,
                       Quantum_QubitType:$target);
}

// The register name is deliberately required.  A QIR Base Profile program
// must explicitly define its output and every output label must be non-empty
// and unique; the profile analysis enforces those semantic constraints.
def MzOp : Quantum_StateOp<"mz"> {
  let summary = "Measure one qubit in the computational basis";
  let arguments = (ins Quantum_QubitType:$target, StrAttr:$registerName);
  let results = (outs Quantum_ResultType:$result);
}

// This op is conservatively side-effecting.  Its regions are intentionally not
// marked pure; the direct QIR lowering must either lower both regions or fail.
def IfOp : Quantum_Op<"if"> {
  let summary = "Conditionally execute one of two quantum regions";
  let arguments = (ins I1:$condition);
  let regions = (region SizedRegion<1>:$thenRegion,
                        SizedRegion<1>:$elseRegion);
}

#endif // ROCQ_COMPILER_QUANTUM_OPS_TD
