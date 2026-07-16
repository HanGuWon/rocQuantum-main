#ifndef ROCQ_COMPILER_SIMULATOR_OPS_TD
#define ROCQ_COMPILER_SIMULATOR_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Simulator_Dialect : Dialect {
  let name = "sim";
  let cppNamespace = "::rocq::sim";
  let summary = "Experimental simulator-call intermediate dialect";
}

class Sim_Op<string mnemonic, list<Trait> traits = []>
    : Op<Simulator_Dialect, mnemonic, traits>;

// Avoid OpaqueType: the registered quantum dialect parses !quantum.qubit into
// this concrete C++ type.  QuantumTypes.h is included by SimulatorOps.h.
def Simulator_QuantumQubitType : Type<
    CPred<"::llvm::isa<::rocq::quantum::QubitType>($_self)">,
    "quantum qubit",
    "::rocq::quantum::QubitType">;

def ApplyGateOp : Sim_Op<"apply_gate",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Apply a named gate through a simulator backend";
  let arguments = (ins StrAttr:$gate_name,
                       Variadic<Simulator_QuantumQubitType>:$targets);
}

def ApplyParamGateOp : Sim_Op<"apply_param_gate",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Apply a named parametrized gate through a simulator backend";
  let arguments = (ins StrAttr:$gate_name,
                       F64Attr:$parameter,
                       Variadic<Simulator_QuantumQubitType>:$targets);
}

#endif // ROCQ_COMPILER_SIMULATOR_OPS_TD
