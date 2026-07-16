#ifndef ROCQ_COMPILER_PASSES_TD
#define ROCQ_COMPILER_PASSES_TD

include "mlir/Pass/PassBase.td"

def QuantumToQIR : Pass<"rocq-quantum-to-qir", "::mlir::ModuleOp"> {
  let summary = "Lower the rocQuantum dialect directly to LLVM-dialect QIR";
  let description = [{
    Converts canonical quantum operations to QIR runtime and QIS calls in the
    MLIR LLVM dialect.  Unsupported quantum operations must make the pass fail;
    they must not be silently erased or treated as successful lowering.
  }];
  let dependentDialects = [
    "::rocq::quantum::QuantumDialect",
    "::mlir::func::FuncDialect",
    "::mlir::LLVM::LLVMDialect"
  ];
  let constructor = "::rocq::compiler::createQuantumToQIRPass()";
}

#endif // ROCQ_COMPILER_PASSES_TD
