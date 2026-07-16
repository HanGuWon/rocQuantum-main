#ifndef ROCQ_COMPILER_PASSES_H
#define ROCQ_COMPILER_PASSES_H

#include "QuantumDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

#include <memory>

namespace rocq::compiler {

std::unique_ptr<::mlir::Pass> createQuantumToQIRPass();

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "passes/Passes.h.inc"

} // namespace rocq::compiler

#endif // ROCQ_COMPILER_PASSES_H
