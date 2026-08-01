#include "rocqCompiler/QuantumDialect.h"
#include "rocqCompiler/passes/Passes.h"
#include "rocqCompiler/passes/Pipelines.h"
#include "rocqCompiler/passes/QuantumToQIRPass.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char** argv) {
    rocq::compiler::registerRocqCompilerPasses();
    rocq::compiler::registerRocqCompilerPipelines();

    mlir::DialectRegistry registry;
    registry.insert<rocq::quantum::QuantumDialect,
                    mlir::func::FuncDialect,
                    mlir::LLVM::LLVMDialect>();
    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "rocQuantum MLIR optimizer\n", registry));
}
