#include "rocqCompiler/passes/Pipelines.h"

#include "rocqCompiler/passes/QuantumToQIRPass.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace rocq::compiler {

void buildQIRPipeline(::mlir::OpPassManager& pass_manager,
                      QIREmissionProfile profile) {
    // Quantum operations are memory-effecting, so these generic passes may
    // simplify their classical operands and surrounding IR but cannot erase
    // or merge gates as if they were pure expressions.
    pass_manager.addNestedPass<::mlir::func::FuncOp>(
        ::mlir::createCanonicalizerPass());
    pass_manager.addNestedPass<::mlir::func::FuncOp>(::mlir::createCSEPass());

    pass_manager.addPass(createQuantumToQIRPass(profile));

    // The direct lowering can introduce duplicate LLVM constants and helper
    // declarations.  Run target-independent cleanup before LLVM translation.
    pass_manager.addPass(::mlir::createCanonicalizerPass());
    pass_manager.addPass(::mlir::createCSEPass());
}

void registerRocqCompilerPipelines() {
    static ::mlir::PassPipelineRegistration<> static_registration(
        "rocq-qir-static-pipeline",
        "Canonicalize a static-custom rocQuantum module and lower it to LLVM-dialect QIR",
        [](::mlir::OpPassManager& pass_manager) {
            buildQIRPipeline(pass_manager, QIREmissionProfile::StaticCustom);
        });
    static ::mlir::PassPipelineRegistration<> base_registration(
        "rocq-qir-base-pipeline",
        "Canonicalize a QIR v2 Base Profile rocQuantum module and lower it to LLVM-dialect QIR",
        [](::mlir::OpPassManager& pass_manager) {
            buildQIRPipeline(pass_manager, QIREmissionProfile::Base);
        });
    (void)static_registration;
    (void)base_registration;
}

} // namespace rocq::compiler
