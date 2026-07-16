#ifndef ROCQ_COMPILER_PIPELINES_H
#define ROCQ_COMPILER_PIPELINES_H

namespace mlir {
class OpPassManager;
}

namespace rocq::compiler {

enum class QIREmissionProfile;

/// Build the canonical, GPU-independent static-QIR lowering pipeline.
///
/// The generic cleanup passes deliberately respect the memory effects on
/// quantum operations.  They must not be described as circuit optimization:
/// quantum-specific cancellation and commutation require dedicated passes.
void buildQIRPipeline(::mlir::OpPassManager& pass_manager,
                      QIREmissionProfile profile);

/// Register the explicit `rocq-qir-static-pipeline` and
/// `rocq-qir-base-pipeline` spellings for MLIR's pass-pipeline parser.  The
/// profile is intentionally part of the name: silently treating Base Profile
/// measurement operations as static-custom QIR would be a correctness bug.
/// Calling this function repeatedly is safe.
void registerRocqCompilerPipelines();

} // namespace rocq::compiler

#endif // ROCQ_COMPILER_PIPELINES_H
