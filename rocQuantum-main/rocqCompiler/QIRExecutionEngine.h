#ifndef ROCQ_COMPILER_QIR_EXECUTION_ENGINE_H
#define ROCQ_COMPILER_QIR_EXECUTION_ENGINE_H

#include "rocqCompiler/QuantumBackend.h"
#include "rocqCompiler/passes/QuantumToQIRPass.h"

#include <complex>
#include <vector>

#include "mlir/IR/BuiltinOps.h"

namespace rocq::compiler {

/// Execute an already-lowered, measurement-free qir-v2-static module through
/// MLIR's in-process LLVM ORC JIT.
///
/// The QIR entry point is required to be `void ()`.  Static QIR qubit handles
/// are decoded by the registered QIS callbacks and forwarded to the supplied
/// QuantumBackend.  Base/adaptive QIR deliberately has no entry through this
/// class until its result and control-flow runtime contracts are implemented.
class QIRExecutionEngine final {
public:
    static std::vector<std::complex<double>> executeStatic(
        ::mlir::ModuleOp lowered_module,
        const QIRModuleInfo& info,
        rocq::QuantumBackend& backend);
};

} // namespace rocq::compiler

#endif // ROCQ_COMPILER_QIR_EXECUTION_ENGINE_H
