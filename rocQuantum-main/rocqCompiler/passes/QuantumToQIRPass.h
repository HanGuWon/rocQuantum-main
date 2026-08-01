#ifndef ROCQ_COMPILER_QUANTUM_TO_QIR_PASS_H
#define ROCQ_COMPILER_QUANTUM_TO_QIR_PASS_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
class Pass;
}

namespace rocq::compiler {

enum class QIREmissionProfile {
    StaticCustom,
    Base,
};

struct QIRModuleInfo {
    std::string entry_point;
    std::uint64_t required_qubits = 0;
    std::uint64_t required_results = 0;
    std::vector<std::string> result_labels;
};

/// Validate the deliberately narrow, static-circuit QIR v2 source contract.
/// Diagnostics are returned to API callers as well as emitted on the MLIR op.
::mlir::LogicalResult analyzeQuantumModuleForQIR(::mlir::ModuleOp module,
                                                 QIRModuleInfo& info,
                                                 QIREmissionProfile profile,
                                                 std::string* diagnostic = nullptr);

inline ::mlir::LogicalResult analyzeQuantumModuleForQIR(
    ::mlir::ModuleOp module,
    QIRModuleInfo& info,
    std::string* diagnostic = nullptr) {
    return analyzeQuantumModuleForQIR(
        module, info, QIREmissionProfile::StaticCustom, diagnostic);
}

std::unique_ptr<::mlir::Pass> createQuantumToQIRPass();
std::unique_ptr<::mlir::Pass> createQuantumToQIRPass(QIREmissionProfile profile);

} // namespace rocq::compiler

#endif // ROCQ_COMPILER_QUANTUM_TO_QIR_PASS_H
