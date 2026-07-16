#ifndef ROCQ_COMPILER_QIR_PROFILE_H
#define ROCQ_COMPILER_QIR_PROFILE_H

#include "rocqCompiler/passes/QuantumToQIRPass.h"

#include <string>

namespace llvm {
class Module;
}

namespace rocq::compiler {

/// Parse the stable public profile spellings accepted by MLIRCompiler and the
/// rocq-translate command-line tool.
bool parseQIREmissionProfile(const std::string& name,
                             QIREmissionProfile& profile);

/// Finish profile-specific LLVM IR construction, attach the exact QIR v2
/// resource metadata, and verify both LLVM well-formedness and the deliberately
/// narrow profile contract.  Returns false with a diagnostic on any mismatch.
bool finalizeAndVerifyQIRModule(::llvm::Module& module,
                                const QIRModuleInfo& info,
                                QIREmissionProfile profile,
                                std::string& diagnostic);

} // namespace rocq::compiler

#endif // ROCQ_COMPILER_QIR_PROFILE_H
