#ifndef ROCQ_COMPILER_ARTIFACTS_H
#define ROCQ_COMPILER_ARTIFACTS_H

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace llvm {
class Module;
}

namespace rocq::compiler {

enum class ArtifactKind {
    LlvmIr,
    LlvmBitcode,
    HostObject,
};

struct CompilerArtifactOptions {
    ArtifactKind kind = ArtifactKind::LlvmIr;
    unsigned optimization_level = 0;
};

struct CompilerArtifact {
    std::vector<std::uint8_t> bytes;

    /// Empty for target-neutral artifacts unless the input module already had
    /// a target triple. HostObject always reports the normalized host triple.
    std::string target_triple;
};

std::string_view artifactKindName(ArtifactKind kind);
std::string_view artifactFileExtension(ArtifactKind kind);
ArtifactKind parseArtifactKind(std::string_view name);

/// Return the normalized target triple used by HostObject emission.
std::string hostTargetTriple();

/// Optimize and serialize an owned LLVM module.
///
/// Ownership is explicit because LLVM optimization and object-code emission
/// mutate the module. The returned byte vector is binary-safe. HostObject is a
/// generic-CPU, position-independent relocatable object for the build host;
/// unresolved QIR/QIS symbols are intentionally left for a QIR runtime linker.
/// A module marked qir_profiles=base_profile may be serialized as LLVM IR or
/// bitcode only at -O0 because generic LLVM optimization can destroy its
/// required four-block shape. HostObject follows a separate native-linker
/// contract and permits -O0 through -O3.
CompilerArtifact emitCompilerArtifact(
    std::unique_ptr<::llvm::Module> module,
    const CompilerArtifactOptions& options = {});

} // namespace rocq::compiler

#endif // ROCQ_COMPILER_ARTIFACTS_H
