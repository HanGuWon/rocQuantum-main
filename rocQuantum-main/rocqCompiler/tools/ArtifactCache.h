#ifndef ROCQ_COMPILER_ARTIFACT_CACHE_H
#define ROCQ_COMPILER_ARTIFACT_CACHE_H

#include "rocqCompiler/CompilerArtifacts.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rocq::compiler::tools {

inline constexpr std::string_view kArtifactCacheSchema =
    "rocq-artifact-cache-v1";

/// SHA-256 over the configured compiler/lowering source inputs.  This prevents
/// a build containing changed lowering semantics from accepting stale entries
/// merely because its LLVM version and cache-envelope schema are unchanged.
std::string_view compilerBuildFingerprint();

struct ArtifactCacheKeyInputs {
    /// Borrowed because source buffers can be large; the referenced bytes must
    /// outlive the immediate computeArtifactCacheKey() call.
    std::string_view mlir_source;
    std::string profile = "qir-v2-static";
    std::uint64_t requested_num_qubits = 0;
    ArtifactKind kind = ArtifactKind::LlvmIr;
    unsigned optimization_level = 0;

    /// HostObject requires the normalized host triple returned by
    /// hostTargetTriple(). Target-neutral artifacts may leave this empty.
    std::string target_triple;
    std::string target_cpu = "generic";
    std::string target_features;
    std::string relocation_model = "pic";
};

/// Compute a lowercase content-addressed SHA-256 cache key. Each field is
/// length-prefixed, and the key includes the cache schema, exact LLVM version,
/// configured compiler build fingerprint, and every artifact input above.
std::string computeArtifactCacheKey(const ArtifactCacheKeyInputs& inputs);

/// Fail-closed, content-addressed artifact cache.
///
/// The cache directory is an explicit, trusted input and is not an
/// authenticity boundary. Entries use a self-validating envelope and atomic
/// same-directory commits. Missing entries return nullopt; malformed entries
/// and I/O errors throw instead of silently recompiling.
/// Stores require the cache filesystem to support same-filesystem hard links;
/// unsupported filesystems fail closed instead of falling back to a replacing
/// rename that could violate entry immutability under concurrent writers.
class ArtifactCache {
public:
    explicit ArtifactCache(std::filesystem::path root_directory);

    const std::filesystem::path& rootDirectory() const noexcept {
        return root_directory_;
    }

    std::filesystem::path entryPath(std::string_view key) const;

    std::optional<std::vector<std::uint8_t>> load(
        std::string_view key) const;

    /// Store a non-empty payload. Re-storing identical bytes is idempotent;
    /// different bytes for an existing key fail as a determinism violation.
    void store(std::string_view key,
               std::span<const std::uint8_t> payload) const;

private:
    std::filesystem::path root_directory_;
};

} // namespace rocq::compiler::tools

#endif // ROCQ_COMPILER_ARTIFACT_CACHE_H
