#include "rocqCompiler/tools/ArtifactCache.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

#ifndef ROCQ_COMPILER_FINGERPRINT
#define ROCQ_COMPILER_FINGERPRINT "unconfigured-development-build"
#endif

namespace {

using rocq::compiler::ArtifactKind;
using rocq::compiler::tools::ArtifactCacheKeyInputs;
using rocq::compiler::tools::kArtifactCacheSchema;

::llvm::ArrayRef<std::uint8_t> asBytes(std::string_view value) {
    return {reinterpret_cast<const std::uint8_t*>(value.data()), value.size()};
}

void updateLength(::llvm::SHA256& hash, std::uint64_t length) {
    std::array<std::uint8_t, 8> encoded{};
    for (std::size_t index = 0; index < encoded.size(); ++index) {
        encoded[encoded.size() - index - 1] =
            static_cast<std::uint8_t>(length & 0xffU);
        length >>= 8U;
    }
    hash.update(encoded);
}

void updateField(::llvm::SHA256& hash,
                 std::string_view name,
                 std::string_view value) {
    updateLength(hash, name.size());
    hash.update(asBytes(name));
    updateLength(hash, value.size());
    hash.update(asBytes(value));
}

std::string sha256(std::span<const std::uint8_t> payload) {
    const auto digest = ::llvm::SHA256::hash(
        ::llvm::ArrayRef<std::uint8_t>(payload.data(), payload.size()));
    return ::llvm::toHex(::llvm::ArrayRef<std::uint8_t>(digest),
                         /*LowerCase=*/true);
}

bool isLowercaseSha256(std::string_view key) {
    return key.size() == 64 &&
           std::all_of(key.begin(), key.end(), [](unsigned char character) {
               return std::isdigit(character) != 0 ||
                      (character >= 'a' && character <= 'f');
           });
}

void validateKey(std::string_view key) {
    if (!isLowercaseSha256(key)) {
        throw std::invalid_argument(
            "artifact cache key must be exactly 64 lowercase hexadecimal characters");
    }
}

[[noreturn]] void throwCacheError(const std::filesystem::path& path,
                                  const std::string& message) {
    throw std::runtime_error(
        "artifact cache '" + path.string() + "': " + message);
}

::llvm::StringRef consumeLine(::llvm::StringRef& remaining,
                              const std::filesystem::path& path,
                              const char* field) {
    const auto newline = remaining.find('\n');
    if (newline == ::llvm::StringRef::npos) {
        throwCacheError(path, std::string("corrupt entry: missing ") + field);
    }
    const auto line = remaining.take_front(newline);
    remaining = remaining.drop_front(newline + 1);
    return line;
}

std::vector<std::uint8_t> parseEnvelope(
    ::llvm::StringRef envelope,
    std::string_view expected_key,
    const std::filesystem::path& path) {
    const auto schema = consumeLine(envelope, path, "schema line");
    if (schema != ::llvm::StringRef(kArtifactCacheSchema.data(),
                                    kArtifactCacheSchema.size())) {
        throwCacheError(path, "corrupt entry: unsupported cache schema");
    }

    const auto stored_key = consumeLine(envelope, path, "cache key line");
    if (stored_key != ::llvm::StringRef(expected_key.data(),
                                        expected_key.size())) {
        throwCacheError(path, "corrupt entry: embedded cache key mismatch");
    }

    const auto size_text = consumeLine(envelope, path, "payload size line");
    std::uint64_t declared_size = 0;
    if (size_text.getAsInteger(10, declared_size) ||
        size_text != std::to_string(declared_size) || declared_size == 0) {
        throwCacheError(path, "corrupt entry: invalid payload size");
    }
    if (declared_size > std::numeric_limits<std::size_t>::max()) {
        throwCacheError(path, "corrupt entry: payload size mismatch");
    }

    const auto stored_payload_hash =
        consumeLine(envelope, path, "payload SHA-256 line");
    // consumeLine above advanced `envelope`, so validate the size against the
    // actual payload after all header fields have been consumed.
    if (declared_size != envelope.size()) {
        throwCacheError(path, "corrupt entry: payload size mismatch");
    }
    if (!isLowercaseSha256(stored_payload_hash.str())) {
        throwCacheError(path, "corrupt entry: invalid payload SHA-256");
    }

    const auto* payload_begin =
        reinterpret_cast<const std::uint8_t*>(envelope.data());
    const std::span<const std::uint8_t> payload(
        payload_begin, envelope.size());
    if (sha256(payload) != stored_payload_hash.str()) {
        throwCacheError(path, "corrupt entry: payload SHA-256 mismatch");
    }
    return {payload.begin(), payload.end()};
}

void discardTemporaryFile(::llvm::sys::fs::TempFile& temporary) {
    if (auto error = temporary.discard()) {
        ::llvm::consumeError(std::move(error));
    }
}

class RemoveFileOnScopeExit {
public:
    explicit RemoveFileOnScopeExit(std::string path)
        : path_(std::move(path)) {}

    ~RemoveFileOnScopeExit() {
        std::error_code ignored = ::llvm::sys::fs::remove(path_);
        (void)ignored;
    }

    RemoveFileOnScopeExit(const RemoveFileOnScopeExit&) = delete;
    RemoveFileOnScopeExit& operator=(const RemoveFileOnScopeExit&) = delete;

private:
    std::string path_;
};

} // namespace

namespace rocq::compiler::tools {

std::string_view compilerBuildFingerprint() {
    return ROCQ_COMPILER_FINGERPRINT;
}

std::string computeArtifactCacheKey(const ArtifactCacheKeyInputs& inputs) {
    if (inputs.profile.empty()) {
        throw std::invalid_argument(
            "artifact cache profile must not be empty");
    }
    if (inputs.optimization_level > 3) {
        throw std::invalid_argument(
            "artifact cache optimization level must be between 0 and 3");
    }

    std::string normalized_triple;
    if (!inputs.target_triple.empty()) {
        normalized_triple = ::llvm::Triple::normalize(inputs.target_triple);
        if (normalized_triple.empty()) {
            throw std::invalid_argument(
                "artifact cache target triple could not be normalized");
        }
    }

    if (inputs.kind == ArtifactKind::HostObject) {
        const std::string host_triple = hostTargetTriple();
        if (normalized_triple.empty() || normalized_triple != host_triple) {
            throw std::invalid_argument(
                "host-object cache keys require the normalized host target triple '" +
                host_triple + "'");
        }
        if (inputs.target_cpu != "generic" ||
            !inputs.target_features.empty() ||
            inputs.relocation_model != "pic") {
            throw std::invalid_argument(
                "host-object cache keys require cpu=generic, empty features, and pic relocation");
        }
    }

    ::llvm::SHA256 hash;
    updateField(hash, "schema", kArtifactCacheSchema);
    updateField(hash, "llvm-version", LLVM_VERSION_STRING);
    updateField(hash, "compiler-fingerprint", compilerBuildFingerprint());
    updateField(hash, "mlir-source", inputs.mlir_source);
    updateField(hash, "profile", inputs.profile);
    updateField(hash,
                "requested-num-qubits",
                std::to_string(inputs.requested_num_qubits));
    updateField(hash, "artifact-kind", artifactKindName(inputs.kind));
    updateField(hash,
                "optimization-level",
                std::to_string(inputs.optimization_level));
    updateField(hash, "target-triple", normalized_triple);
    updateField(hash, "target-cpu", inputs.target_cpu);
    updateField(hash, "target-features", inputs.target_features);
    updateField(hash, "relocation-model", inputs.relocation_model);

    const auto digest = hash.final();
    return ::llvm::toHex(::llvm::ArrayRef<std::uint8_t>(digest),
                         /*LowerCase=*/true);
}

ArtifactCache::ArtifactCache(std::filesystem::path root_directory) {
    if (root_directory.empty()) {
        throw std::invalid_argument(
            "artifact cache root directory must not be empty");
    }
    std::error_code error;
    root_directory_ = std::filesystem::absolute(root_directory, error);
    if (error) {
        throw std::runtime_error(
            "could not resolve artifact cache root '" +
            root_directory.string() + "': " + error.message());
    }
}

std::filesystem::path ArtifactCache::entryPath(std::string_view key) const {
    validateKey(key);
    return root_directory_ / "v1" / std::string(key.substr(0, 2)) /
           (std::string(key.substr(2)) + ".cache");
}

std::optional<std::vector<std::uint8_t>> ArtifactCache::load(
    std::string_view key) const {
    const auto path = entryPath(key);
    auto buffer = ::llvm::MemoryBuffer::getFile(
        path.string(), /*IsText=*/false, /*RequiresNullTerminator=*/false);
    if (!buffer) {
        if (buffer.getError().default_error_condition() ==
            std::make_error_condition(
                std::errc::no_such_file_or_directory)) {
            return std::nullopt;
        }
        throwCacheError(path, "read failed: " + buffer.getError().message());
    }
    return parseEnvelope((*buffer)->getBuffer(), key, path);
}

void ArtifactCache::store(
    std::string_view key,
    std::span<const std::uint8_t> payload) const {
    validateKey(key);
    if (payload.empty()) {
        throw std::invalid_argument(
            "artifact cache refuses to store an empty payload");
    }

    if (const auto existing = load(key)) {
        if (std::equal(existing->begin(), existing->end(),
                       payload.begin(), payload.end())) {
            return;
        }
        throw std::runtime_error(
            "artifact cache determinism violation: key '" +
            std::string(key) + "' already maps to different bytes");
    }

    const auto final_path = entryPath(key);
    std::error_code directory_error;
    std::filesystem::create_directories(
        final_path.parent_path(), directory_error);
    if (directory_error) {
        throwCacheError(
            final_path,
            "could not create cache directory: " + directory_error.message());
    }

    const std::string header =
        std::string(kArtifactCacheSchema) + "\n" + std::string(key) + "\n" +
        std::to_string(payload.size()) + "\n" + sha256(payload) + "\n";
    auto temporary_or_error = ::llvm::sys::fs::TempFile::create(
        final_path.string() + ".tmp-%%%%%%",
        ::llvm::sys::fs::owner_read | ::llvm::sys::fs::owner_write,
        ::llvm::sys::fs::OF_None);
    if (!temporary_or_error) {
        throwCacheError(
            final_path,
            "could not create atomic temporary file: " +
                ::llvm::toString(temporary_or_error.takeError()));
    }
    auto temporary = std::move(*temporary_or_error);

    std::error_code write_error;
    {
        ::llvm::raw_fd_ostream output(
            temporary.FD, /*shouldClose=*/false);
        output << header;
        output.write(
            reinterpret_cast<const char*>(payload.data()), payload.size());
        output.flush();
        write_error = output.error();
        output.clear_error();
    }
    if (write_error) {
        discardTemporaryFile(temporary);
        throwCacheError(final_path,
                        "temporary write failed: " + write_error.message());
    }

    const std::string temporary_path = temporary.TmpName;
    if (auto error = temporary.keep()) {
        throwCacheError(final_path,
                        "could not close temporary cache entry: " +
                            ::llvm::toString(std::move(error)));
    }
    RemoveFileOnScopeExit remove_temporary(temporary_path);

    // A hard-link commit is atomic and refuses to replace an existing path.
    // This makes cache entries immutable even when independent compiler
    // processes race after observing the same initial miss.
    const std::error_code commit_error = ::llvm::sys::fs::create_hard_link(
        temporary_path, final_path.string());
    if (commit_error) {
        if (commit_error.default_error_condition() !=
            std::make_error_condition(std::errc::file_exists)) {
            throwCacheError(final_path,
                            "atomic no-replace commit failed: " +
                                commit_error.message());
        }
        const auto winner = load(key);
        if (winner && std::equal(winner->begin(), winner->end(),
                                 payload.begin(), payload.end())) {
            return;
        }
        throw std::runtime_error(
            "artifact cache determinism violation: concurrent writer for key '" +
            std::string(key) + "' committed different bytes");
    }

    const auto committed = load(key);
    if (!committed ||
        !std::equal(committed->begin(), committed->end(),
                    payload.begin(), payload.end())) {
        throwCacheError(
            final_path,
            "atomic commit verification returned different artifact bytes");
    }
}

} // namespace rocq::compiler::tools
