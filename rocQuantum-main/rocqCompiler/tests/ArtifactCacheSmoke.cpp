#include "rocqCompiler/tools/ArtifactCache.h"

#include <algorithm>
#include <atomic>
#include <barrier>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"

namespace {

using rocq::compiler::ArtifactKind;
using rocq::compiler::tools::ArtifactCache;
using rocq::compiler::tools::ArtifactCacheKeyInputs;

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        const auto prefix =
            std::filesystem::temp_directory_path() /
            "rocq-artifact-cache-smoke";
        ::llvm::SmallString<128> created;
        if (const auto error = ::llvm::sys::fs::createUniqueDirectory(
                prefix.string(), created)) {
            throw std::runtime_error(
                "could not create cache smoke-test directory: " +
                error.message());
        }
        path_ = std::string(created);
    }

    ~TemporaryDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }

    const std::filesystem::path& path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
};

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Callback>
void requireThrowsContaining(Callback&& callback, const std::string& needle) {
    try {
        callback();
    } catch (const std::exception& error) {
        if (std::string(error.what()).find(needle) != std::string::npos) {
            return;
        }
        throw std::runtime_error(
            "exception did not contain '" + needle + "': " + error.what());
    }
    throw std::runtime_error("expected exception containing: " + needle);
}

bool isLowercaseHex(const std::string& value) {
    return value.size() == 64 &&
           std::all_of(value.begin(), value.end(), [](unsigned char character) {
               return (character >= '0' && character <= '9') ||
                      (character >= 'a' && character <= 'f');
           });
}

} // namespace

int main() {
    try {
        ArtifactCacheKeyInputs inputs;
        inputs.mlir_source = "module { func.func @cache_fixture() { return } }";
        inputs.profile = "qir-v2-static";
        inputs.requested_num_qubits = 1;
        inputs.kind = ArtifactKind::LlvmBitcode;
        inputs.optimization_level = 2;

        const auto key = rocq::compiler::tools::computeArtifactCacheKey(inputs);
        const auto repeated_key =
            rocq::compiler::tools::computeArtifactCacheKey(inputs);
        require(key == repeated_key && isLowercaseHex(key),
                "artifact cache key was not a deterministic lowercase SHA-256");
        require(isLowercaseHex(std::string(
                    rocq::compiler::tools::compilerBuildFingerprint())),
                "configured compiler fingerprint was not a lowercase SHA-256");

        auto changed_inputs = inputs;
        changed_inputs.optimization_level = 3;
        require(key != rocq::compiler::tools::computeArtifactCacheKey(
                           changed_inputs),
                "optimization level was absent from the cache key");
        changed_inputs = inputs;
        changed_inputs.mlir_source =
            "module { func.func @different() { return } }";
        require(key != rocq::compiler::tools::computeArtifactCacheKey(
                           changed_inputs),
                "MLIR source bytes were absent from the cache key");

        ArtifactCacheKeyInputs object_inputs = inputs;
        object_inputs.kind = ArtifactKind::HostObject;
        const std::string host_triple = rocq::compiler::hostTargetTriple();
        object_inputs.target_triple = host_triple;
        require(isLowercaseHex(
                    rocq::compiler::tools::computeArtifactCacheKey(
                        object_inputs)),
                "host-object cache key construction failed");
        object_inputs.target_triple = {};
        requireThrowsContaining(
            [&] {
                (void)rocq::compiler::tools::computeArtifactCacheKey(
                    object_inputs);
            },
            "host target triple");

        TemporaryDirectory temporary;
        ArtifactCache cache(temporary.path());
        require(!cache.load(key).has_value(),
                "new cache unexpectedly contained an artifact");

        const std::vector<std::uint8_t> payload = {
            'B', 'C', 0xc0, 0xde, 0x00, 0x01, 0x00, 0xff};
        cache.store(key, payload);
        const auto loaded = cache.load(key);
        require(loaded && *loaded == payload,
                "cache did not round-trip binary artifact bytes");
        cache.store(key, payload);
        require(std::filesystem::is_regular_file(cache.entryPath(key)),
                "cache entry was not committed as a regular file");

        const std::vector<std::uint8_t> different_payload = {1, 2, 3};
        requireThrowsContaining(
            [&] { cache.store(key, different_payload); },
            "determinism violation");
        requireThrowsContaining(
            [&] { cache.store(key, std::span<const std::uint8_t>{}); },
            "empty payload");
        requireThrowsContaining(
            [&] { (void)cache.load("../../not-a-cache-key"); },
            "64 lowercase hexadecimal");

        auto concurrent_inputs = inputs;
        concurrent_inputs.mlir_source = "concurrent-identical";
        const auto concurrent_identical_key =
            rocq::compiler::tools::computeArtifactCacheKey(concurrent_inputs);
        std::barrier identical_start(9);
        std::mutex errors_mutex;
        std::vector<std::string> errors;
        std::vector<std::thread> identical_writers;
        for (unsigned index = 0; index < 8; ++index) {
            identical_writers.emplace_back([&] {
                identical_start.arrive_and_wait();
                try {
                    cache.store(concurrent_identical_key, payload);
                } catch (const std::exception& error) {
                    std::lock_guard lock(errors_mutex);
                    errors.push_back(error.what());
                }
            });
        }
        identical_start.arrive_and_wait();
        for (auto& writer : identical_writers) {
            writer.join();
        }
        require(errors.empty(),
                "concurrent identical cache writers did not remain idempotent");
        require(cache.load(concurrent_identical_key) == payload,
                "concurrent identical writers committed the wrong bytes");

        concurrent_inputs.mlir_source = "concurrent-different";
        const auto concurrent_different_key =
            rocq::compiler::tools::computeArtifactCacheKey(concurrent_inputs);
        std::barrier different_start(3);
        std::atomic<unsigned> successful_writers = 0;
        errors.clear();
        auto write_concurrently = [&](const std::vector<std::uint8_t>& bytes) {
            different_start.arrive_and_wait();
            try {
                cache.store(concurrent_different_key, bytes);
                ++successful_writers;
            } catch (const std::exception& error) {
                std::lock_guard lock(errors_mutex);
                errors.push_back(error.what());
            }
        };
        std::thread first_writer(write_concurrently, std::cref(payload));
        std::thread second_writer(
            write_concurrently, std::cref(different_payload));
        different_start.arrive_and_wait();
        first_writer.join();
        second_writer.join();
        require(successful_writers == 1 && errors.size() == 1 &&
                    errors.front().find("determinism violation") !=
                        std::string::npos,
                "concurrent different cache writers did not fail closed");
        const auto concurrent_winner = cache.load(concurrent_different_key);
        require(concurrent_winner &&
                    (*concurrent_winner == payload ||
                     *concurrent_winner == different_payload),
                "concurrent cache commit did not preserve either complete payload");

        {
            std::ofstream corrupt(
                cache.entryPath(key),
                std::ios::binary | std::ios::trunc);
            corrupt << "corrupt";
            if (!corrupt) {
                throw std::runtime_error(
                    "could not corrupt cache entry for fail-closed test");
            }
        }
        requireThrowsContaining(
            [&] { (void)cache.load(key); }, "corrupt entry");

        std::cout << "rocq artifact cache smoke tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
