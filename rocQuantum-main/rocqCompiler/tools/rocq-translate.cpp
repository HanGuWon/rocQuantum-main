#include "rocqCompiler/MLIRCompiler.h"

#include "rocqCompiler/CompilerArtifacts.h"
#include "rocqCompiler/QIRProfile.h"
#include "rocqCompiler/tools/ArtifactCache.h"

#include <charconv>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#ifndef ROCQ_VERSION
#define ROCQ_VERSION "unknown"
#endif

namespace {

using rocq::compiler::ArtifactKind;
using rocq::compiler::CompilerArtifactOptions;
using rocq::compiler::tools::ArtifactCache;
using rocq::compiler::tools::ArtifactCacheKeyInputs;

class UsageError final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct CommandLine {
    std::string profile = "qir-v2-static";
    std::string input_path;
    std::string output_path;
    std::optional<std::filesystem::path> cache_directory;
    ArtifactKind kind = ArtifactKind::LlvmIr;
    unsigned optimization_level = 0;
    unsigned num_qubits = 0;
    bool verbose = false;
};

void printUsage(std::ostream& output, const char* program) {
    output
        << "usage: " << program << " [options] <input.mlir|->\n"
        << "\n"
        << "Lower rocQuantum MLIR to verified QIR compiler artifacts.\n"
        << "\n"
        << "options:\n"
        << "  --profile NAME       qir-v2-static (default) or qir-v2-base\n"
        << "  --num-qubits N       require the source allocation to contain N qubits\n"
        << "  --emit FORMAT        llvm-ir (default), llvm-bc, or object\n"
        << "  -O0|-O1|-O2|-O3      LLVM optimization level (default: -O0)\n"
        << "  -o PATH              output path; LLVM IR defaults to stdout\n"
        << "  --cache-dir PATH     content-addressed artifact cache directory\n"
        << "  --verbose            report cache hits and misses to stderr\n"
        << "  --version            print compiler and LLVM versions\n"
        << "  -h, --help           show this help\n"
        << "\n"
        << "qir-v2-base LLVM IR/bitcode requires -O0 because generic LLVM\n"
        << "optimization does not preserve its four-block profile contract.\n";
}

std::string requireValue(int& index,
                         int argc,
                         char** argv,
                         std::string_view option) {
    if (++index >= argc) {
        throw UsageError("option '" + std::string(option) +
                         "' requires a value");
    }
    return argv[index];
}

unsigned parseUnsigned(std::string_view text, std::string_view option) {
    if (text.empty()) {
        throw UsageError("option '" + std::string(option) +
                         "' requires a non-empty unsigned integer");
    }
    unsigned value = 0;
    const auto [end, error] = std::from_chars(
        text.data(), text.data() + text.size(), value, 10);
    if (error != std::errc{} || end != text.data() + text.size()) {
        throw UsageError("option '" + std::string(option) +
                         "' requires an unsigned integer, got '" +
                         std::string(text) + "'");
    }
    return value;
}

void setOnce(bool& seen, std::string_view option) {
    if (seen) {
        throw UsageError("option '" + std::string(option) +
                         "' was specified more than once");
    }
    seen = true;
}

CommandLine parseCommandLine(int argc, char** argv) {
    CommandLine command;
    bool profile_seen = false;
    bool qubits_seen = false;
    bool emit_seen = false;
    bool optimization_seen = false;
    bool output_seen = false;
    bool cache_seen = false;
    bool positional_only = false;

    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (!positional_only && argument == "--") {
            positional_only = true;
            continue;
        }
        if (!positional_only && (argument == "--help" || argument == "-h")) {
            printUsage(std::cout, argv[0]);
            std::exit(0);
        }
        if (!positional_only && argument == "--version") {
            std::cout << "rocq-translate " << ROCQ_VERSION << " (LLVM "
                      << LLVM_VERSION_STRING << ")\n";
            std::exit(0);
        }
        if (!positional_only && argument == "--verbose") {
            if (command.verbose) {
                throw UsageError("option '--verbose' was specified more than once");
            }
            command.verbose = true;
            continue;
        }

        auto optionValue = [&](std::string_view long_name)
            -> std::optional<std::string> {
            if (argument == long_name) {
                return requireValue(index, argc, argv, long_name);
            }
            const std::string prefix = std::string(long_name) + "=";
            if (argument.rfind(prefix, 0) == 0) {
                return argument.substr(prefix.size());
            }
            return std::nullopt;
        };

        if (!positional_only) {
            if (auto value = optionValue("--profile")) {
                setOnce(profile_seen, "--profile");
                if (value->empty()) {
                    throw UsageError("option '--profile' requires a non-empty value");
                }
                command.profile = std::move(*value);
                continue;
            }
            if (auto value = optionValue("--num-qubits")) {
                setOnce(qubits_seen, "--num-qubits");
                command.num_qubits = parseUnsigned(*value, "--num-qubits");
                continue;
            }
            if (auto value = optionValue("--emit")) {
                setOnce(emit_seen, "--emit");
                try {
                    command.kind = rocq::compiler::parseArtifactKind(*value);
                } catch (const std::invalid_argument& error) {
                    throw UsageError(error.what());
                }
                continue;
            }
            if (auto value = optionValue("--opt-level")) {
                setOnce(optimization_seen, "--opt-level");
                command.optimization_level =
                    parseUnsigned(*value, "--opt-level");
                if (command.optimization_level > 3) {
                    throw UsageError("optimization level must be between 0 and 3");
                }
                continue;
            }
            if (auto value = optionValue("--output")) {
                setOnce(output_seen, "--output");
                if (value->empty()) {
                    throw UsageError("option '--output' requires a non-empty path");
                }
                command.output_path = std::move(*value);
                continue;
            }
            if (auto value = optionValue("--cache-dir")) {
                setOnce(cache_seen, "--cache-dir");
                if (value->empty()) {
                    throw UsageError("option '--cache-dir' requires a non-empty path");
                }
                command.cache_directory = std::filesystem::path(*value);
                continue;
            }
            if (argument == "-o") {
                setOnce(output_seen, "-o");
                command.output_path = requireValue(index, argc, argv, "-o");
                if (command.output_path.empty()) {
                    throw UsageError("option '-o' requires a non-empty path");
                }
                continue;
            }
            if (argument.size() == 3 && argument[0] == '-' &&
                argument[1] == 'O' && argument[2] >= '0' &&
                argument[2] <= '3') {
                setOnce(optimization_seen, "-O");
                command.optimization_level =
                    static_cast<unsigned>(argument[2] - '0');
                continue;
            }
        }

        if (!positional_only && argument != "-" && !argument.empty() &&
            argument.front() == '-') {
            throw UsageError("unknown option '" + argument + "'");
        }
        if (!command.input_path.empty()) {
            throw UsageError("exactly one input path is required");
        }
        command.input_path = argument;
    }

    if (command.input_path.empty()) {
        throw UsageError("an input path is required");
    }
    if (command.kind != ArtifactKind::LlvmIr &&
        (command.output_path.empty() || command.output_path == "-")) {
        throw UsageError("llvm-bc and object output require -o with a file path");
    }

    rocq::compiler::QIREmissionProfile parsed_profile;
    if (!rocq::compiler::parseQIREmissionProfile(
            command.profile, parsed_profile)) {
        throw UsageError(
            "unknown QIR profile '" + command.profile +
            "'; expected qir-v2-static or qir-v2-base");
    }
    if (parsed_profile == rocq::compiler::QIREmissionProfile::Base &&
        command.kind != ArtifactKind::HostObject &&
        command.optimization_level != 0) {
        throw UsageError(
            "qir-v2-base LLVM IR and bitcode require -O0; generic LLVM "
            "optimization does not preserve the Base Profile four-block contract");
    }
    return command;
}

std::string readInput(const std::string& path) {
    std::istream* input = &std::cin;
    std::ifstream file;
    if (path != "-") {
        file.open(path, std::ios::binary);
        if (!file) {
            throw std::runtime_error("could not open input file '" + path + "'");
        }
        input = &file;
    }

    std::string contents(
        (std::istreambuf_iterator<char>(*input)),
        std::istreambuf_iterator<char>());
    if (input->bad()) {
        throw std::runtime_error("failed while reading input '" + path + "'");
    }
    return contents;
}

void discardTemporaryFile(::llvm::sys::fs::TempFile& temporary) {
    if (auto error = temporary.discard()) {
        ::llvm::consumeError(std::move(error));
    }
}

void writeFileAtomically(const std::filesystem::path& path,
                         std::span<const std::uint8_t> bytes) {
    auto temporary_or_error = ::llvm::sys::fs::TempFile::create(
        path.string() + ".tmp-%%%%%%",
        ::llvm::sys::fs::owner_read | ::llvm::sys::fs::owner_write,
        ::llvm::sys::fs::OF_None);
    if (!temporary_or_error) {
        throw std::runtime_error(
            "could not create temporary output beside '" + path.string() +
            "': " + ::llvm::toString(temporary_or_error.takeError()));
    }
    auto temporary = std::move(*temporary_or_error);

    std::error_code write_error;
    {
        ::llvm::raw_fd_ostream output(temporary.FD, /*shouldClose=*/false);
        output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        output.flush();
        write_error = output.error();
        output.clear_error();
    }
    if (write_error) {
        discardTemporaryFile(temporary);
        throw std::runtime_error(
            "failed to write temporary output for '" + path.string() +
            "': " + write_error.message());
    }
    if (auto error = temporary.keep(path.string())) {
        throw std::runtime_error(
            "could not atomically commit output '" + path.string() +
            "': " + ::llvm::toString(std::move(error)));
    }
}

void writeOutput(const CommandLine& command,
                 std::span<const std::uint8_t> bytes) {
    if (command.output_path.empty() || command.output_path == "-") {
        std::cout.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        std::cout.flush();
        if (!std::cout) {
            throw std::runtime_error("failed to write LLVM IR to stdout");
        }
        return;
    }
    writeFileAtomically(command.output_path, bytes);
}

std::vector<std::uint8_t> compileOrLoad(const CommandLine& command,
                                        const std::string& source) {
    const std::string target_triple =
        command.kind == ArtifactKind::HostObject
            ? rocq::compiler::hostTargetTriple()
            : std::string{};

    std::optional<ArtifactCache> cache;
    std::string cache_key;
    if (command.cache_directory) {
        cache.emplace(*command.cache_directory);
        ArtifactCacheKeyInputs key_inputs;
        key_inputs.mlir_source = source;
        key_inputs.profile = command.profile;
        key_inputs.requested_num_qubits = command.num_qubits;
        key_inputs.kind = command.kind;
        key_inputs.optimization_level = command.optimization_level;
        key_inputs.target_triple = target_triple;
        cache_key = rocq::compiler::tools::computeArtifactCacheKey(key_inputs);
        if (auto hit = cache->load(cache_key)) {
            if (command.verbose) {
                std::cerr << "rocq-translate: cache hit " << cache_key << '\n';
            }
            return std::move(*hit);
        }
        if (command.verbose) {
            std::cerr << "rocq-translate: cache miss " << cache_key << '\n';
        }
    }

    rocq::MLIRCompiler compiler(command.num_qubits);
    const auto artifact = compiler.emit_artifact(
        source,
        CompilerArtifactOptions{command.kind, command.optimization_level},
        command.profile);
    if (cache) {
        cache->store(cache_key, artifact.bytes);
    }
    return artifact.bytes;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const CommandLine command = parseCommandLine(argc, argv);
        const std::string source = readInput(command.input_path);
        const auto artifact = compileOrLoad(command, source);
        writeOutput(command, artifact);
        return 0;
    } catch (const UsageError& error) {
        std::cerr << "rocq-translate: " << error.what() << "\n\n";
        printUsage(std::cerr, argv[0]);
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "rocq-translate: " << error.what() << '\n';
        return 1;
    }
}
