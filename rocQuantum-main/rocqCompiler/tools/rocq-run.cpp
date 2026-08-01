#include "rocqCompiler/MLIRCompiler.h"
#include "rocqCompiler/ReferenceStateVecBackend.h"

#include <cmath>
#include <complex>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "llvm/Config/llvm-config.h"

#ifndef ROCQ_VERSION
#define ROCQ_VERSION "unknown"
#endif

namespace {

class UsageError final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

void printUsage(std::ostream& output, const char* program) {
    output
        << "usage: " << program << " <input.mlir|->\n"
        << "\n"
        << "Execute measurement-free rocQuantum MLIR through the static QIR\n"
        << "LLVM ORC JIT and deterministic cpu_statevec backend. The source\n"
        << "qubit count is inferred from quantum.qalloc. JSON is written to\n"
        << "standard output.\n"
        << "\n"
        << "options:\n"
        << "  --version            print compiler and LLVM versions\n"
        << "  -h, --help           show this help\n";
}

std::string parseCommandLine(int argc, char** argv) {
    std::string input_path;
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
            std::cout << "rocq-run " << ROCQ_VERSION << " (LLVM "
                      << LLVM_VERSION_STRING << ")\n";
            std::exit(0);
        }
        if (!positional_only && argument != "-" && !argument.empty() &&
            argument.front() == '-') {
            throw UsageError("unknown option '" + argument + "'");
        }
        if (!input_path.empty()) {
            throw UsageError("exactly one MLIR input is required");
        }
        input_path = argument;
    }
    if (input_path.empty()) {
        throw UsageError("exactly one MLIR input is required");
    }
    return input_path;
}

std::string readInput(const std::string& input_path) {
    if (input_path == "-") {
        return {std::istreambuf_iterator<char>(std::cin),
                std::istreambuf_iterator<char>()};
    }

    std::ifstream input(input_path, std::ios::binary);
    if (!input) {
        throw std::runtime_error(
            "unable to open MLIR input '" + input_path + "'");
    }
    std::string source{
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()};
    if (input.bad()) {
        throw std::runtime_error(
            "failed while reading MLIR input '" + input_path + "'");
    }
    return source;
}

unsigned stateQubitCount(std::size_t dimension) {
    if (dimension == 0 || (dimension & (dimension - 1)) != 0) {
        throw std::runtime_error(
            "cpu_statevec returned a non-power-of-two state dimension");
    }
    unsigned qubits = 0;
    while (dimension > 1) {
        dimension >>= 1;
        ++qubits;
    }
    return qubits;
}

void writeStateJson(
    std::ostream& output,
    const std::vector<std::complex<double>>& state) {
    for (const auto& amplitude : state) {
        if (!std::isfinite(amplitude.real()) ||
            !std::isfinite(amplitude.imag())) {
            throw std::runtime_error(
                "cpu_statevec returned a non-finite amplitude");
        }
    }

    output << "{\"schema\":\"rocq-state-vector-v1\","
           << "\"backend\":\"cpu_statevec\","
           << "\"num_qubits\":" << stateQubitCount(state.size()) << ','
           << "\"amplitudes\":[";
    output << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (std::size_t index = 0; index < state.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        output << "{\"real\":" << state[index].real()
               << ",\"imag\":" << state[index].imag() << '}';
    }
    output << "]}\n";
    if (!output) {
        throw std::runtime_error("failed while writing JSON state output");
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto input_path = parseCommandLine(argc, argv);
        const auto source = readInput(input_path);
        rocq::MLIRCompiler compiler(
            /*num_qubits=*/0, rocq::create_reference_backend());
        writeStateJson(
            std::cout, compiler.compile_and_execute(source, {}));
        return 0;
    } catch (const UsageError& error) {
        std::cerr << "rocq-run: " << error.what() << '\n';
        printUsage(std::cerr, argv[0]);
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "rocq-run: " << error.what() << '\n';
        return 1;
    }
}
