#ifndef MLIR_COMPILER_H
#define MLIR_COMPILER_H

#include <string>
#include <vector>
#include <complex>
#include <memory>
#include <map>
#include "rocqCompiler/CompilerArtifacts.h"
#include "QuantumBackend.h"

namespace rocq {

class MLIRCompiler {
public:
    /// Construct a compiler suitable for offline MLIR -> QIR emission. This
    /// path has no HIP dependency; num_qubits=0 infers the static resource
    /// count from quantum.qalloc.
    explicit MLIRCompiler(unsigned num_qubits = 0);
    MLIRCompiler(unsigned num_qubits, std::unique_ptr<QuantumBackend> backend);
    ~MLIRCompiler();

    std::vector<std::complex<double>> compile_and_execute(
        const std::string& mlir_string,
        const std::map<std::string, bool>& args);

    /// Emit LLVM QIR. qir-v2-static preserves the original measurement-free
    /// custom profile; qir-v2-base emits the verified QIR v2 Base Profile.
    std::string emit_qir(
        const std::string& mlir_string,
        const std::string& profile = "qir-v2-static");

    /// Lower MLIR to a binary-safe compiler artifact.  QIR v2 Base Profile
    /// LLVM IR/bitcode is deliberately restricted to -O0: LLVM's generic
    /// optimization pipelines may merge the profile's required four-block
    /// control-flow shape.  Host objects are executable-linker artifacts, not
    /// QIR interchange modules, and may use -O0 through -O3.
    compiler::CompilerArtifact emit_artifact(
        const std::string& mlir_string,
        const compiler::CompilerArtifactOptions& options = {},
        const std::string& profile = "qir-v2-static");

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;

    unsigned num_qubits;
    std::unique_ptr<QuantumBackend> backend;
};

} // namespace rocq

#endif // MLIR_COMPILER_H
