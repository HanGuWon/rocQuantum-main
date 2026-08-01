#include "MLIRCompiler.h"

#include "rocqCompiler/CompilerArtifacts.h"
#include "rocqCompiler/QIRExecutionEngine.h"
#include "rocqCompiler/QIRProfile.h"
#include "rocqCompiler/QuantumDialect.h"
#include "rocqCompiler/passes/Pipelines.h"

#include <stdexcept>
#include <string>
#include <utility>

#include "llvm/Config/llvm-config.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#if __has_include("mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h")
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#define ROCQ_HAS_BUILTIN_LLVM_TRANSLATION 1
#endif

namespace {

[[noreturn]] void throwCompileDiagnostic(const std::string& message) {
    throw std::runtime_error("compile_and_execute() failed: " + message);
}

} // namespace

namespace rocq {

struct MLIRCompiler::Impl {
    Impl() {
        ::mlir::DialectRegistry registry;
        registry.insert<rocq::quantum::QuantumDialect,
                        ::mlir::func::FuncDialect,
                        ::mlir::LLVM::LLVMDialect>();
        ::mlir::registerLLVMDialectTranslation(registry);
#ifdef ROCQ_HAS_BUILTIN_LLVM_TRANSLATION
        ::mlir::registerBuiltinDialectTranslation(registry);
#endif
        context.appendDialectRegistry(registry);
        context.loadAllAvailableDialects();
    }

    ::mlir::MLIRContext context;
};

MLIRCompiler::MLIRCompiler(unsigned n_qubits)
    : pimpl(std::make_unique<Impl>()), num_qubits(n_qubits), backend(nullptr) {}

MLIRCompiler::MLIRCompiler(unsigned n_qubits, std::unique_ptr<QuantumBackend> execution_backend)
    : pimpl(std::make_unique<Impl>()),
      num_qubits(n_qubits),
      backend(std::move(execution_backend)) {}

MLIRCompiler::~MLIRCompiler() = default;

std::string MLIRCompiler::emit_qir(const std::string& mlir_string,
                                   const std::string& profile) {
    const auto artifact = emit_artifact(
        mlir_string,
        {compiler::ArtifactKind::LlvmIr, /*optimization_level=*/0},
        profile);
    return {reinterpret_cast<const char*>(artifact.bytes.data()),
            artifact.bytes.size()};
}

compiler::CompilerArtifact MLIRCompiler::emit_artifact(
    const std::string& mlir_string,
    const compiler::CompilerArtifactOptions& options,
    const std::string& profile) {
    compiler::QIREmissionProfile emission_profile;
    if (!compiler::parseQIREmissionProfile(profile, emission_profile)) {
        throw std::invalid_argument(
            "emit_artifact() supports profile='qir-v2-static' or 'qir-v2-base'; "
            "requested '" + profile + "'");
    }
    if (options.optimization_level > 3) {
        throw std::invalid_argument(
            "emit_artifact() optimization level must be between 0 and 3");
    }
    if (emission_profile == compiler::QIREmissionProfile::Base &&
        options.kind != compiler::ArtifactKind::HostObject &&
        options.optimization_level != 0) {
        throw std::invalid_argument(
            "emit_artifact() requires -O0 for qir-v2-base LLVM IR and "
            "bitcode because generic LLVM optimization does not preserve "
            "the Base Profile four-block control-flow contract");
    }

    auto module = ::mlir::parseSourceString<::mlir::ModuleOp>(
        mlir_string, &pimpl->context);
    if (!module) {
        throw std::runtime_error(
            "emit_artifact() failed: unable to parse the MLIR module");
    }
    if (::mlir::failed(::mlir::verify(*module))) {
        throw std::runtime_error(
            "emit_artifact() failed: input MLIR verification failed");
    }

    compiler::QIRModuleInfo info;
    std::string diagnostic;
    if (::mlir::failed(compiler::analyzeQuantumModuleForQIR(
            *module, info, emission_profile, &diagnostic))) {
        throw std::runtime_error("emit_artifact() failed: " + diagnostic);
    }
    if (num_qubits != 0 && info.required_qubits != num_qubits) {
        throw std::runtime_error(
            "emit_artifact() failed: quantum.qalloc size " +
            std::to_string(info.required_qubits) +
            " does not match compiler num_qubits " + std::to_string(num_qubits));
    }

    ::mlir::PassManager pass_manager(&pimpl->context);
    pass_manager.enableVerifier(true);
    compiler::buildQIRPipeline(pass_manager, emission_profile);
    if (::mlir::failed(pass_manager.run(*module))) {
        throw std::runtime_error(
            "emit_artifact() failed: Quantum-to-LLVM/QIR lowering failed");
    }
    if (::mlir::failed(::mlir::verify(*module))) {
        throw std::runtime_error(
            "emit_artifact() failed: lowered LLVM-dialect module is invalid");
    }

    ::llvm::LLVMContext llvm_context;
    auto llvm_module = ::mlir::translateModuleToLLVMIR(*module, llvm_context);
    if (!llvm_module) {
        throw std::runtime_error(
            "emit_artifact() failed: LLVM-dialect translation returned no module");
    }

    if (!compiler::finalizeAndVerifyQIRModule(
            *llvm_module, info, emission_profile, diagnostic)) {
        throw std::runtime_error("emit_artifact() failed: " + diagnostic);
    }
    return compiler::emitCompilerArtifact(std::move(llvm_module), options);
}

std::vector<std::complex<double>> MLIRCompiler::compile_and_execute(
    const std::string& mlir_string,
    const std::map<std::string, bool>& args) {
    for (const auto& argument : args) {
        if (argument.first != "strict") {
            throwCompileDiagnostic(
                "kernel argument binding is not implemented; only the boolean "
                "compiler option 'strict' is accepted");
        }
    }
    if (!backend) {
        throwCompileDiagnostic(
            "no execution backend is configured (offline QIR compilation remains available)");
    }
    auto module = ::mlir::parseSourceString<::mlir::ModuleOp>(
        mlir_string, &pimpl->context);
    if (!module || ::mlir::failed(::mlir::verify(*module))) {
        throwCompileDiagnostic("failed to parse or verify the MLIR module");
    }

    compiler::QIRModuleInfo info;
    std::string diagnostic;
    if (::mlir::failed(compiler::analyzeQuantumModuleForQIR(
            *module,
            info,
            compiler::QIREmissionProfile::StaticCustom,
            &diagnostic))) {
        throwCompileDiagnostic(diagnostic);
    }
    if (num_qubits != 0 && info.required_qubits != num_qubits) {
        throwCompileDiagnostic(
            "quantum.qalloc size " + std::to_string(info.required_qubits) +
            " does not match compiler num_qubits " +
            std::to_string(num_qubits));
    }

    ::mlir::PassManager pass_manager(&pimpl->context);
    pass_manager.enableVerifier(true);
    compiler::buildQIRPipeline(
        pass_manager, compiler::QIREmissionProfile::StaticCustom);
    if (::mlir::failed(pass_manager.run(*module))) {
        throwCompileDiagnostic("Quantum-to-LLVM/QIR lowering failed");
    }
    if (::mlir::failed(::mlir::verify(*module))) {
        throwCompileDiagnostic("lowered LLVM-dialect module is invalid");
    }

    try {
        return compiler::QIRExecutionEngine::executeStatic(
            *module, info, *backend);
    } catch (const std::exception& error) {
        throwCompileDiagnostic(error.what());
    }
}

} // namespace rocq
