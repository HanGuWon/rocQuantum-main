#include "MLIRCompiler.h"

#include "rocqCompiler/CompilerArtifacts.h"
#include "rocqCompiler/QIRProfile.h"
#include "rocqCompiler/QuantumDialect.h"
#include "rocqCompiler/passes/Pipelines.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
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
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
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

struct ExecutableOp {
    std::string gate_name;
    std::vector<unsigned> targets;
    double parameter = 0.0;
    bool parametrized = false;
};

std::string supportedCompileExecuteSubset() {
    return "supported subset: quantum.qalloc, H/X/Y/Z/S/Sdg/T/Tdg, "
           "CNOT/CZ/SWAP/CCX/MCX/CSWAP, RX/RY/RZ/P, CRX/CRY/CRZ/CP";
}

[[noreturn]] void throwCompileDiagnostic(const std::string& message) {
    throw std::runtime_error("compile_and_execute() failed: " + message);
}

unsigned resolveQubitIndex(
    ::mlir::Value value,
    const ::llvm::DenseMap<::mlir::Value, unsigned>& qubit_indices,
    const std::string& operation_name) {
    auto iterator = qubit_indices.find(value);
    if (iterator == qubit_indices.end()) {
        throwCompileDiagnostic(
            "operation '" + operation_name +
            "' references a qubit value that was not produced by quantum.qalloc");
    }
    return iterator->second;
}

void validateDistinctTargets(const std::vector<unsigned>& targets,
                             const std::string& operation_name) {
    std::unordered_set<unsigned> seen;
    for (unsigned target : targets) {
        if (!seen.insert(target).second) {
            throwCompileDiagnostic(
                "operation '" + operation_name + "' qubit operands must be distinct");
        }
    }
}

std::vector<unsigned> resolveTargets(
    ::mlir::Operation* operation,
    const ::llvm::DenseMap<::mlir::Value, unsigned>& qubit_indices,
    unsigned expected_operands) {
    const std::string operation_name = operation->getName().getStringRef().str();
    if (operation->getNumOperands() != expected_operands) {
        throwCompileDiagnostic(
            "operation '" + operation_name + "' expected " +
            std::to_string(expected_operands) + " qubit operands but received " +
            std::to_string(operation->getNumOperands()));
    }

    std::vector<unsigned> targets;
    targets.reserve(expected_operands);
    for (::mlir::Value operand : operation->getOperands()) {
        targets.push_back(resolveQubitIndex(operand, qubit_indices, operation_name));
    }
    validateDistinctTargets(targets, operation_name);
    return targets;
}

std::vector<unsigned> resolveTargetsAtLeast(
    ::mlir::Operation* operation,
    const ::llvm::DenseMap<::mlir::Value, unsigned>& qubit_indices,
    unsigned minimum_operands) {
    const std::string operation_name = operation->getName().getStringRef().str();
    if (operation->getNumOperands() < minimum_operands) {
        throwCompileDiagnostic(
            "operation '" + operation_name + "' expected at least " +
            std::to_string(minimum_operands) + " qubit operands but received " +
            std::to_string(operation->getNumOperands()));
    }

    std::vector<unsigned> targets;
    targets.reserve(operation->getNumOperands());
    for (::mlir::Value operand : operation->getOperands()) {
        targets.push_back(resolveQubitIndex(operand, qubit_indices, operation_name));
    }
    validateDistinctTargets(targets, operation_name);
    return targets;
}

::mlir::func::FuncOp requireSingleStraightLineFunction(::mlir::ModuleOp module) {
    ::mlir::func::FuncOp entry;
    unsigned function_count = 0;
    for (auto function : module.getOps<::mlir::func::FuncOp>()) {
        ++function_count;
        entry = function;
    }
    if (function_count != 1 || !entry || entry.isExternal()) {
        throwCompileDiagnostic("exactly one defined func.func entry point is required");
    }
    if (entry.getNumArguments() != 0 || entry.getFunctionType().getNumResults() != 0) {
        throwCompileDiagnostic("the entry point must have no arguments or results");
    }
    if (!entry.getBody().hasOneBlock()) {
        throwCompileDiagnostic("classical control flow is not supported by the execution MVP");
    }
    return entry;
}

std::vector<ExecutableOp> extractExecutableOps(::mlir::ModuleOp module,
                                               unsigned expected_num_qubits) {
    if (expected_num_qubits == 0) {
        throwCompileDiagnostic("num_qubits must be positive for backend execution");
    }

    auto entry = requireSingleStraightLineFunction(module);
    ::llvm::DenseMap<::mlir::Value, unsigned> qubit_indices;
    std::vector<ExecutableOp> executable_ops;
    bool saw_qalloc = false;

    static const std::unordered_set<std::string> simple_gates = {
        "quantum.h", "quantum.x", "quantum.y", "quantum.z",
        "quantum.s", "quantum.sdg", "quantum.t", "quantum.tdg",
    };
    static const std::unordered_map<std::string, unsigned> parametrized_gate_arities = {
        {"quantum.rx", 1}, {"quantum.ry", 1}, {"quantum.rz", 1}, {"quantum.p", 1},
        {"quantum.crx", 2}, {"quantum.cry", 2}, {"quantum.crz", 2}, {"quantum.cp", 2},
    };
    static const std::unordered_map<std::string, unsigned> fixed_arity_gates = {
        {"quantum.cnot", 2}, {"quantum.cz", 2}, {"quantum.swap", 2},
        {"quantum.ccx", 3}, {"quantum.cswap", 3},
    };

    for (::mlir::Operation& operation : entry.getBody().front()) {
        const std::string operation_name = operation.getName().getStringRef().str();
        if (operation_name == "func.return") {
            continue;
        }
        if (operation_name == "quantum.qalloc") {
            if (saw_qalloc) {
                throwCompileDiagnostic("multiple quantum.qalloc operations are not supported");
            }
            auto size = operation.getAttrOfType<::mlir::IntegerAttr>("size");
            if (!size || size.getInt() <= 0) {
                throwCompileDiagnostic("quantum.qalloc requires a positive i64 size attribute");
            }
            if (size.getInt() > static_cast<std::int64_t>(
                                    std::numeric_limits<unsigned>::max())) {
                throwCompileDiagnostic("qalloc size exceeds the supported compiler range");
            }
            const auto allocated = static_cast<unsigned>(size.getInt());
            if (allocated != expected_num_qubits) {
                throwCompileDiagnostic(
                    "quantum.qalloc size " + std::to_string(allocated) +
                    " does not match compiler num_qubits " +
                    std::to_string(expected_num_qubits));
            }
            if (operation.getNumResults() != allocated) {
                throwCompileDiagnostic(
                    "quantum.qalloc result count does not match its size attribute");
            }
            for (unsigned index = 0; index < allocated; ++index) {
                qubit_indices[operation.getResult(index)] = index;
            }
            saw_qalloc = true;
            continue;
        }

        if (simple_gates.count(operation_name) != 0) {
            executable_ops.push_back({
                operation_name.substr(std::string("quantum.").size()),
                resolveTargets(&operation, qubit_indices, 1)});
            continue;
        }

        auto fixed = fixed_arity_gates.find(operation_name);
        if (fixed != fixed_arity_gates.end()) {
            executable_ops.push_back({
                operation_name.substr(std::string("quantum.").size()),
                resolveTargets(&operation, qubit_indices, fixed->second)});
            continue;
        }

        if (operation_name == "quantum.mcx") {
            executable_ops.push_back({"mcx", resolveTargetsAtLeast(
                &operation, qubit_indices, /*minimum_operands=*/2)});
            continue;
        }

        auto parametrized = parametrized_gate_arities.find(operation_name);
        if (parametrized != parametrized_gate_arities.end()) {
            auto angle = operation.getAttrOfType<::mlir::FloatAttr>("angle");
            if (!angle || !std::isfinite(angle.getValueAsDouble())) {
                throwCompileDiagnostic(
                    "operation '" + operation_name + "' angle must be finite");
            }
            executable_ops.push_back({
                operation_name.substr(std::string("quantum.").size()),
                resolveTargets(&operation, qubit_indices, parametrized->second),
                angle.getValueAsDouble(),
                true});
            continue;
        }

        if (operation_name.rfind("quantum.", 0) == 0) {
            throwCompileDiagnostic(
                "unsupported quantum op '" + operation_name + "'; " +
                supportedCompileExecuteSubset());
        }
        throwCompileDiagnostic(
            "unsupported classical operation '" + operation_name +
            "' in the straight-line execution MVP");
    }

    if (!saw_qalloc) {
        throwCompileDiagnostic(
            "no quantum.qalloc operation found; " + supportedCompileExecuteSubset());
    }
    return executable_ops;
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
    auto executable_ops = extractExecutableOps(*module, num_qubits);

    backend->destroy();
    try {
        backend->initialize(num_qubits);
        for (const auto& operation : executable_ops) {
            if (operation.parametrized) {
                backend->apply_parametrized_gate(
                    operation.gate_name, operation.parameter, operation.targets);
            } else {
                backend->apply_gate(operation.gate_name, operation.targets);
            }
        }
        auto state = backend->get_state_vector();
        backend->destroy();
        return state;
    } catch (...) {
        backend->destroy();
        throw;
    }
}

} // namespace rocq
