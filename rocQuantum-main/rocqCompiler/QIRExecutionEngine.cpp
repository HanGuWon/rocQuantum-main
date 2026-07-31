#include "rocqCompiler/QIRExecutionEngine.h"

#include "rocqCompiler/QIRProfile.h"

#include <cstdint>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Target/LLVMIR/Export.h"

namespace {

struct ActiveExecution {
    rocq::QuantumBackend* backend = nullptr;
    std::uint64_t required_qubits = 0;
    std::exception_ptr failure;
};

thread_local ActiveExecution* active_execution = nullptr;

class ActiveExecutionScope final {
public:
    explicit ActiveExecutionScope(ActiveExecution& execution) {
        if (active_execution != nullptr) {
            throw std::runtime_error(
                "nested QIR execution on one host thread is not supported");
        }
        active_execution = &execution;
    }

    ~ActiveExecutionScope() {
        active_execution = nullptr;
    }

    ActiveExecutionScope(const ActiveExecutionScope&) = delete;
    ActiveExecutionScope& operator=(const ActiveExecutionScope&) = delete;
};

class BackendLifetime final {
public:
    BackendLifetime(rocq::QuantumBackend& backend, unsigned num_qubits)
        : backend_(backend) {
        backend_.destroy();
        try {
            backend_.initialize(num_qubits);
            initialized_ = true;
        } catch (...) {
            try {
                backend_.destroy();
            } catch (...) {
                // Preserve the initialization failure.
            }
            throw;
        }
    }

    ~BackendLifetime() {
        if (!initialized_) {
            return;
        }
        try {
            backend_.destroy();
        } catch (...) {
            // Destructors must not mask the execution failure being propagated.
        }
    }

    void destroy() {
        if (!initialized_) {
            return;
        }
        initialized_ = false;
        backend_.destroy();
    }

    BackendLifetime(const BackendLifetime&) = delete;
    BackendLifetime& operator=(const BackendLifetime&) = delete;

private:
    rocq::QuantumBackend& backend_;
    bool initialized_ = false;
};

template <typename Callback>
void protectQisCallback(Callback&& callback) noexcept {
    auto* execution = active_execution;
    if (execution == nullptr || execution->failure) {
        return;
    }
    try {
        callback(*execution);
    } catch (...) {
        execution->failure = std::current_exception();
    }
}

unsigned decodeQubit(ActiveExecution& execution, void* handle) {
    const auto index = static_cast<std::uint64_t>(
        reinterpret_cast<std::uintptr_t>(handle));
    if (index >= execution.required_qubits) {
        throw std::runtime_error(
            "QIR referenced static qubit handle " + std::to_string(index) +
            " outside the declared resource range [0, " +
            std::to_string(execution.required_qubits) + ")");
    }
    return static_cast<unsigned>(index);
}

void applySingleQubitGate(const char* gate_name, void* target) noexcept {
    protectQisCallback([=](ActiveExecution& execution) {
        execution.backend->apply_gate(
            gate_name, {decodeQubit(execution, target)});
    });
}

void applyParametricSingleQubitGate(
    const char* gate_name, double angle, void* target) noexcept {
    protectQisCallback([=](ActiveExecution& execution) {
        execution.backend->apply_parametrized_gate(
            gate_name, angle, {decodeQubit(execution, target)});
    });
}

extern "C" {

void rocqQisH(void* target) noexcept {
    applySingleQubitGate("h", target);
}

void rocqQisX(void* target) noexcept {
    applySingleQubitGate("x", target);
}

void rocqQisY(void* target) noexcept {
    applySingleQubitGate("y", target);
}

void rocqQisZ(void* target) noexcept {
    applySingleQubitGate("z", target);
}

void rocqQisS(void* target) noexcept {
    applySingleQubitGate("s", target);
}

void rocqQisSAdj(void* target) noexcept {
    applySingleQubitGate("sdg", target);
}

void rocqQisT(void* target) noexcept {
    applySingleQubitGate("t", target);
}

void rocqQisTAdj(void* target) noexcept {
    applySingleQubitGate("tdg", target);
}

void rocqQisCnot(void* control, void* target) noexcept {
    protectQisCallback([=](ActiveExecution& execution) {
        execution.backend->apply_gate(
            "cnot",
            {decodeQubit(execution, control), decodeQubit(execution, target)});
    });
}

void rocqQisRx(double angle, void* target) noexcept {
    applyParametricSingleQubitGate("rx", angle, target);
}

void rocqQisRy(double angle, void* target) noexcept {
    applyParametricSingleQubitGate("ry", angle, target);
}

void rocqQisRz(double angle, void* target) noexcept {
    applyParametricSingleQubitGate("rz", angle, target);
}

void rocqQisR1(double angle, void* target) noexcept {
    applyParametricSingleQubitGate("p", angle, target);
}

} // extern "C"

void initializeNativeTarget() {
    static std::once_flag initialized;
    std::call_once(initialized, [] {
        if (::llvm::InitializeNativeTarget()) {
            throw std::runtime_error("LLVM failed to initialize the native target");
        }
        if (::llvm::InitializeNativeTargetAsmPrinter()) {
            throw std::runtime_error(
                "LLVM failed to initialize the native assembly printer");
        }
    });
}

template <typename Function>
void addSymbol(::llvm::orc::SymbolMap& symbols,
               ::llvm::orc::MangleAndInterner interner,
               ::llvm::StringRef name,
               Function* function) {
    symbols[interner(name)] = {
        ::llvm::orc::ExecutorAddr::fromPtr(function),
        ::llvm::JITSymbolFlags::Exported};
}

void registerStaticQisSymbols(::mlir::ExecutionEngine& engine) {
    engine.registerSymbols([](::llvm::orc::MangleAndInterner interner) {
        ::llvm::orc::SymbolMap symbols;
        addSymbol(symbols, interner, "__quantum__qis__h__body", &rocqQisH);
        addSymbol(symbols, interner, "__quantum__qis__x__body", &rocqQisX);
        addSymbol(symbols, interner, "__quantum__qis__y__body", &rocqQisY);
        addSymbol(symbols, interner, "__quantum__qis__z__body", &rocqQisZ);
        addSymbol(symbols, interner, "__quantum__qis__s__body", &rocqQisS);
        addSymbol(symbols, interner, "__quantum__qis__s__adj", &rocqQisSAdj);
        addSymbol(symbols, interner, "__quantum__qis__t__body", &rocqQisT);
        addSymbol(symbols, interner, "__quantum__qis__t__adj", &rocqQisTAdj);
        addSymbol(
            symbols, interner, "__quantum__qis__cnot__body", &rocqQisCnot);
        addSymbol(symbols, interner, "__quantum__qis__rx__body", &rocqQisRx);
        addSymbol(symbols, interner, "__quantum__qis__ry__body", &rocqQisRy);
        addSymbol(symbols, interner, "__quantum__qis__rz__body", &rocqQisRz);
        addSymbol(symbols, interner, "__quantum__qis__r1__body", &rocqQisR1);
        return symbols;
    });
}

std::string errorToString(::llvm::Error error) {
    return ::llvm::toString(std::move(error));
}

} // namespace

namespace rocq::compiler {

std::vector<std::complex<double>> QIRExecutionEngine::executeStatic(
    ::mlir::ModuleOp lowered_module,
    const QIRModuleInfo& info,
    rocq::QuantumBackend& backend) {
    if (!lowered_module) {
        throw std::invalid_argument("QIR JIT received an empty MLIR module");
    }
    if (info.entry_point.empty()) {
        throw std::invalid_argument("QIR JIT requires a named entry point");
    }
    if (info.required_results != 0) {
        throw std::invalid_argument(
            "QIR JIT only supports measurement-free qir-v2-static modules");
    }
    if (info.required_qubits == 0 ||
        info.required_qubits >
            static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max())) {
        throw std::invalid_argument(
            "QIR JIT requires a positive qubit count in the backend range");
    }

    initializeNativeTarget();

    std::string translation_diagnostic;
    auto module_builder =
        [&](::mlir::Operation* operation,
            ::llvm::LLVMContext& context) -> std::unique_ptr<::llvm::Module> {
        auto module = ::llvm::dyn_cast<::mlir::ModuleOp>(operation);
        if (!module) {
            translation_diagnostic =
                "QIR JIT expected a builtin.module operation";
            return nullptr;
        }
        auto llvm_module = ::mlir::translateModuleToLLVMIR(module, context);
        if (!llvm_module) {
            translation_diagnostic =
                "LLVM-dialect translation returned no module";
            return nullptr;
        }
        if (!finalizeAndVerifyQIRModule(
                *llvm_module,
                info,
                QIREmissionProfile::StaticCustom,
                translation_diagnostic)) {
            return nullptr;
        }
        return llvm_module;
    };

    ::mlir::ExecutionEngineOptions options;
    options.llvmModuleBuilder = module_builder;
    auto expected_engine =
        ::mlir::ExecutionEngine::create(lowered_module.getOperation(), options);
    if (!expected_engine) {
        auto message = errorToString(expected_engine.takeError());
        if (!translation_diagnostic.empty()) {
            message = translation_diagnostic + "; " + message;
        }
        throw std::runtime_error("unable to create LLVM ORC JIT: " + message);
    }
    auto engine = std::move(*expected_engine);
    registerStaticQisSymbols(*engine);

    const auto qubit_count = static_cast<unsigned>(info.required_qubits);
    BackendLifetime backend_lifetime(backend, qubit_count);
    ActiveExecution execution{&backend, info.required_qubits, nullptr};
    ActiveExecutionScope execution_scope(execution);

    engine->initialize();
    auto expected_entry = engine->lookup(info.entry_point);
    if (!expected_entry) {
        throw std::runtime_error(
            "unable to resolve QIR entry point '" + info.entry_point +
            "': " + errorToString(expected_entry.takeError()));
    }

    auto entry_point = reinterpret_cast<void (*)()>(*expected_entry);
    entry_point();
    if (execution.failure) {
        std::rethrow_exception(execution.failure);
    }

    auto state = backend.get_state_vector();
    backend_lifetime.destroy();
    return state;
}

} // namespace rocq::compiler
