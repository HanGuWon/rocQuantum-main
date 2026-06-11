#include "MLIRCompiler.h"
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

// MLIR Core & Passes
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

// LLVM
#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

// Our Dialects and Passes
#include "QuantumOps.h.inc"
#include "SimulatorOps.h.inc"

// Forward declarations for our custom pass creation functions
std::unique_ptr<mlir::Pass> createQuantumToSimulatorPass();
std::unique_ptr<mlir::Pass> createSimulatorToQIRPass();

namespace rocq {

namespace {

struct ExecutableOp {
    std::string gate_name;
    std::vector<unsigned> targets;
    double parameter = 0.0;
    bool parametrized = false;
};

std::string supported_compile_execute_subset() {
    return "supported subset: quantum.qalloc, H/X/Y/Z/S/Sdg/T/Tdg, CNOT/CZ/SWAP/CCX/MCX/CSWAP, "
           "RX/RY/RZ, CRX/CRY/CRZ";
}

[[noreturn]] void throw_compile_diagnostic(const std::string& message) {
    throw std::runtime_error("compile_and_execute() failed: " + message);
}

unsigned resolve_qubit_index(mlir::Value value,
                             const llvm::DenseMap<mlir::Value, unsigned>& qubit_indices,
                             const std::string& op_name) {
    auto it = qubit_indices.find(value);
    if (it == qubit_indices.end()) {
        throw_compile_diagnostic(
            "operation '" + op_name + "' references a qubit value that was not produced by quantum.qalloc.");
    }
    return it->second;
}

std::vector<unsigned> resolve_targets(mlir::Operation* op,
                                      const llvm::DenseMap<mlir::Value, unsigned>& qubit_indices,
                                      unsigned expected_operands) {
    const std::string op_name = op->getName().getStringRef().str();
    if (op->getNumOperands() != expected_operands) {
        throw_compile_diagnostic(
            "operation '" + op_name + "' expected " + std::to_string(expected_operands) +
            " qubit operands but received " + std::to_string(op->getNumOperands()) + ".");
    }

    std::vector<unsigned> targets;
    targets.reserve(expected_operands);
    for (mlir::Value operand : op->getOperands()) {
        targets.push_back(resolve_qubit_index(operand, qubit_indices, op_name));
    }
    return targets;
}

std::vector<unsigned> resolve_targets_at_least(mlir::Operation* op,
                                               const llvm::DenseMap<mlir::Value, unsigned>& qubit_indices,
                                               unsigned min_operands) {
    const std::string op_name = op->getName().getStringRef().str();
    if (op->getNumOperands() < min_operands) {
        throw_compile_diagnostic(
            "operation '" + op_name + "' expected at least " + std::to_string(min_operands) +
            " qubit operands but received " + std::to_string(op->getNumOperands()) + ".");
    }

    std::vector<unsigned> targets;
    targets.reserve(op->getNumOperands());
    for (mlir::Value operand : op->getOperands()) {
        targets.push_back(resolve_qubit_index(operand, qubit_indices, op_name));
    }
    return targets;
}

std::vector<ExecutableOp> extract_executable_ops(mlir::ModuleOp module, unsigned expected_num_qubits) {
    llvm::DenseMap<mlir::Value, unsigned> qubit_indices;
    std::vector<ExecutableOp> executable_ops;
    bool saw_qalloc = false;

    static const std::unordered_set<std::string> simple_gates = {
        "quantum.h",
        "quantum.x",
        "quantum.y",
        "quantum.z",
        "quantum.s",
        "quantum.sdg",
        "quantum.t",
        "quantum.tdg",
    };
    static const std::unordered_map<std::string, unsigned> parametrized_gate_arities = {
        {"quantum.rx", 1},
        {"quantum.ry", 1},
        {"quantum.rz", 1},
        {"quantum.crx", 2},
        {"quantum.cry", 2},
        {"quantum.crz", 2},
    };
    static const std::unordered_map<std::string, unsigned> fixed_arity_gates = {
        {"quantum.cnot", 2},
        {"quantum.cz", 2},
        {"quantum.swap", 2},
        {"quantum.ccx", 3},
        {"quantum.cswap", 3},
    };
    static const std::unordered_map<std::string, unsigned> variadic_min_arity_gates = {
        {"quantum.mcx", 2},
    };

    module.walk([&](mlir::Operation* op) {
        const std::string op_name = op->getName().getStringRef().str();
        if (op_name == "quantum.qalloc") {
            if (saw_qalloc) {
                throw_compile_diagnostic(
                    "multiple quantum.qalloc operations are not supported by the compile-and-execute MVP.");
            }
            auto size_attr = op->getAttrOfType<mlir::IntegerAttr>("size");
            if (!size_attr) {
                throw_compile_diagnostic("quantum.qalloc is missing the required 'size' attribute.");
            }

            const auto allocated_qubits_signed = size_attr.getInt();
            if (allocated_qubits_signed <= 0) {
                throw_compile_diagnostic(
                    "quantum.qalloc size must be positive, received " +
                    std::to_string(allocated_qubits_signed) + ".");
            }

            const auto allocated_qubits = static_cast<unsigned>(allocated_qubits_signed);
            if (allocated_qubits != expected_num_qubits) {
                throw_compile_diagnostic(
                    "quantum.qalloc size " + std::to_string(allocated_qubits) +
                    " does not match compiler num_qubits " + std::to_string(expected_num_qubits) + ".");
            }
            if (op->getNumResults() != allocated_qubits) {
                throw_compile_diagnostic(
                    "quantum.qalloc produced " + std::to_string(op->getNumResults()) +
                    " results for size " + std::to_string(allocated_qubits) + ".");
            }

            for (unsigned i = 0; i < allocated_qubits; ++i) {
                qubit_indices[op->getResult(i)] = i;
            }
            saw_qalloc = true;
            return;
        }

        if (simple_gates.count(op_name) != 0) {
            executable_ops.push_back({op_name.substr(std::string("quantum.").size()),
                                      resolve_targets(op, qubit_indices, 1)});
            return;
        }

        auto fixed_it = fixed_arity_gates.find(op_name);
        if (fixed_it != fixed_arity_gates.end()) {
            executable_ops.push_back({op_name.substr(std::string("quantum.").size()),
                                      resolve_targets(op, qubit_indices, fixed_it->second)});
            return;
        }

        auto variadic_it = variadic_min_arity_gates.find(op_name);
        if (variadic_it != variadic_min_arity_gates.end()) {
            executable_ops.push_back({op_name.substr(std::string("quantum.").size()),
                                      resolve_targets_at_least(op, qubit_indices, variadic_it->second)});
            return;
        }

        auto param_it = parametrized_gate_arities.find(op_name);
        if (param_it != parametrized_gate_arities.end()) {
            auto angle_attr = op->getAttrOfType<mlir::FloatAttr>("angle");
            if (!angle_attr) {
                throw_compile_diagnostic("operation '" + op_name + "' is missing the required 'angle' attribute.");
            }
            executable_ops.push_back({op_name.substr(std::string("quantum.").size()),
                                      resolve_targets(op, qubit_indices, param_it->second),
                                      angle_attr.getValueAsDouble(),
                                      true});
            return;
        }

        if (op_name.rfind("quantum.", 0) == 0) {
            throw_compile_diagnostic(
                "unsupported quantum op '" + op_name + "'; " + supported_compile_execute_subset() + ".");
        }
    });

    if (!saw_qalloc) {
        throw_compile_diagnostic("no quantum.qalloc operation found; " + supported_compile_execute_subset() + ".");
    }

    return executable_ops;
}

} // namespace

struct MLIRCompiler::Impl {
    mlir::MLIRContext context;
};

MLIRCompiler::MLIRCompiler(unsigned n_qubits, std::unique_ptr<QuantumBackend> be)
    : num_qubits(n_qubits), backend(std::move(be)), pimpl(new Impl) {
    pimpl->context.getOrLoadDialect<rocq::mlir::quantum::QuantumDialect>();
    pimpl->context.getOrLoadDialect<rocq::mlir::sim::SimulatorDialect>();
    pimpl->context.getOrLoadDialect<mlir::func::FuncDialect>();
    pimpl->context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
    mlir::registerLLVMDialectTranslation(pimpl->context);
}

MLIRCompiler::~MLIRCompiler() {
    delete pimpl;
}

std::string MLIRCompiler::emit_qir(const std::string& mlir_string) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(mlir_string, &pimpl->context);
    if (!module) {
        return "Error: Failed to parse MLIR string.";
    }

    // --- Run the full lowering pipeline ---
    mlir::PassManager pm(&pimpl->context);
    pm.addPass(createQuantumToSimulatorPass()); // quantum -> sim
    pm.addPass(createSimulatorToQIRPass());     // sim -> llvm
    
    // Add standard passes to convert the rest of the IR to LLVM
    pm.addPass(mlir::createConvertSCFToCFPass());
    pm.addPass(mlir::createConvertControlFlowToLLVMPass());
    pm.addPass(mlir::createConvertFuncToLLVMPass());

    if (mlir::failed(pm.run(*module))) {
        return "Error: Failed to run lowering passes to LLVM.";
    }

    // --- Translate to LLVM IR ---
    llvm::LLVMContext llvmContext;
    auto llvmModule = mlir::translateModuleToLLVMIR(*module, llvmContext);
    if (!llvmModule) {
        return "Error: Failed to translate MLIR to LLVM IR.";
    }

    std::string llvm_ir_string;
    llvm::raw_string_ostream os(llvm_ir_string);
    llvmModule->print(os, nullptr);
    
    return os.str();
}


std::vector<std::complex<double>> MLIRCompiler::compile_and_execute(
    const std::string& mlir_string,
    const std::map<std::string, bool>& args) {
    (void)args;

    if (!backend) {
        throw_compile_diagnostic("no execution backend is configured.");
    }

    auto module = mlir::parseSourceString<mlir::ModuleOp>(mlir_string, &pimpl->context);
    if (!module) {
        throw_compile_diagnostic("failed to parse MLIR module.");
    }

    auto executable_ops = extract_executable_ops(*module, num_qubits);

    backend->destroy();
    try {
        backend->initialize(num_qubits);

        for (const auto& op : executable_ops) {
            if (op.parametrized) {
                backend->apply_parametrized_gate(op.gate_name, op.parameter, op.targets);
            } else {
                backend->apply_gate(op.gate_name, op.targets);
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
