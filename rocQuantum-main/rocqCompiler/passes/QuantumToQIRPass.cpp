#include "QuantumToQIRPass.h"

#include "rocqCompiler/QuantumDialect.h"
#include "rocqCompiler/QuantumOps.h"
#include "rocqCompiler/QuantumTypes.h"
#include "rocqCompiler/passes/Passes.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Config/llvm-config.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace rocq::compiler {

#if LLVM_VERSION_MAJOR >= 17
#define GEN_PASS_DEF_QUANTUMTOQIR
#include "passes/Passes.h.inc"
#else
#define GEN_PASS_CLASSES
#include "passes/Passes.h.inc"
#endif

namespace {

enum class GateKind {
    DirectH,
    DirectX,
    DirectY,
    DirectZ,
    DirectS,
    DirectSAdj,
    DirectT,
    DirectTAdj,
    DirectCnot,
    DecomposeCz,
    DecomposeSwap,
    DecomposeCcx,
    DecomposeMcx,
    DecomposeCswap,
    DirectRx,
    DirectRy,
    DirectRz,
    DirectR1,
    DecomposeCrx,
    DecomposeCry,
    DecomposeCrz,
    DecomposeCp,
};

struct GateSpec {
    const char* operation_name;
    GateKind kind;
    unsigned arity;
    bool parametrized;
};

constexpr std::array<GateSpec, 22> kGateSpecs{{
    {"quantum.h", GateKind::DirectH, 1, false},
    {"quantum.x", GateKind::DirectX, 1, false},
    {"quantum.y", GateKind::DirectY, 1, false},
    {"quantum.z", GateKind::DirectZ, 1, false},
    {"quantum.s", GateKind::DirectS, 1, false},
    {"quantum.sdg", GateKind::DirectSAdj, 1, false},
    {"quantum.t", GateKind::DirectT, 1, false},
    {"quantum.tdg", GateKind::DirectTAdj, 1, false},
    {"quantum.cnot", GateKind::DirectCnot, 2, false},
    {"quantum.cz", GateKind::DecomposeCz, 2, false},
    {"quantum.swap", GateKind::DecomposeSwap, 2, false},
    {"quantum.ccx", GateKind::DecomposeCcx, 3, false},
    {"quantum.mcx", GateKind::DecomposeMcx, 2, false},
    {"quantum.cswap", GateKind::DecomposeCswap, 3, false},
    {"quantum.rx", GateKind::DirectRx, 1, true},
    {"quantum.ry", GateKind::DirectRy, 1, true},
    {"quantum.rz", GateKind::DirectRz, 1, true},
    {"quantum.p", GateKind::DirectR1, 1, true},
    {"quantum.crx", GateKind::DecomposeCrx, 2, true},
    {"quantum.cry", GateKind::DecomposeCry, 2, true},
    {"quantum.crz", GateKind::DecomposeCrz, 2, true},
    {"quantum.cp", GateKind::DecomposeCp, 2, true},
}};

const GateSpec* findGateSpec(::llvm::StringRef operation_name) {
    for (const auto& spec : kGateSpecs) {
        if (operation_name == spec.operation_name) {
            return &spec;
        }
    }
    return nullptr;
}

::mlir::LogicalResult fail(::mlir::Operation* operation,
                           const ::llvm::Twine& message,
                           std::string* diagnostic) {
    const std::string text = message.str();
    if (diagnostic) {
        *diagnostic = text;
    }
    operation->emitError(text);
    return ::mlir::failure();
}

::mlir::LLVM::LLVMPointerType qubitPointerType(::mlir::MLIRContext* context) {
    // LLVM 15+ and QIR 2 use opaque pointers. The release CMake graph pins
    // MLIR 22.1.x; this call also keeps development smoke builds possible.
    return ::mlir::LLVM::LLVMPointerType::get(context);
}

::mlir::Value emitStaticHandle(::mlir::ConversionPatternRewriter& rewriter,
                               ::mlir::Location location,
                               ::mlir::LLVM::LLVMPointerType pointer_type,
                               std::uint64_t index) {
    auto id = rewriter.create<::mlir::LLVM::ConstantOp>(
        location,
        rewriter.getI64Type(),
        rewriter.getI64IntegerAttr(index));
#if LLVM_VERSION_MAJOR >= 22
    return rewriter.create<::mlir::LLVM::IntToPtrOp>(
        location,
        pointer_type,
        id.getResult(),
        ::mlir::LLVM::DereferenceableAttr{});
#else
    return rewriter.create<::mlir::LLVM::IntToPtrOp>(
        location, pointer_type, id);
#endif
}

class QallocLowering final : public ::mlir::ConversionPattern {
public:
    QallocLowering(::mlir::LLVMTypeConverter& type_converter,
                   ::mlir::MLIRContext* context,
                   ::mlir::LLVM::LLVMPointerType qubit_pointer)
        : ConversionPattern(type_converter,
                            "quantum.qalloc",
                            /*benefit=*/1,
                            context),
          qubit_pointer_(qubit_pointer) {}

    ::mlir::LogicalResult matchAndRewrite(
        ::mlir::Operation* operation,
        ::llvm::ArrayRef<::mlir::Value> operands,
        ::mlir::ConversionPatternRewriter& rewriter) const override {
        if (!operands.empty()) {
            return rewriter.notifyMatchFailure(operation, "qalloc unexpectedly had operands");
        }

        ::llvm::SmallVector<::mlir::Value> handles;
        handles.reserve(operation->getNumResults());
        for (std::uint64_t index = 0; index < operation->getNumResults(); ++index) {
            handles.push_back(emitStaticHandle(
                rewriter, operation->getLoc(), qubit_pointer_, index));
        }
        rewriter.replaceOp(operation, handles);
        return ::mlir::success();
    }

private:
    ::mlir::LLVM::LLVMPointerType qubit_pointer_;
};

::mlir::FailureOr<::mlir::LLVM::LLVMFuncOp> getOrInsertQisDeclaration(
    ::mlir::ConversionPatternRewriter& rewriter,
    ::mlir::ModuleOp module,
    ::mlir::Location location,
    ::llvm::StringRef suffix,
    ::llvm::ArrayRef<::mlir::Type> argument_types) {
    const std::string name = (::llvm::Twine("__quantum__qis__") + suffix).str();
    auto function_type = ::mlir::LLVM::LLVMFunctionType::get(
        ::mlir::LLVM::LLVMVoidType::get(rewriter.getContext()),
        argument_types,
        /*isVarArg=*/false);

    if (auto existing = module.lookupSymbol<::mlir::LLVM::LLVMFuncOp>(name)) {
        if (existing.getFunctionType() != function_type) {
            existing.emitError("conflicting QIS declaration for '")
                << name << "' (the same symbol was requested with another signature)";
            return ::mlir::failure();
        }
        return existing;
    }

    ::mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    return rewriter.create<::mlir::LLVM::LLVMFuncOp>(location, name, function_type);
}

::mlir::LogicalResult emitQisCall(::mlir::ConversionPatternRewriter& rewriter,
                                  ::mlir::ModuleOp module,
                                  ::mlir::Location location,
                                  ::llvm::StringRef suffix,
                                  ::mlir::ValueRange arguments) {
    ::llvm::SmallVector<::mlir::Type> argument_types;
    argument_types.reserve(arguments.size());
    for (::mlir::Value argument : arguments) {
        argument_types.push_back(argument.getType());
    }
    auto declaration = getOrInsertQisDeclaration(
        rewriter, module, location, suffix, argument_types);
    if (::mlir::failed(declaration)) {
        return ::mlir::failure();
    }
    rewriter.create<::mlir::LLVM::CallOp>(
        location, declaration.value(), arguments);
    return ::mlir::success();
}

class MzLowering final : public ::mlir::ConversionPattern {
public:
    MzLowering(
        ::mlir::LLVMTypeConverter& type_converter,
        ::mlir::MLIRContext* context,
        ::mlir::LLVM::LLVMPointerType result_pointer,
        const ::llvm::DenseMap<::mlir::Operation*, std::uint64_t>& result_indices)
        : ConversionPattern(type_converter,
                            "quantum.mz",
                            /*benefit=*/1,
                            context),
          result_pointer_(result_pointer),
          result_indices_(result_indices) {}

    ::mlir::LogicalResult matchAndRewrite(
        ::mlir::Operation* operation,
        ::llvm::ArrayRef<::mlir::Value> operands,
        ::mlir::ConversionPatternRewriter& rewriter) const override {
        if (operands.size() != 1 || operation->getNumResults() != 1) {
            return rewriter.notifyMatchFailure(
                operation, "quantum.mz requires one qubit and one result");
        }
        auto index = result_indices_.find(operation);
        if (index == result_indices_.end()) {
            return rewriter.notifyMatchFailure(
                operation, "quantum.mz did not receive a static result index");
        }
        auto module = operation->getParentOfType<::mlir::ModuleOp>();
        if (!module) {
            return rewriter.notifyMatchFailure(
                operation, "quantum.mz must be nested in a module");
        }

        auto result_handle = emitStaticHandle(
            rewriter, operation->getLoc(), result_pointer_, index->second);
        if (::mlir::failed(emitQisCall(
                rewriter,
                module,
                operation->getLoc(),
                "mz__body",
                {operands.front(), result_handle}))) {
            return ::mlir::failure();
        }
        rewriter.replaceOp(operation, result_handle);
        return ::mlir::success();
    }

private:
    ::mlir::LLVM::LLVMPointerType result_pointer_;
    ::llvm::DenseMap<::mlir::Operation*, std::uint64_t> result_indices_;
};

::mlir::Value emitF64Constant(::mlir::ConversionPatternRewriter& rewriter,
                              ::mlir::Location location,
                              double value) {
    return rewriter.create<::mlir::LLVM::ConstantOp>(
        location, rewriter.getF64Type(), rewriter.getF64FloatAttr(value));
}

::mlir::LogicalResult emitSingleQubit(
    ::mlir::ConversionPatternRewriter& rewriter,
    ::mlir::ModuleOp module,
    ::mlir::Location location,
    ::llvm::StringRef suffix,
    ::mlir::Value qubit) {
    return emitQisCall(rewriter, module, location, suffix, {qubit});
}

::mlir::LogicalResult emitParametricSingleQubit(
    ::mlir::ConversionPatternRewriter& rewriter,
    ::mlir::ModuleOp module,
    ::mlir::Location location,
    ::llvm::StringRef suffix,
    double angle,
    ::mlir::Value qubit) {
    auto angle_value = emitF64Constant(rewriter, location, angle);
    return emitQisCall(rewriter, module, location, suffix, {angle_value, qubit});
}

::mlir::LogicalResult emitCnot(::mlir::ConversionPatternRewriter& rewriter,
                               ::mlir::ModuleOp module,
                               ::mlir::Location location,
                               ::mlir::Value control,
                               ::mlir::Value target) {
    return emitQisCall(rewriter, module, location, "cnot__body", {control, target});
}

::mlir::LogicalResult emitCcx(::mlir::ConversionPatternRewriter& rewriter,
                              ::mlir::ModuleOp module,
                              ::mlir::Location location,
                              ::mlir::Value control0,
                              ::mlir::Value control1,
                              ::mlir::Value target) {
    // Standard no-ancilla Clifford+T decomposition of Toffoli.
    if (::mlir::failed(emitSingleQubit(rewriter, module, location, "h__body", target)) ||
        ::mlir::failed(emitCnot(rewriter, module, location, control1, target)) ||
        ::mlir::failed(emitSingleQubit(rewriter, module, location, "t__adj", target)) ||
        ::mlir::failed(emitCnot(rewriter, module, location, control0, target)) ||
        ::mlir::failed(emitSingleQubit(rewriter, module, location, "t__body", target)) ||
        ::mlir::failed(emitCnot(rewriter, module, location, control1, target)) ||
        ::mlir::failed(emitSingleQubit(rewriter, module, location, "t__adj", target)) ||
        ::mlir::failed(emitCnot(rewriter, module, location, control0, target)) ||
        ::mlir::failed(emitSingleQubit(rewriter, module, location, "t__body", control1)) ||
        ::mlir::failed(emitSingleQubit(rewriter, module, location, "t__body", target)) ||
        ::mlir::failed(emitSingleQubit(rewriter, module, location, "h__body", target)) ||
        ::mlir::failed(emitCnot(rewriter, module, location, control0, control1)) ||
        ::mlir::failed(emitSingleQubit(rewriter, module, location, "t__body", control0)) ||
        ::mlir::failed(emitSingleQubit(rewriter, module, location, "t__adj", control1)) ||
        ::mlir::failed(emitCnot(rewriter, module, location, control0, control1))) {
        return ::mlir::failure();
    }
    return ::mlir::success();
}

class GateLowering final : public ::mlir::ConversionPattern {
public:
    GateLowering(::mlir::LLVMTypeConverter& type_converter,
                 ::mlir::MLIRContext* context,
                 GateSpec spec)
        : ConversionPattern(type_converter,
                            spec.operation_name,
                            /*benefit=*/1,
                            context),
          spec_(spec) {}

    ::mlir::LogicalResult matchAndRewrite(
        ::mlir::Operation* operation,
        ::llvm::ArrayRef<::mlir::Value> operands,
        ::mlir::ConversionPatternRewriter& rewriter) const override {
        auto module = operation->getParentOfType<::mlir::ModuleOp>();
        const bool has_valid_minimum_arity =
            spec_.kind == GateKind::DecomposeMcx
                ? operands.size() >= spec_.arity
                : operands.size() == spec_.arity;
        if (!module || !has_valid_minimum_arity) {
            return rewriter.notifyMatchFailure(operation, "invalid gate parent or operand arity");
        }

        double angle = 0.0;
        if (spec_.parametrized) {
            auto angle_attr = operation->getAttrOfType<::mlir::FloatAttr>("angle");
            if (!angle_attr) {
                return rewriter.notifyMatchFailure(operation, "missing angle attribute");
            }
            angle = angle_attr.getValueAsDouble();
        }

        const auto location = operation->getLoc();
        ::mlir::LogicalResult result = ::mlir::failure();
        switch (spec_.kind) {
        case GateKind::DirectH:
            result = emitSingleQubit(rewriter, module, location, "h__body", operands[0]);
            break;
        case GateKind::DirectX:
            result = emitSingleQubit(rewriter, module, location, "x__body", operands[0]);
            break;
        case GateKind::DirectY:
            result = emitSingleQubit(rewriter, module, location, "y__body", operands[0]);
            break;
        case GateKind::DirectZ:
            result = emitSingleQubit(rewriter, module, location, "z__body", operands[0]);
            break;
        case GateKind::DirectS:
            result = emitSingleQubit(rewriter, module, location, "s__body", operands[0]);
            break;
        case GateKind::DirectSAdj:
            result = emitSingleQubit(rewriter, module, location, "s__adj", operands[0]);
            break;
        case GateKind::DirectT:
            result = emitSingleQubit(rewriter, module, location, "t__body", operands[0]);
            break;
        case GateKind::DirectTAdj:
            result = emitSingleQubit(rewriter, module, location, "t__adj", operands[0]);
            break;
        case GateKind::DirectCnot:
            result = emitCnot(rewriter, module, location, operands[0], operands[1]);
            break;
        case GateKind::DecomposeCz:
            result = emitSingleQubit(rewriter, module, location, "h__body", operands[1]);
            if (::mlir::succeeded(result)) {
                result = emitCnot(rewriter, module, location, operands[0], operands[1]);
            }
            if (::mlir::succeeded(result)) {
                result = emitSingleQubit(rewriter, module, location, "h__body", operands[1]);
            }
            break;
        case GateKind::DecomposeSwap:
            result = emitCnot(rewriter, module, location, operands[0], operands[1]);
            if (::mlir::succeeded(result)) {
                result = emitCnot(rewriter, module, location, operands[1], operands[0]);
            }
            if (::mlir::succeeded(result)) {
                result = emitCnot(rewriter, module, location, operands[0], operands[1]);
            }
            break;
        case GateKind::DecomposeCcx:
            result = emitCcx(rewriter, module, location, operands[0], operands[1], operands[2]);
            break;
        case GateKind::DecomposeMcx:
            if (operands.size() == 2) {
                result = emitCnot(
                    rewriter, module, location, operands[0], operands[1]);
            } else if (operands.size() == 3) {
                result = emitCcx(
                    rewriter,
                    module,
                    location,
                    operands[0],
                    operands[1],
                    operands[2]);
            } else {
                operation->emitError(
                    "quantum.mcx with more than two controls requires QIR "
                    "control-array lowering and runtime support");
                return ::mlir::failure();
            }
            break;
        case GateKind::DecomposeCswap:
            result = emitCnot(rewriter, module, location, operands[2], operands[1]);
            if (::mlir::succeeded(result)) {
                result = emitCcx(rewriter, module, location, operands[0], operands[1], operands[2]);
            }
            if (::mlir::succeeded(result)) {
                result = emitCnot(rewriter, module, location, operands[2], operands[1]);
            }
            break;
        case GateKind::DirectRx:
            result = emitParametricSingleQubit(
                rewriter, module, location, "rx__body", angle, operands[0]);
            break;
        case GateKind::DirectRy:
            result = emitParametricSingleQubit(
                rewriter, module, location, "ry__body", angle, operands[0]);
            break;
        case GateKind::DirectRz:
            result = emitParametricSingleQubit(
                rewriter, module, location, "rz__body", angle, operands[0]);
            break;
        case GateKind::DirectR1:
            result = emitParametricSingleQubit(
                rewriter, module, location, "r1__body", angle, operands[0]);
            break;
        case GateKind::DecomposeCrx:
            result = emitSingleQubit(rewriter, module, location, "h__body", operands[1]);
            if (::mlir::succeeded(result)) {
                result = emitParametricSingleQubit(
                    rewriter, module, location, "rz__body", angle / 2.0, operands[1]);
            }
            if (::mlir::succeeded(result)) {
                result = emitCnot(rewriter, module, location, operands[0], operands[1]);
            }
            if (::mlir::succeeded(result)) {
                result = emitParametricSingleQubit(
                    rewriter, module, location, "rz__body", -angle / 2.0, operands[1]);
            }
            if (::mlir::succeeded(result)) {
                result = emitCnot(rewriter, module, location, operands[0], operands[1]);
            }
            if (::mlir::succeeded(result)) {
                result = emitSingleQubit(rewriter, module, location, "h__body", operands[1]);
            }
            break;
        case GateKind::DecomposeCry:
            result = emitParametricSingleQubit(
                rewriter, module, location, "ry__body", angle / 2.0, operands[1]);
            if (::mlir::succeeded(result)) {
                result = emitCnot(rewriter, module, location, operands[0], operands[1]);
            }
            if (::mlir::succeeded(result)) {
                result = emitParametricSingleQubit(
                    rewriter, module, location, "ry__body", -angle / 2.0, operands[1]);
            }
            if (::mlir::succeeded(result)) {
                result = emitCnot(rewriter, module, location, operands[0], operands[1]);
            }
            break;
        case GateKind::DecomposeCrz:
            result = emitParametricSingleQubit(
                rewriter, module, location, "rz__body", angle / 2.0, operands[1]);
            if (::mlir::succeeded(result)) {
                result = emitCnot(rewriter, module, location, operands[0], operands[1]);
            }
            if (::mlir::succeeded(result)) {
                result = emitParametricSingleQubit(
                    rewriter, module, location, "rz__body", -angle / 2.0, operands[1]);
            }
            if (::mlir::succeeded(result)) {
                result = emitCnot(rewriter, module, location, operands[0], operands[1]);
            }
            break;
        case GateKind::DecomposeCp:
            result = emitParametricSingleQubit(
                rewriter, module, location, "r1__body", angle / 2.0, operands[0]);
            if (::mlir::succeeded(result)) {
                result = emitCnot(rewriter, module, location, operands[0], operands[1]);
            }
            if (::mlir::succeeded(result)) {
                result = emitParametricSingleQubit(
                    rewriter, module, location, "r1__body", -angle / 2.0, operands[1]);
            }
            if (::mlir::succeeded(result)) {
                result = emitCnot(rewriter, module, location, operands[0], operands[1]);
            }
            if (::mlir::succeeded(result)) {
                result = emitParametricSingleQubit(
                    rewriter, module, location, "r1__body", angle / 2.0, operands[1]);
            }
            break;
        }

        if (::mlir::failed(result)) {
            return ::mlir::failure();
        }
        rewriter.eraseOp(operation);
        return ::mlir::success();
    }

private:
    GateSpec spec_;
};

struct QuantumToQIRPass final
#if LLVM_VERSION_MAJOR >= 17
    : public impl::QuantumToQIRBase<QuantumToQIRPass> {
#else
    : public QuantumToQIRBase<QuantumToQIRPass> {
#endif

    QuantumToQIRPass() = default;
    explicit QuantumToQIRPass(QIREmissionProfile requested_profile)
        : profile(requested_profile) {}

    void runOnOperation() override {
        QIRModuleInfo info;
        if (::mlir::failed(
                analyzeQuantumModuleForQIR(getOperation(), info, profile))) {
            signalPassFailure();
            return;
        }

        auto* context = &getContext();
        const auto qubit_pointer = qubitPointerType(context);
        ::mlir::LLVMTypeConverter type_converter(context);
        type_converter.addConversion(
            [qubit_pointer](rocq::quantum::QubitType) -> ::mlir::Type {
                return qubit_pointer;
            });
        type_converter.addConversion(
            [qubit_pointer](rocq::quantum::ResultType) -> ::mlir::Type {
                return qubit_pointer;
            });

        ::llvm::DenseMap<::mlir::Operation*, std::uint64_t> result_indices;
        if (profile == QIREmissionProfile::Base) {
            std::uint64_t next_result = 0;
            getOperation().walk([&](rocq::quantum::MzOp measurement) {
                result_indices.try_emplace(measurement.getOperation(), next_result++);
            });
        }

        ::mlir::RewritePatternSet patterns(context);
        patterns.add<QallocLowering>(type_converter, context, qubit_pointer);
        patterns.add<MzLowering>(
            type_converter, context, qubit_pointer, result_indices);
        for (const auto& spec : kGateSpecs) {
            patterns.add<GateLowering>(type_converter, context, spec);
        }
        ::mlir::populateFuncToLLVMConversionPatterns(type_converter, patterns);

        ::mlir::LLVMConversionTarget target(*context);
        target.addLegalOp<::mlir::ModuleOp>();
        target.addIllegalDialect<rocq::quantum::QuantumDialect>();

        if (::mlir::failed(::mlir::applyFullConversion(
                getOperation(), target, std::move(patterns)))) {
            signalPassFailure();
        }
    }

private:
    QIREmissionProfile profile = QIREmissionProfile::StaticCustom;
};

} // namespace

::mlir::LogicalResult analyzeQuantumModuleForQIR(::mlir::ModuleOp module,
                                                 QIRModuleInfo& info,
                                                 QIREmissionProfile profile,
                                                 std::string* diagnostic) {
    info = {};

    ::llvm::SmallVector<::mlir::func::FuncOp> functions;
    unsigned total_functions = 0;
    for (::mlir::Operation& operation : module.getBody()->without_terminator()) {
        if (!::llvm::isa<::mlir::func::FuncOp>(operation)) {
            return fail(
                &operation,
                ::llvm::Twine("unsupported top-level operation in QIR module: ") +
                    operation.getName().getStringRef(),
                diagnostic);
        }
    }
    for (auto function : module.getOps<::mlir::func::FuncOp>()) {
        ++total_functions;
        if (!function.isExternal()) {
            functions.push_back(function);
        }
    }
    if (functions.size() != 1 || total_functions != 1) {
        return fail(module,
                    "QIR emission requires exactly one func.func and no extra declarations",
                    diagnostic);
    }

    auto function = functions.front();
    if (function.getNumArguments() != 0 || function.getFunctionType().getNumResults() != 0) {
        return fail(function,
                    "QIR v2 static-circuit entry point must have no arguments or results",
                    diagnostic);
    }
    if (!function.getBody().hasOneBlock()) {
        return fail(function,
                    "QIR emission currently supports one straight-line entry block; "
                    "classical control flow is not implemented",
                    diagnostic);
    }

    info.entry_point = function.getSymName().str();
    bool saw_qalloc = false;
    bool measurement_phase = false;
    ::llvm::DenseSet<::mlir::Value> allocated_qubits;
    ::llvm::DenseSet<::mlir::Value> measured_qubits;
    ::llvm::StringSet<> result_labels;

    for (::mlir::Operation& operation : function.getBody().front()) {
        const auto name = operation.getName().getStringRef();
        if (name == "func.return") {
            continue;
        }
        if (name == "quantum.qalloc") {
            if (saw_qalloc) {
                return fail(&operation,
                            "QIR emission currently requires exactly one quantum.qalloc",
                            diagnostic);
            }
            saw_qalloc = true;
            auto size = operation.getAttrOfType<::mlir::IntegerAttr>("size");
            if (!size || size.getInt() <= 0) {
                return fail(&operation,
                            "quantum.qalloc requires a positive i64 'size' attribute",
                            diagnostic);
            }
            const auto count = static_cast<std::uint64_t>(size.getInt());
            if (count != operation.getNumResults()) {
                return fail(&operation,
                            "quantum.qalloc result count must equal its size attribute",
                            diagnostic);
            }
            if (count > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
                return fail(&operation,
                            "quantum.qalloc exceeds the QIR static resource limit",
                            diagnostic);
            }
            info.required_qubits = count;
            for (::mlir::Value result : operation.getResults()) {
                allocated_qubits.insert(result);
            }
            continue;
        }

        if (name == "quantum.mz") {
            if (profile != QIREmissionProfile::Base) {
                return fail(
                    &operation,
                    "quantum.mz requires profile='qir-v2-base'; the "
                    "qir-v2-static profile remains a measurement-free custom profile",
                    diagnostic);
            }
            if (operation.getNumOperands() != 1 || operation.getNumResults() != 1) {
                return fail(&operation,
                            "quantum.mz requires one qubit operand and one result",
                            diagnostic);
            }
            const auto target = operation.getOperand(0);
            if (!allocated_qubits.contains(target)) {
                return fail(&operation,
                            "quantum.mz references a value not produced by quantum.qalloc",
                            diagnostic);
            }
            if (!measured_qubits.insert(target).second) {
                return fail(&operation,
                            "QIR Base Profile forbids using a qubit after measurement",
                            diagnostic);
            }
            if (!operation.getResult(0).use_empty()) {
                return fail(
                    &operation,
                    "QIR Base Profile measurement results are output resources and "
                    "cannot drive classical or quantum operations",
                    diagnostic);
            }
            auto label = operation.getAttrOfType<::mlir::StringAttr>("registerName");
            if (!label || label.getValue().empty() ||
                label.getValue().contains('\0')) {
                return fail(&operation,
                            "quantum.mz requires a non-empty, NUL-free registerName",
                            diagnostic);
            }
            if (!result_labels.insert(label.getValue()).second) {
                return fail(&operation,
                            ::llvm::Twine("duplicate QIR output label '") +
                                label.getValue() + "'",
                            diagnostic);
            }
            if (info.required_results >=
                static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
                return fail(&operation,
                            "quantum.mz exceeds the QIR static result resource limit",
                            diagnostic);
            }
            ++info.required_results;
            info.result_labels.push_back(label.getValue().str());
            measurement_phase = true;
            continue;
        }

        const bool is_quantum_operation =
#if LLVM_VERSION_MAJOR >= 18
            name.starts_with("quantum.");
#else
            name.startswith("quantum.");
#endif
        if (!is_quantum_operation) {
            return fail(&operation,
                        ::llvm::Twine("unsupported operation in the static quantum entry block: ") + name,
                        diagnostic);
        }

        const auto* spec = findGateSpec(name);
        if (!spec) {
            return fail(&operation,
                        ::llvm::Twine("unsupported quantum operation for QIR v2 emission: ") + name,
                        diagnostic);
        }
        if (profile == QIREmissionProfile::Base && measurement_phase) {
            return fail(
                &operation,
                "QIR Base Profile requires every unitary gate to precede all measurements",
                diagnostic);
        }
        if (spec->kind == GateKind::DecomposeMcx &&
            operation.getNumOperands() > 3) {
            return fail(
                &operation,
                "quantum.mcx with more than two controls requires QIR "
                "control-array lowering and runtime support",
                diagnostic);
        }
        const bool has_valid_arity =
            spec->kind == GateKind::DecomposeMcx
                ? operation.getNumOperands() >= spec->arity
                : operation.getNumOperands() == spec->arity;
        if (!has_valid_arity) {
            return fail(&operation,
                         ::llvm::Twine("operation '") + name +
                             "' has the wrong qubit operand arity",
                        diagnostic);
        }

        ::llvm::DenseSet<::mlir::Value> distinct;
        for (::mlir::Value operand : operation.getOperands()) {
            if (!allocated_qubits.contains(operand)) {
                return fail(&operation,
                            ::llvm::Twine("operation '") + name +
                                "' references a value not produced by quantum.qalloc",
                            diagnostic);
            }
            if (!distinct.insert(operand).second) {
                return fail(&operation,
                            ::llvm::Twine("operation '") + name +
                                "' requires distinct qubit operands",
                            diagnostic);
            }
        }

        if (spec->parametrized) {
            auto angle = operation.getAttrOfType<::mlir::FloatAttr>("angle");
            if (!angle || !std::isfinite(angle.getValueAsDouble())) {
                return fail(&operation,
                            ::llvm::Twine("operation '") + name +
                                "' requires a finite f64 angle",
                            diagnostic);
            }
        }
    }

    if (!saw_qalloc) {
        return fail(function,
                    "QIR emission requires one quantum.qalloc operation",
                    diagnostic);
    }
    if (profile == QIREmissionProfile::Base && info.required_results == 0) {
        return fail(function,
                    "qir-v2-base currently requires at least one terminal quantum.mz",
                    diagnostic);
    }
    return ::mlir::success();
}

std::unique_ptr<::mlir::Pass> createQuantumToQIRPass() {
    return std::make_unique<QuantumToQIRPass>();
}

std::unique_ptr<::mlir::Pass> createQuantumToQIRPass(
    QIREmissionProfile profile) {
    return std::make_unique<QuantumToQIRPass>(profile);
}

} // namespace rocq::compiler
