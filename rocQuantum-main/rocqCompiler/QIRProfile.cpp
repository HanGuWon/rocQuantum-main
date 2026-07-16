#include "QIRProfile.h"

#include <cstdint>
#include <optional>
#include <string>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/raw_ostream.h"

namespace rocq::compiler {
namespace {

constexpr ::llvm::StringLiteral kMeasureBody = "__quantum__qis__mz__body";
constexpr ::llvm::StringLiteral kInitialize = "__quantum__rt__initialize";
constexpr ::llvm::StringLiteral kRecordResult =
    "__quantum__rt__result_record_output";

bool fail(const ::llvm::Twine& message, std::string& diagnostic) {
    diagnostic = message.str();
    return false;
}

::llvm::Function* getOrCreateDeclaration(::llvm::Module& module,
                                         ::llvm::StringRef name,
                                         ::llvm::FunctionType* type,
                                         std::string& diagnostic) {
    if (auto* function = module.getFunction(name)) {
        if (function->getFunctionType() != type) {
            fail(::llvm::Twine("conflicting QIR declaration for '") + name + "'",
                 diagnostic);
            return nullptr;
        }
        return function;
    }
    return ::llvm::Function::Create(
        type, ::llvm::GlobalValue::ExternalLinkage, name, module);
}

::llvm::CallInst* cloneCall(::llvm::IRBuilder<>& builder,
                            const ::llvm::CallInst& source) {
    ::llvm::SmallVector<::llvm::Value*> arguments;
    arguments.reserve(source.arg_size());
    for (const ::llvm::Use& argument : source.args()) {
        arguments.push_back(argument.get());
    }
    auto* call = builder.CreateCall(
        source.getFunctionType(), source.getCalledOperand(), arguments);
    call->setCallingConv(source.getCallingConv());
    call->setTailCallKind(source.getTailCallKind());
    call->setAttributes(source.getAttributes());
    return call;
}

::llvm::Constant* staticPointer(::llvm::LLVMContext& context,
                                std::uint64_t index) {
    auto* integer = ::llvm::ConstantInt::get(
        ::llvm::Type::getInt64Ty(context), index);
    return ::llvm::ConstantExpr::getIntToPtr(
        integer, ::llvm::PointerType::get(context, 0));
}

::llvm::Constant* createLabelPointer(::llvm::Module& module,
                                     ::llvm::StringRef label,
                                     std::uint64_t index,
                                     std::string& diagnostic) {
    auto& context = module.getContext();
    const std::string global_name =
        "__rocq_result_label_" + std::to_string(index);
    if (module.getNamedGlobal(global_name)) {
        fail(::llvm::Twine("duplicate generated QIR label global '") +
                 global_name + "'",
             diagnostic);
        return nullptr;
    }

    auto* data = ::llvm::ConstantDataArray::getString(
        context, label, /*AddNull=*/true);
    auto* global = new ::llvm::GlobalVariable(
        module,
        data->getType(),
        /*isConstant=*/true,
        ::llvm::GlobalValue::PrivateLinkage,
        data,
        global_name);
    global->setUnnamedAddr(::llvm::GlobalValue::UnnamedAddr::Global);
    global->setAlignment(::llvm::Align(1));

    auto* zero = ::llvm::ConstantInt::get(::llvm::Type::getInt32Ty(context), 0);
    ::llvm::Constant* indices[] = {zero, zero};
    return ::llvm::ConstantExpr::getInBoundsGetElementPtr(
        data->getType(), global, indices);
}

bool shapeBaseProfile(::llvm::Module& module,
                      const QIRModuleInfo& info,
                      std::string& diagnostic) {
    if (info.result_labels.size() != info.required_results) {
        return fail("QIR result label count does not match required_num_results",
                    diagnostic);
    }
    auto* lowered_entry = module.getFunction(info.entry_point);
    if (!lowered_entry || lowered_entry->isDeclaration()) {
        return fail(::llvm::Twine("translated entry point '") + info.entry_point +
                        "' was not found",
                    diagnostic);
    }
    if (!lowered_entry->arg_empty() ||
        !lowered_entry->getReturnType()->isVoidTy() ||
        lowered_entry->size() != 1) {
        return fail(
            "QIR Base Profile shaping requires one void, no-argument LLVM entry block",
            diagnostic);
    }

    ::llvm::SmallVector<::llvm::CallInst*> body_calls;
    ::llvm::SmallVector<::llvm::CallInst*> measurement_calls;
    bool measurement_phase = false;
    for (::llvm::Instruction& instruction : lowered_entry->front()) {
        if (::llvm::isa<::llvm::ReturnInst>(instruction)) {
            if (!instruction.isTerminator() ||
                &instruction != lowered_entry->front().getTerminator()) {
                return fail("unexpected return in the lowered QIR entry block",
                            diagnostic);
            }
            continue;
        }
        auto* call = ::llvm::dyn_cast<::llvm::CallInst>(&instruction);
        if (!call || call->isIndirectCall() || !call->getCalledFunction()) {
            return fail(
                "QIR Base Profile lowering produced a non-call instruction or indirect call",
                diagnostic);
        }
        const auto name = call->getCalledFunction()->getName();
        if (name == kMeasureBody) {
            measurement_phase = true;
            measurement_calls.push_back(call);
            continue;
        }
        if (!name.starts_with("__quantum__qis__")) {
            return fail(::llvm::Twine("unexpected call before Base Profile shaping: ") +
                            name,
                        diagnostic);
        }
        if (measurement_phase) {
            return fail(
                "QIR Base Profile lowering placed a unitary call after measurement",
                diagnostic);
        }
        body_calls.push_back(call);
    }
    if (measurement_calls.size() != info.required_results) {
        return fail("lowered measurement count does not match required_num_results",
                    diagnostic);
    }

    auto& context = module.getContext();
    auto* void_type = ::llvm::Type::getVoidTy(context);
    auto* i64_type = ::llvm::Type::getInt64Ty(context);
    auto* pointer_type = ::llvm::PointerType::get(context, 0);
    auto* initialize_type =
        ::llvm::FunctionType::get(void_type, {pointer_type}, false);
    auto* record_type = ::llvm::FunctionType::get(
        void_type, {pointer_type, pointer_type}, false);
    auto* initialize = getOrCreateDeclaration(
        module, kInitialize, initialize_type, diagnostic);
    auto* record = getOrCreateDeclaration(
        module, kRecordResult, record_type, diagnostic);
    auto* measurement = module.getFunction(kMeasureBody);
    auto* measurement_type = ::llvm::FunctionType::get(
        void_type, {pointer_type, pointer_type}, false);
    if (!initialize || !record || !measurement ||
        measurement->getFunctionType() != measurement_type) {
        if (diagnostic.empty()) {
            diagnostic = "missing or invalid __quantum__qis__mz__body declaration";
        }
        return false;
    }
    measurement->addFnAttr("irreversible");
    measurement->addParamAttr(1, ::llvm::Attribute::WriteOnly);

    const std::string old_name = info.entry_point + ".rocq.lowered";
    lowered_entry->setName(old_name);
    auto* entry_type = ::llvm::FunctionType::get(i64_type, false);
    auto* entry = ::llvm::Function::Create(
        entry_type,
        lowered_entry->getLinkage(),
        lowered_entry->getAddressSpace(),
        info.entry_point,
        &module);
    entry->setCallingConv(lowered_entry->getCallingConv());
    entry->setVisibility(lowered_entry->getVisibility());
    entry->setDSOLocal(lowered_entry->isDSOLocal());

    auto* initialize_block = ::llvm::BasicBlock::Create(context, "entry", entry);
    auto* body_block = ::llvm::BasicBlock::Create(context, "body", entry);
    auto* measurements_block =
        ::llvm::BasicBlock::Create(context, "measurements", entry);
    auto* output_block = ::llvm::BasicBlock::Create(context, "output", entry);

    ::llvm::IRBuilder<> initialize_builder(initialize_block);
    initialize_builder.CreateCall(
        initialize, {::llvm::ConstantPointerNull::get(pointer_type)});
    initialize_builder.CreateBr(body_block);

    ::llvm::IRBuilder<> body_builder(body_block);
    for (const auto* source : body_calls) {
        cloneCall(body_builder, *source);
    }
    body_builder.CreateBr(measurements_block);

    ::llvm::IRBuilder<> measurements_builder(measurements_block);
    for (const auto* source : measurement_calls) {
        auto* call = cloneCall(measurements_builder, *source);
        call->addParamAttr(1, ::llvm::Attribute::WriteOnly);
    }
    measurements_builder.CreateBr(output_block);

    ::llvm::IRBuilder<> output_builder(output_block);
    for (std::uint64_t index = 0; index < info.required_results; ++index) {
        auto* label = createLabelPointer(
            module, info.result_labels[index], index, diagnostic);
        if (!label) {
            entry->eraseFromParent();
            lowered_entry->setName(info.entry_point);
            return false;
        }
        output_builder.CreateCall(
            record, {staticPointer(context, index), label});
    }
    output_builder.CreateRet(::llvm::ConstantInt::get(i64_type, 0));

    lowered_entry->eraseFromParent();
    return true;
}

void setEntryAttributes(::llvm::Function& entry,
                        const QIRModuleInfo& info,
                        QIREmissionProfile profile) {
    entry.addFnAttr("entry_point");
    entry.addFnAttr(
        "qir_profiles",
        profile == QIREmissionProfile::Base ? "base_profile" : "custom");
    entry.addFnAttr(
        "required_num_qubits", std::to_string(info.required_qubits));
    entry.addFnAttr(
        "required_num_results", std::to_string(info.required_results));
    if (profile == QIREmissionProfile::Base) {
        entry.addFnAttr("output_labeling_schema", "schema_id");
    }
}

void addQirModuleFlags(::llvm::Module& module) {
    auto& context = module.getContext();
    auto add_integer_flag = [&](::llvm::Module::ModFlagBehavior behavior,
                                const char* name,
                                unsigned bits,
                                std::uint64_t value) {
        auto* constant = ::llvm::ConstantInt::get(
            ::llvm::IntegerType::get(context, bits), value);
        module.addModuleFlag(
            behavior, name, ::llvm::ConstantAsMetadata::get(constant));
    };

    add_integer_flag(::llvm::Module::Error, "qir_major_version", 32, 2);
    add_integer_flag(::llvm::Module::Max, "qir_minor_version", 32, 0);
    add_integer_flag(::llvm::Module::Error, "dynamic_qubit_management", 1, 0);
    add_integer_flag(::llvm::Module::Error, "dynamic_result_management", 1, 0);
}

std::optional<std::uint64_t> staticResourceId(const ::llvm::Value* value) {
    if (::llvm::isa<::llvm::ConstantPointerNull>(value)) {
        return 0;
    }
    auto* expression = ::llvm::dyn_cast<::llvm::ConstantExpr>(value);
    if (!expression || expression->getOpcode() != ::llvm::Instruction::IntToPtr) {
        return std::nullopt;
    }
    auto* integer = ::llvm::dyn_cast<::llvm::ConstantInt>(expression->getOperand(0));
    if (!integer || integer->getValue().getActiveBits() > 64) {
        return std::nullopt;
    }
    return integer->getZExtValue();
}

bool hasStringAttribute(const ::llvm::Function& function,
                        ::llvm::StringRef name,
                        ::llvm::StringRef value) {
    auto attribute = function.getFnAttribute(name);
    return attribute.isStringAttribute() &&
           attribute.getValueAsString() == value;
}

bool hasIntegerModuleFlag(const ::llvm::Module& module,
                          ::llvm::StringRef name,
                          unsigned bits,
                          std::uint64_t value) {
    auto* metadata = module.getModuleFlag(name);
    auto* constant_metadata =
        ::llvm::dyn_cast_or_null<::llvm::ConstantAsMetadata>(metadata);
    auto* integer = constant_metadata
                        ? ::llvm::dyn_cast<::llvm::ConstantInt>(
                              constant_metadata->getValue())
                        : nullptr;
    return integer && integer->getBitWidth() == bits &&
           integer->getZExtValue() == value;
}

bool verifyBaseProfile(const ::llvm::Module& module,
                       const QIRModuleInfo& info,
                       std::string& diagnostic) {
    const auto* entry = module.getFunction(info.entry_point);
    if (!entry || entry->isDeclaration() || !entry->arg_empty() ||
        !entry->getReturnType()->isIntegerTy(64) || entry->size() != 4) {
        return fail(
            "QIR Base Profile entry must be a no-argument i64 function with four blocks",
            diagnostic);
    }
    if (!entry->hasFnAttribute("entry_point") ||
        !hasStringAttribute(*entry, "qir_profiles", "base_profile") ||
        !hasStringAttribute(*entry,
                            "required_num_qubits",
                            std::to_string(info.required_qubits)) ||
        !hasStringAttribute(*entry,
                            "required_num_results",
                            std::to_string(info.required_results)) ||
        !hasStringAttribute(*entry, "output_labeling_schema", "schema_id")) {
        return fail("QIR Base Profile entry attributes are incomplete", diagnostic);
    }

    auto block = entry->begin();
    const auto* initialize_block = &*block++;
    const auto* body_block = &*block++;
    const auto* measurements_block = &*block++;
    const auto* output_block = &*block;

    if (initialize_block->size() != 2) {
        return fail("QIR Base Profile initialization block has an invalid shape",
                    diagnostic);
    }
    const auto* initialize_call =
        ::llvm::dyn_cast<::llvm::CallInst>(&initialize_block->front());
    const auto* initialize_branch =
        ::llvm::dyn_cast<::llvm::BranchInst>(initialize_block->getTerminator());
    if (!initialize_call || !initialize_call->getCalledFunction() ||
        initialize_call->getCalledFunction()->getName() != kInitialize ||
        initialize_call->arg_size() != 1 ||
        !::llvm::isa<::llvm::ConstantPointerNull>(
            initialize_call->getArgOperand(0)) ||
        !initialize_branch || initialize_branch->isConditional() ||
        initialize_branch->getSuccessor(0) != body_block) {
        return fail(
            "QIR Base Profile entry must begin with initialize(null) and branch to body",
            diagnostic);
    }

    for (const ::llvm::Instruction& instruction : *body_block) {
        if (&instruction == body_block->getTerminator()) {
            continue;
        }
        const auto* call = ::llvm::dyn_cast<::llvm::CallInst>(&instruction);
        if (!call || !call->getCalledFunction() ||
            !call->getCalledFunction()->getName().starts_with("__quantum__qis__") ||
            call->getCalledFunction()->getName() == kMeasureBody ||
            !call->getType()->isVoidTy()) {
            return fail("QIR Base Profile body contains a non-unitary instruction",
                        diagnostic);
        }
        for (const ::llvm::Use& argument : call->args()) {
            if (!::llvm::isa<::llvm::Constant>(argument.get())) {
                return fail("QIR Base Profile QIS arguments must be constants",
                            diagnostic);
            }
        }
    }
    const auto* body_branch =
        ::llvm::dyn_cast<::llvm::BranchInst>(body_block->getTerminator());
    if (!body_branch || body_branch->isConditional() ||
        body_branch->getSuccessor(0) != measurements_block) {
        return fail("QIR Base Profile body must branch to measurements", diagnostic);
    }

    std::uint64_t measurement_count = 0;
    for (const ::llvm::Instruction& instruction : *measurements_block) {
        if (&instruction == measurements_block->getTerminator()) {
            continue;
        }
        const auto* call = ::llvm::dyn_cast<::llvm::CallInst>(&instruction);
        if (!call || !call->getCalledFunction() ||
            call->getCalledFunction()->getName() != kMeasureBody ||
            call->arg_size() != 2) {
            return fail("QIR Base Profile measurements block contains an invalid call",
                        diagnostic);
        }
        const auto qubit = staticResourceId(call->getArgOperand(0));
        const auto result = staticResourceId(call->getArgOperand(1));
        if (!qubit || *qubit >= info.required_qubits || !result ||
            *result != measurement_count) {
            return fail("QIR Base Profile measurement resource ids are invalid",
                        diagnostic);
        }
        ++measurement_count;
    }
    const auto* measurement_branch =
        ::llvm::dyn_cast<::llvm::BranchInst>(measurements_block->getTerminator());
    if (measurement_count != info.required_results || !measurement_branch ||
        measurement_branch->isConditional() ||
        measurement_branch->getSuccessor(0) != output_block) {
        return fail("QIR Base Profile measurement block has an invalid shape",
                    diagnostic);
    }

    std::uint64_t output_count = 0;
    for (const ::llvm::Instruction& instruction : *output_block) {
        if (&instruction == output_block->getTerminator()) {
            continue;
        }
        const auto* call = ::llvm::dyn_cast<::llvm::CallInst>(&instruction);
        if (!call || !call->getCalledFunction() ||
            call->getCalledFunction()->getName() != kRecordResult ||
            call->arg_size() != 2) {
            return fail("QIR Base Profile output block contains an invalid call",
                        diagnostic);
        }
        const auto result = staticResourceId(call->getArgOperand(0));
        if (!result || *result != output_count ||
            ::llvm::isa<::llvm::ConstantPointerNull>(call->getArgOperand(1))) {
            return fail("QIR Base Profile output resources or labels are invalid",
                        diagnostic);
        }
        ++output_count;
    }
    const auto* result =
        ::llvm::dyn_cast<::llvm::ReturnInst>(output_block->getTerminator());
    const auto* exit_code = result
                                ? ::llvm::dyn_cast<::llvm::ConstantInt>(
                                      result->getReturnValue())
                                : nullptr;
    if (output_count != info.required_results || !exit_code ||
        !exit_code->isZero()) {
        return fail("QIR Base Profile output block must return i64 zero", diagnostic);
    }

    const auto* measurement = module.getFunction(kMeasureBody);
    if (!measurement || !measurement->hasFnAttribute("irreversible") ||
        !measurement->hasParamAttribute(1, ::llvm::Attribute::WriteOnly)) {
        return fail(
            "QIR Base Profile measurement declaration lacks irreversible/writeonly",
            diagnostic);
    }
    if (!hasIntegerModuleFlag(module, "qir_major_version", 32, 2) ||
        !hasIntegerModuleFlag(module, "qir_minor_version", 32, 0) ||
        !hasIntegerModuleFlag(module, "dynamic_qubit_management", 1, 0) ||
        !hasIntegerModuleFlag(module, "dynamic_result_management", 1, 0)) {
        return fail("QIR v2 module flags are incomplete or invalid", diagnostic);
    }
    return true;
}

void removeUnusedSupportDeclarations(::llvm::Module& module) {
    ::llvm::SmallVector<::llvm::Function*> unused;
    for (auto& function : module.functions()) {
        if (function.isDeclaration() && function.use_empty() &&
            !function.getName().starts_with("__quantum__")) {
            unused.push_back(&function);
        }
    }
    for (auto* function : unused) {
        function->eraseFromParent();
    }
}

} // namespace

bool parseQIREmissionProfile(const std::string& name,
                             QIREmissionProfile& profile) {
    if (name == "qir-v2-static") {
        profile = QIREmissionProfile::StaticCustom;
        return true;
    }
    if (name == "qir-v2-base") {
        profile = QIREmissionProfile::Base;
        return true;
    }
    return false;
}

bool finalizeAndVerifyQIRModule(::llvm::Module& module,
                                const QIRModuleInfo& info,
                                QIREmissionProfile profile,
                                std::string& diagnostic) {
    diagnostic.clear();
    if (profile == QIREmissionProfile::StaticCustom &&
        info.required_results != 0) {
        return fail("qir-v2-static does not permit measurement results", diagnostic);
    }
    if (profile == QIREmissionProfile::Base &&
        !shapeBaseProfile(module, info, diagnostic)) {
        return false;
    }

    auto* entry = module.getFunction(info.entry_point);
    if (!entry || entry->isDeclaration()) {
        return fail(::llvm::Twine("translated entry point '") + info.entry_point +
                        "' was not found",
                    diagnostic);
    }
    setEntryAttributes(*entry, info, profile);
    module.setModuleIdentifier(info.entry_point);
    module.setSourceFileName(info.entry_point + ".qir");

    ::llvm::StripDebugInfo(module);
    if (auto* flags = module.getNamedMetadata("llvm.module.flags")) {
        module.eraseNamedMetadata(flags);
    }
    removeUnusedSupportDeclarations(module);
    addQirModuleFlags(module);

    std::string verifier_diagnostic;
    ::llvm::raw_string_ostream verifier_stream(verifier_diagnostic);
    if (::llvm::verifyModule(module, &verifier_stream)) {
        verifier_stream.flush();
        return fail(::llvm::Twine("LLVM verifier rejected the QIR module: ") +
                        verifier_diagnostic,
                    diagnostic);
    }
    if (profile == QIREmissionProfile::Base &&
        !verifyBaseProfile(module, info, diagnostic)) {
        return false;
    }
    return true;
}

} // namespace rocq::compiler
