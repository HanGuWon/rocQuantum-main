#include "rocqCompiler/CompilerArtifacts.h"

#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"

namespace {

using rocq::compiler::ArtifactKind;

bool isBaseProfileModule(const ::llvm::Module& module) {
    for (const auto& function : module.functions()) {
        const auto profile = function.getFnAttribute("qir_profiles");
        if (profile.isStringAttribute() &&
            profile.getValueAsString() == "base_profile") {
            return true;
        }
    }
    return false;
}

void validateOptimizationLevel(unsigned level) {
    if (level > 3) {
        throw std::invalid_argument(
            "compiler artifact optimization level must be between 0 and 3");
    }
}

::llvm::OptimizationLevel llvmOptimizationLevel(unsigned level) {
    switch (level) {
    case 1:
        return ::llvm::OptimizationLevel::O1;
    case 2:
        return ::llvm::OptimizationLevel::O2;
    case 3:
        return ::llvm::OptimizationLevel::O3;
    default:
        throw std::invalid_argument(
            "LLVM optimization pipeline requires a level between 1 and 3");
    }
}

::llvm::CodeGenOptLevel codeGenerationLevel(unsigned level) {
    switch (level) {
    case 0:
        return ::llvm::CodeGenOptLevel::None;
    case 1:
        return ::llvm::CodeGenOptLevel::Less;
    case 2:
        return ::llvm::CodeGenOptLevel::Default;
    case 3:
        return ::llvm::CodeGenOptLevel::Aggressive;
    default:
        throw std::invalid_argument(
            "object-code optimization level must be between 0 and 3");
    }
}

void verifyModuleOrThrow(const ::llvm::Module& module,
                         const char* stage) {
    std::string diagnostic;
    ::llvm::raw_string_ostream diagnostic_stream(diagnostic);
    if (::llvm::verifyModule(module, &diagnostic_stream)) {
        diagnostic_stream.flush();
        throw std::runtime_error(
            std::string("compiler artifact ") + stage +
            " LLVM verification failed: " + diagnostic);
    }
}

void optimizeModule(::llvm::Module& module,
                    unsigned level,
                    ::llvm::TargetMachine* target_machine) {
    if (level == 0) {
        return;
    }

    ::llvm::LoopAnalysisManager loop_analyses;
    ::llvm::FunctionAnalysisManager function_analyses;
    ::llvm::CGSCCAnalysisManager cgscc_analyses;
    ::llvm::ModuleAnalysisManager module_analyses;
    ::llvm::PassBuilder pass_builder(target_machine);

    pass_builder.registerModuleAnalyses(module_analyses);
    pass_builder.registerCGSCCAnalyses(cgscc_analyses);
    pass_builder.registerFunctionAnalyses(function_analyses);
    pass_builder.registerLoopAnalyses(loop_analyses);
    pass_builder.crossRegisterProxies(
        loop_analyses, function_analyses, cgscc_analyses, module_analyses);

    auto pipeline = pass_builder.buildPerModuleDefaultPipeline(
        llvmOptimizationLevel(level));
    pipeline.run(module, module_analyses);
}

struct NativeTargetMachine {
    std::string triple;
    std::unique_ptr<::llvm::TargetMachine> machine;
};

NativeTargetMachine createNativeTargetMachine(unsigned optimization_level) {
    static std::once_flag initialization_once;
    static std::string initialization_error;
    std::call_once(initialization_once, [] {
        if (::llvm::InitializeNativeTarget()) {
            initialization_error =
                "host object emission is unavailable: LLVM has no native target";
            return;
        }
        if (::llvm::InitializeNativeTargetAsmPrinter()) {
            initialization_error =
                "host object emission is unavailable: LLVM has no native assembler printer";
        }
    });
    if (!initialization_error.empty()) {
        throw std::runtime_error(initialization_error);
    }

    const std::string triple_text = ::llvm::Triple::normalize(
        ::llvm::sys::getDefaultTargetTriple());
    if (triple_text.empty()) {
        throw std::runtime_error(
            "host object emission failed to determine a target triple");
    }

    const ::llvm::Triple triple(triple_text);
    std::string lookup_error;
    const auto* target = ::llvm::TargetRegistry::lookupTarget(
        triple, lookup_error);
    if (!target) {
        throw std::runtime_error(
            "host object emission could not find target '" + triple_text +
            "': " + lookup_error);
    }

    ::llvm::TargetOptions target_options;
    std::unique_ptr<::llvm::TargetMachine> machine(target->createTargetMachine(
        triple,
        /*CPU=*/"generic",
        /*Features=*/"",
        target_options,
        ::llvm::Reloc::PIC_,
        std::nullopt,
        codeGenerationLevel(optimization_level)));
    if (!machine) {
        throw std::runtime_error(
            "host object emission failed to create a target machine for '" +
            triple_text + "'");
    }

    return {triple_text, std::move(machine)};
}

std::vector<std::uint8_t> copyBytes(::llvm::ArrayRef<char> source) {
    if (source.empty()) {
        return {};
    }
    const auto* begin = reinterpret_cast<const std::uint8_t*>(source.data());
    return {begin, begin + source.size()};
}

std::vector<std::uint8_t> emitLlvmIr(const ::llvm::Module& module) {
    ::llvm::SmallVector<char, 0> storage;
    ::llvm::raw_svector_ostream output(storage);
    module.print(output, nullptr);
    return copyBytes(storage);
}

std::vector<std::uint8_t> emitBitcode(const ::llvm::Module& module) {
    ::llvm::SmallVector<char, 0> storage;
    ::llvm::raw_svector_ostream output(storage);
    ::llvm::WriteBitcodeToFile(module, output);
    return copyBytes(storage);
}

std::vector<std::uint8_t> emitHostObject(
    ::llvm::Module& module,
    ::llvm::TargetMachine& target_machine) {
    ::llvm::SmallVector<char, 0> storage;
    ::llvm::raw_svector_ostream output(storage);
    ::llvm::legacy::PassManager code_generation;

    if (target_machine.addPassesToEmitFile(
            code_generation,
            output,
            /*DwoOut=*/nullptr,
            ::llvm::CodeGenFileType::ObjectFile,
            /*DisableVerify=*/false)) {
        throw std::runtime_error(
            "host target machine does not support relocatable object emission");
    }
    code_generation.run(module);
    if (storage.empty()) {
        throw std::runtime_error(
            "host target machine emitted an empty relocatable object");
    }
    return copyBytes(storage);
}

} // namespace

namespace rocq::compiler {

std::string_view artifactKindName(ArtifactKind kind) {
    switch (kind) {
    case ArtifactKind::LlvmIr:
        return "llvm-ir";
    case ArtifactKind::LlvmBitcode:
        return "llvm-bc";
    case ArtifactKind::HostObject:
        return "object";
    }
    throw std::invalid_argument("unknown compiler artifact kind");
}

std::string_view artifactFileExtension(ArtifactKind kind) {
    switch (kind) {
    case ArtifactKind::LlvmIr:
        return ".ll";
    case ArtifactKind::LlvmBitcode:
        return ".bc";
    case ArtifactKind::HostObject:
        return ".o";
    }
    throw std::invalid_argument("unknown compiler artifact kind");
}

ArtifactKind parseArtifactKind(std::string_view name) {
    if (name == "llvm-ir") {
        return ArtifactKind::LlvmIr;
    }
    if (name == "llvm-bc") {
        return ArtifactKind::LlvmBitcode;
    }
    if (name == "object") {
        return ArtifactKind::HostObject;
    }
    throw std::invalid_argument(
        "unknown artifact format '" + std::string(name) +
        "'; expected llvm-ir, llvm-bc, or object");
}

std::string hostTargetTriple() {
    return ::llvm::Triple::normalize(::llvm::sys::getDefaultTargetTriple());
}

CompilerArtifact emitCompilerArtifact(
    std::unique_ptr<::llvm::Module> module,
    const CompilerArtifactOptions& options) {
    if (!module) {
        throw std::invalid_argument(
            "compiler artifact emission requires a non-null LLVM module");
    }
    validateOptimizationLevel(options.optimization_level);
    if (isBaseProfileModule(*module) &&
        options.kind != ArtifactKind::HostObject &&
        options.optimization_level != 0) {
        throw std::invalid_argument(
            "QIR v2 Base Profile LLVM IR and bitcode artifacts require -O0; "
            "generic LLVM optimization does not preserve the required "
            "four-block control-flow contract");
    }
    verifyModuleOrThrow(*module, "input");

    std::optional<NativeTargetMachine> native_target;
    if (options.kind == ArtifactKind::HostObject) {
        native_target.emplace(
            createNativeTargetMachine(options.optimization_level));
        module->setTargetTriple(::llvm::Triple(native_target->triple));
        module->setDataLayout(native_target->machine->createDataLayout());
    }

    optimizeModule(
        *module,
        options.optimization_level,
        native_target ? native_target->machine.get() : nullptr);
    verifyModuleOrThrow(*module, "post-optimization");

    CompilerArtifact artifact;
    artifact.target_triple = module->getTargetTriple().str();
    switch (options.kind) {
    case ArtifactKind::LlvmIr:
        artifact.bytes = emitLlvmIr(*module);
        break;
    case ArtifactKind::LlvmBitcode:
        artifact.bytes = emitBitcode(*module);
        break;
    case ArtifactKind::HostObject:
        artifact.bytes = emitHostObject(
            *module, *native_target->machine);
        break;
    }

    if (artifact.bytes.empty()) {
        throw std::runtime_error(
            "compiler artifact emission produced an empty payload");
    }
    return artifact;
}

} // namespace rocq::compiler
