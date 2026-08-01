#include "rocqCompiler/CompilerArtifacts.h"

#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

namespace {

using rocq::compiler::ArtifactKind;
using rocq::compiler::CompilerArtifact;
using rocq::compiler::CompilerArtifactOptions;

std::unique_ptr<::llvm::Module> makeQirModule(
    ::llvm::LLVMContext& context,
    const std::string& profile = "custom") {
    auto module = std::make_unique<::llvm::Module>("artifact_fixture", context);
    module->setSourceFileName("artifact_fixture.qir");

    auto* void_type = ::llvm::Type::getVoidTy(context);
    auto* qubit_type = ::llvm::PointerType::get(context, /*AddressSpace=*/0);
    auto* qis_type = ::llvm::FunctionType::get(
        void_type, {qubit_type}, /*isVarArg=*/false);
    auto* h = ::llvm::Function::Create(
        qis_type,
        ::llvm::Function::ExternalLinkage,
        "__quantum__qis__h__body",
        *module);

    auto* entry_type = ::llvm::FunctionType::get(
        void_type, /*isVarArg=*/false);
    auto* entry = ::llvm::Function::Create(
        entry_type,
        ::llvm::Function::ExternalLinkage,
        "artifact_fixture",
        *module);
    entry->addFnAttr("entry_point");
    entry->addFnAttr("qir_profiles", profile);
    entry->addFnAttr("required_num_qubits", "1");
    entry->addFnAttr("required_num_results", "0");

    auto* block = ::llvm::BasicBlock::Create(context, "entry", entry);
    ::llvm::IRBuilder<> builder(block);
    auto* index = ::llvm::ConstantInt::get(
        ::llvm::Type::getInt64Ty(context), 0);
    auto* qubit = ::llvm::ConstantExpr::getIntToPtr(index, qubit_type);
    builder.CreateCall(h, {qubit});
    builder.CreateRetVoid();
    return module;
}

CompilerArtifact emitFresh(const CompilerArtifactOptions& options,
                           const std::string& profile = "custom") {
    ::llvm::LLVMContext context;
    return rocq::compiler::emitCompilerArtifact(
        makeQirModule(context, profile), options);
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Callback>
void requireThrowsContaining(Callback&& callback, const std::string& needle) {
    try {
        callback();
    } catch (const std::exception& error) {
        if (std::string(error.what()).find(needle) != std::string::npos) {
            return;
        }
        throw std::runtime_error(
            "exception did not contain '" + needle + "': " + error.what());
    }
    throw std::runtime_error("expected exception containing: " + needle);
}

std::string asString(const std::vector<std::uint8_t>& bytes) {
    return {reinterpret_cast<const char*>(bytes.data()), bytes.size()};
}

} // namespace

int main() {
    try {
        const auto ir = emitFresh({ArtifactKind::LlvmIr, 0});
        const auto ir_repeat = emitFresh({ArtifactKind::LlvmIr, 0});
        require(ir.bytes == ir_repeat.bytes,
                "LLVM IR emission was not byte deterministic");
        require(ir.target_triple.empty(),
                "target-neutral LLVM IR unexpectedly acquired a host triple");
        require(asString(ir.bytes).find("__quantum__qis__h__body") !=
                    std::string::npos,
                "LLVM IR omitted its unresolved QIS declaration");

        for (unsigned level = 0; level <= 3; ++level) {
            const auto bitcode = emitFresh(
                {ArtifactKind::LlvmBitcode, level});
            const auto repeated = emitFresh(
                {ArtifactKind::LlvmBitcode, level});
            require(bitcode.bytes == repeated.bytes,
                    "LLVM bitcode emission was not byte deterministic at -O" +
                        std::to_string(level));
            require(bitcode.bytes.size() >= 4 && bitcode.bytes[0] == 'B' &&
                        bitcode.bytes[1] == 'C' && bitcode.bytes[2] == 0xc0 &&
                        bitcode.bytes[3] == 0xde,
                    "LLVM bitcode did not have the raw bitcode magic");
        }

        const auto object = emitFresh({ArtifactKind::HostObject, 2});
        const auto object_repeat = emitFresh({ArtifactKind::HostObject, 2});
        require(object.bytes == object_repeat.bytes,
                "host relocatable object emission was not byte deterministic");
        require(!object.target_triple.empty() &&
                    object.target_triple == rocq::compiler::hostTargetTriple(),
                "host object reported the wrong target triple");

        std::mutex parallel_mutex;
        std::vector<std::vector<std::uint8_t>> parallel_objects;
        std::vector<std::string> parallel_errors;
        std::vector<std::thread> object_workers;
        for (unsigned index = 0; index < 4; ++index) {
            object_workers.emplace_back([&] {
                try {
                    auto emitted = emitFresh({ArtifactKind::HostObject, 2});
                    std::lock_guard lock(parallel_mutex);
                    parallel_objects.push_back(std::move(emitted.bytes));
                } catch (const std::exception& error) {
                    std::lock_guard lock(parallel_mutex);
                    parallel_errors.push_back(error.what());
                }
            });
        }
        for (auto& worker : object_workers) {
            worker.join();
        }
        require(parallel_errors.empty() && parallel_objects.size() == 4,
                "parallel host object emission failed");
        for (const auto& parallel_object : parallel_objects) {
            require(parallel_object == object.bytes,
                    "parallel host object emission was not deterministic");
        }

        require(rocq::compiler::parseArtifactKind("llvm-ir") ==
                    ArtifactKind::LlvmIr,
                "llvm-ir format parsing failed");
        require(rocq::compiler::parseArtifactKind("llvm-bc") ==
                    ArtifactKind::LlvmBitcode,
                "llvm-bc format parsing failed");
        require(rocq::compiler::parseArtifactKind("object") ==
                    ArtifactKind::HostObject,
                "object format parsing failed");
        requireThrowsContaining(
            [] { (void)rocq::compiler::parseArtifactKind("executable"); },
            "unknown artifact format");
        requireThrowsContaining(
            [] {
                (void)rocq::compiler::emitCompilerArtifact(
                    nullptr, {ArtifactKind::LlvmIr, 0});
            },
            "non-null");
        requireThrowsContaining(
            [] { (void)emitFresh({ArtifactKind::LlvmIr, 4}); },
            "between 0 and 3");
        (void)emitFresh(
            {ArtifactKind::LlvmBitcode, 0}, "base_profile");
        requireThrowsContaining(
            [] {
                (void)emitFresh(
                    {ArtifactKind::LlvmIr, 1}, "base_profile");
            },
            "Base Profile LLVM IR and bitcode artifacts require -O0");
        requireThrowsContaining(
            [] {
                (void)emitFresh(
                    {ArtifactKind::LlvmBitcode, 3}, "base_profile");
            },
            "Base Profile LLVM IR and bitcode artifacts require -O0");
        const auto base_object = emitFresh(
            {ArtifactKind::HostObject, 2}, "base_profile");
        require(!base_object.bytes.empty(),
                "Base Profile host object contract unexpectedly rejected -O2");

        std::cout << "rocq compiler artifact smoke tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
