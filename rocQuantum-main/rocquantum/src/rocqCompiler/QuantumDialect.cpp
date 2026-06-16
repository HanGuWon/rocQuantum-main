#include "rocquantum/Dialect/QuantumDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h" // For parsing/printing custom types

// Legacy source scaffold for the old rocquantum::quantum dialect path.
// The release build is guarded at the canonical rocqCompiler/ tree; this file is
// kept syntactically valid for audits and experiments, not as parity evidence.

namespace rocquantum {
namespace quantum {

// --- QubitType Methods ---
// (Normally in a separate Types.cpp or generated)
namespace detail {
// Define storage for QubitType. For simple types without parameters, it can be empty.
struct QubitTypeStorage : public mlir::TypeStorage {
    using KeyTy = void; // No parameters for this simple type
    bool operator==(const KeyTy &) const { return true; }
    static llvm::hash_code hashKey(const KeyTy &) { return llvm::hash_code(0); } // Dummy hash
    static KeyTy getKey() { return {}; } // Dummy key
    static mlir::TypeStorage *construct(mlir::TypeStorageAllocator &allocator, KeyTy) {
        return new (allocator.allocate<QubitTypeStorage>()) QubitTypeStorage();
    }
};
} // namespace detail

QubitType QubitType::get(mlir::MLIRContext *context) {
    return Base::get(context);
}

// --- QuantumDialect Methods ---

QuantumDialect::QuantumDialect(mlir::MLIRContext *ctx)
    : mlir::Dialect(getDialectNamespace(), ctx, mlir::TypeID::get<QuantumDialect>()) {
    // Register custom types here
    registerTypes();

    // Add operations to the dialect.
    // If using C++ Op definitions directly:
    // addOperations<
    //     AllocQubitOp,
    //     DeallocQubitOp,
    //     GenericGateOp,
    //     MeasureOp
    // >();
    // The small legacy op scaffold is registered in initialize().
}

void QuantumDialect::initialize() {
    // Register only the manually maintained legacy scaffold ops.
    addOperations<
#define GET_OP_LIST
#include "rocquantum/Dialect/QuantumOps.cpp.inc"
    >();
}


// Method to parse a custom type from its textual representation
mlir::Type QuantumDialect::parseType(mlir::DialectAsmParser &parser) const {
    llvm::StringRef keyword;
    if (parser.parseKeyword(&keyword)) return nullptr;

    if (keyword == QubitType::name) { // "quantum.qubit"
        return QubitType::get(getContext());
    }
    // Add other types here if any

    parser.emitError(parser.getNameLoc(), "unknown rocquantum.quantum type: ") << keyword;
    return nullptr;
}

// Method to print a custom type to its textual representation
void QuantumDialect::printType(mlir::Type type, mlir::DialectAsmPrinter &printer) const {
    if (type.isa<QubitType>()) {
        printer << QubitType::name;
        return;
    }
    // Add other types here
    llvm_unreachable("Unhandled rocquantum.quantum type");
}

void QuantumDialect::registerTypes() {
    addTypes<QubitType>();
}


// --- Manual C++ Op Definitions ---
// These are intentionally minimal and remain outside the release compiler path.

void AllocQubitOp::build(mlir::OpBuilder &builder, mlir::OperationState &state) {
    state.addTypes(QubitType::get(builder.getContext()));
}

void AllocQubitOp::getEffects(
    llvm::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>> &effects) {
    effects.emplace_back(
        mlir::MemoryEffects::Allocate::get(),
        mlir::SideEffects::DefaultResource::get());
}

void DeallocQubitOp::build(
    mlir::OpBuilder &builder,
    mlir::OperationState &state,
    mlir::Value qubit) {
    state.addOperands(qubit);
}

void DeallocQubitOp::getEffects(
    llvm::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>> &effects) {
    effects.emplace_back(mlir::MemoryEffects::Free::get(), getQubit());
}

void GenericGateOp::build(
    mlir::OpBuilder &builder,
    mlir::OperationState &state,
    llvm::StringRef gate_name,
    mlir::ValueRange qubits) {
    state.addAttribute("gate_name", builder.getStringAttr(gate_name));
    state.addOperands(qubits);
}

llvm::StringRef GenericGateOp::getGateName() {
    return getOperation()->getAttr("gate_name").cast<mlir::StringAttr>().getValue();
}

void GenericGateOp::setGateName(llvm::StringRef name) {
    getOperation()->setAttr("gate_name", mlir::StringAttr::get(getContext(), name));
}

void MeasureOp::build(
    mlir::OpBuilder &builder,
    mlir::OperationState &state,
    mlir::Value qubit) {
    state.addOperands(qubit);
    state.addTypes(builder.getI1Type());
}

void MeasureOp::getEffects(
    llvm::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>> &effects) {
    effects.emplace_back(
        mlir::MemoryEffects::Read::get(),
        mlir::SideEffects::DefaultResource::get());
    effects.emplace_back(
        mlir::MemoryEffects::Write::get(),
        mlir::SideEffects::DefaultResource::get());
}

} // namespace quantum
} // namespace rocquantum
