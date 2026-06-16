// Legacy source scaffold for the old rocquantum::quantum dialect path.
//
// Release builds use the guarded canonical rocqCompiler/ source tree instead.
// Keep this file syntactically valid for source audits and experimental includes,
// but do not treat it as generated TableGen coverage or CUDA-Q-style parity.
#ifndef ROCQUANTUM_DIALECT_QUANTUMOPS_H_INC
#define ROCQUANTUM_DIALECT_QUANTUMOPS_H_INC

namespace mlir {
class OpBuilder;
class OperationState;
} // namespace mlir

namespace rocquantum {
namespace quantum {

class AllocQubitOp : public mlir::Op<
    AllocQubitOp,
    mlir::OpTrait::ZeroOperands,
    mlir::OpTrait::OneResult,
    mlir::MemoryEffectOpInterface::Trait> {
public:
    using Op::Op;
    static llvm::StringRef getOperationName() { return "quantum.alloc_qubit"; }
    static void build(mlir::OpBuilder &builder, mlir::OperationState &state);
    mlir::Value getQubit() { return getResult(0); }
    void getEffects(
        llvm::SmallVectorImpl<
            mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>> &effects);
};

class DeallocQubitOp : public mlir::Op<
    DeallocQubitOp,
    mlir::OpTrait::OneOperand,
    mlir::OpTrait::ZeroResults,
    mlir::MemoryEffectOpInterface::Trait> {
public:
    using Op::Op;
    static llvm::StringRef getOperationName() { return "quantum.dealloc_qubit"; }
    static void build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::Value qubit);
    mlir::Value getQubit() { return getOperand(0); }
    void getEffects(
        llvm::SmallVectorImpl<
            mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>> &effects);
};

class GenericGateOp : public mlir::Op<
    GenericGateOp,
    mlir::OpTrait::VariadicOperands,
    mlir::OpTrait::ZeroResults> {
public:
    using Op::Op;
    static llvm::StringRef getOperationName() { return "quantum.gate"; }
    static void build(
        mlir::OpBuilder &builder,
        mlir::OperationState &state,
        llvm::StringRef gate_name,
        mlir::ValueRange qubits);
    llvm::StringRef getGateName();
    void setGateName(llvm::StringRef name);
    mlir::Operation::operand_range getQubits() { return getOperands(); }
};

class MeasureOp : public mlir::Op<
    MeasureOp,
    mlir::OpTrait::OneOperand,
    mlir::OpTrait::OneResult,
    mlir::MemoryEffectOpInterface::Trait> {
public:
    using Op::Op;
    static llvm::StringRef getOperationName() { return "quantum.measure"; }
    static void build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::Value qubit);
    mlir::Value getQubit() { return getOperand(0); }
    mlir::Value getMeasurement() { return getOperation()->getResult(0); }
    void getEffects(
        llvm::SmallVectorImpl<
            mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>> &effects);
};

} // namespace quantum
} // namespace rocquantum

#endif // ROCQUANTUM_DIALECT_QUANTUMOPS_H_INC
