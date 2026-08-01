#ifndef ROCQ_COMPILER_QUANTUM_OPS_H
#define ROCQ_COMPILER_QUANTUM_OPS_H

#include "QuantumDialect.h"
#include "QuantumTypes.h"
#if __has_include("mlir/Bytecode/BytecodeOpInterface.h")
#include "mlir/Bytecode/BytecodeOpInterface.h"
#endif
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "QuantumOps.h.inc"

#endif // ROCQ_COMPILER_QUANTUM_OPS_H
