#ifndef ROCQ_COMPILER_SIMULATOR_OPS_H
#define ROCQ_COMPILER_SIMULATOR_OPS_H

#include "QuantumTypes.h"
#include "SimulatorDialect.h"
#if __has_include("mlir/Bytecode/BytecodeOpInterface.h")
#include "mlir/Bytecode/BytecodeOpInterface.h"
#endif
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "SimulatorOps.h.inc"

#endif // ROCQ_COMPILER_SIMULATOR_OPS_H
