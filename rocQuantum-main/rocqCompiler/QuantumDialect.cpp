#include "QuantumDialect.h"
#include "QuantumOps.h"
#include "QuantumTypes.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include "QuantumDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "QuantumTypes.cpp.inc"

#define GET_OP_CLASSES
#include "QuantumOps.cpp.inc"

namespace rocq::quantum {

void QuantumDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "QuantumTypes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "QuantumOps.cpp.inc"
      >();
}

} // namespace rocq::quantum
