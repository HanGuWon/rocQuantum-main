#include "SimulatorDialect.h"
#include "SimulatorOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include "SimulatorDialect.cpp.inc"

#define GET_OP_CLASSES
#include "SimulatorOps.cpp.inc"

namespace rocq::sim {

void SimulatorDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "SimulatorOps.cpp.inc"
      >();
}

} // namespace rocq::sim
