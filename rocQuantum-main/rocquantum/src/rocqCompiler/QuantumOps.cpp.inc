// Legacy source scaffold for the old rocquantum::quantum dialect path.
//
// This file is included from QuantumDialect.cpp with GET_OP_LIST defined.
// Release builds use the guarded canonical rocqCompiler/ source tree instead.
#ifdef GET_OP_LIST
#undef GET_OP_LIST
AllocQubitOp,
DeallocQubitOp,
GenericGateOp,
MeasureOp
#endif
