// CPU-only QIR v2 Base Profile fixture: all unitary calls precede terminal
// measurements, and each selected output has a unique explicit label.
module {
  func.func @bell_base_profile() {
    %q0, %q1 = "quantum.qalloc"() {size = 2 : i64} : () -> (!quantum.qubit, !quantum.qubit)
    "quantum.h"(%q0) : (!quantum.qubit) -> ()
    "quantum.cnot"(%q0, %q1) : (!quantum.qubit, !quantum.qubit) -> ()
    %r0 = "quantum.mz"(%q0) {registerName = "bell.0"} : (!quantum.qubit) -> !quantum.result
    %r1 = "quantum.mz"(%q1) {registerName = "bell.1"} : (!quantum.qubit) -> !quantum.result
    return
  }
}
