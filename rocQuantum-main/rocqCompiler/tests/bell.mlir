// A CPU-only compiler fixture: static allocation, fixed and parameter gates.
module {
  func.func @bell_with_rotation() {
    %q0, %q1 = "quantum.qalloc"() {size = 2 : i64} : () -> (!quantum.qubit, !quantum.qubit)
    "quantum.h"(%q0) : (!quantum.qubit) -> ()
    "quantum.cnot"(%q0, %q1) : (!quantum.qubit, !quantum.qubit) -> ()
    "quantum.rx"(%q1) {angle = 2.500000e-01 : f64} : (!quantum.qubit) -> ()
    return
  }
}
