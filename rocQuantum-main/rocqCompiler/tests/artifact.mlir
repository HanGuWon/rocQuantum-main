// GPU-independent compiler artifact fixture. The QIS declarations remain
// unresolved in host object output and are linked by a QIR runtime later.
module {
  func.func @artifact_fixture() {
    %q0, %q1 = "quantum.qalloc"() {size = 2 : i64} : () -> (!quantum.qubit, !quantum.qubit)
    "quantum.h"(%q0) : (!quantum.qubit) -> ()
    "quantum.cnot"(%q0, %q1) : (!quantum.qubit, !quantum.qubit) -> ()
    "quantum.rz"(%q1) {angle = 2.500000e-01 : f64} : (!quantum.qubit) -> ()
    return
  }
}
