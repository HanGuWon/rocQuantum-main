module {
  func.func @invalid_artifact() {
    %q0 = "quantum.qalloc"() {size = 1 : i64} : () -> !quantum.qubit
    "quantum.cnot"(%q0, %q0) : (!quantum.qubit, !quantum.qubit) -> ()
    return
  }
}
