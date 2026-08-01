if(NOT DEFINED TRANSLATOR OR NOT DEFINED INPUT OR NOT DEFINED OUTPUT)
  message(FATAL_ERROR "TRANSLATOR, INPUT, and OUTPUT are required")
endif()

execute_process(
  COMMAND "${TRANSLATOR}" "${INPUT}"
  OUTPUT_FILE "${OUTPUT}"
  ERROR_VARIABLE translate_error
  RESULT_VARIABLE translate_result
)
if(NOT translate_result EQUAL 0)
  message(FATAL_ERROR "rocq-translate failed: ${translate_error}")
endif()

file(READ "${OUTPUT}" qir)
foreach(required
    "__quantum__qis__h__body"
    "__quantum__qis__cnot__body"
    "__quantum__qis__rx__body"
    "qir_major_version"
    "qir_minor_version"
    "dynamic_qubit_management"
    "dynamic_result_management"
    "required_num_qubits\"=\"2"
    "required_num_results\"=\"0"
    "qir_profiles\"=\"custom"
    "entry_point")
  string(FIND "${qir}" "${required}" position)
  if(position EQUAL -1)
    message(FATAL_ERROR "QIR output is missing '${required}'")
  endif()
endforeach()

foreach(forbidden "Debug Info Version" "!quantum." "\"quantum.")
  string(FIND "${qir}" "${forbidden}" position)
  if(NOT position EQUAL -1)
    message(FATAL_ERROR "QIR output unexpectedly contains '${forbidden}'")
  endif()
endforeach()

if(NOT DEFINED LLVM_AS OR NOT EXISTS "${LLVM_AS}")
  message(FATAL_ERROR "LLVM_AS must name the selected toolchain's llvm-as")
endif()
execute_process(
  COMMAND "${LLVM_AS}" "${OUTPUT}" -o "${OUTPUT}.bc"
  ERROR_VARIABLE llvm_as_error
  RESULT_VARIABLE llvm_as_result
)
if(NOT llvm_as_result EQUAL 0)
  message(FATAL_ERROR "llvm-as rejected generated QIR: ${llvm_as_error}")
endif()

if(NOT DEFINED LLVM_OPT OR NOT EXISTS "${LLVM_OPT}")
  message(FATAL_ERROR "LLVM_OPT must name the selected toolchain's opt")
endif()
execute_process(
  COMMAND "${LLVM_OPT}" -passes=verify -disable-output "${OUTPUT}.bc"
  ERROR_VARIABLE opt_error
  RESULT_VARIABLE opt_result
)
if(NOT opt_result EQUAL 0)
  message(FATAL_ERROR "opt verifier rejected generated QIR: ${opt_error}")
endif()
