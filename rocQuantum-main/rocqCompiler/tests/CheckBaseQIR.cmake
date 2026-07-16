if(NOT DEFINED TRANSLATOR OR NOT DEFINED INPUT OR NOT DEFINED OUTPUT)
  message(FATAL_ERROR "TRANSLATOR, INPUT, and OUTPUT are required")
endif()

execute_process(
  COMMAND "${TRANSLATOR}" --profile=qir-v2-base "${INPUT}"
  OUTPUT_FILE "${OUTPUT}"
  ERROR_VARIABLE translate_error
  RESULT_VARIABLE translate_result
)
if(NOT translate_result EQUAL 0)
  message(FATAL_ERROR "rocq-translate Base Profile emission failed: ${translate_error}")
endif()

file(READ "${OUTPUT}" qir)
foreach(required
    "define i64 @bell_base_profile()"
    "__quantum__rt__initialize(ptr null)"
    "__quantum__qis__h__body"
    "__quantum__qis__cnot__body"
    "__quantum__qis__mz__body"
    "ptr writeonly"
    "irreversible"
    "__quantum__rt__result_record_output"
    "qir_profiles\"=\"base_profile"
    "output_labeling_schema\"=\"schema_id"
    "required_num_qubits\"=\"2"
    "required_num_results\"=\"2"
    "dynamic_qubit_management"
    "dynamic_result_management"
    "entry:"
    "body:"
    "measurements:"
    "output:"
    "ret i64 0")
  string(FIND "${qir}" "${required}" position)
  if(position EQUAL -1)
    message(FATAL_ERROR "Base Profile QIR output is missing '${required}'")
  endif()
endforeach()

foreach(forbidden
    "qir_profiles\"=\"custom"
    "Debug Info Version"
    "!quantum."
    "\"quantum.")
  string(FIND "${qir}" "${forbidden}" position)
  if(NOT position EQUAL -1)
    message(FATAL_ERROR "Base Profile QIR unexpectedly contains '${forbidden}'")
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
  message(FATAL_ERROR "llvm-as rejected Base Profile QIR: ${llvm_as_error}")
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
  message(FATAL_ERROR "opt verifier rejected Base Profile QIR: ${opt_error}")
endif()
