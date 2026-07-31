if(NOT DEFINED RUNNER OR NOT EXISTS "${RUNNER}")
  message(FATAL_ERROR "RUNNER does not name an existing rocq-run executable")
endif()
if(NOT DEFINED INPUT OR NOT EXISTS "${INPUT}")
  message(FATAL_ERROR "INPUT does not name an existing MLIR file")
endif()
if(NOT DEFINED BASE_INPUT OR NOT EXISTS "${BASE_INPUT}")
  message(FATAL_ERROR "BASE_INPUT does not name an existing MLIR file")
endif()

execute_process(
  COMMAND "${RUNNER}" "${INPUT}"
  RESULT_VARIABLE run_status
  OUTPUT_VARIABLE run_output
  ERROR_VARIABLE run_error
)
if(NOT run_status EQUAL 0)
  message(FATAL_ERROR
    "rocq-run failed for Bell input (${run_status}): ${run_error}")
endif()

string(JSON schema ERROR_VARIABLE json_error GET "${run_output}" schema)
if(json_error OR NOT schema STREQUAL "rocq-state-vector-v1")
  message(FATAL_ERROR "rocq-run returned an invalid schema: ${run_output}")
endif()
string(JSON backend ERROR_VARIABLE json_error GET "${run_output}" backend)
if(json_error OR NOT backend STREQUAL "cpu_statevec")
  message(FATAL_ERROR "rocq-run returned an invalid backend: ${run_output}")
endif()
string(JSON num_qubits ERROR_VARIABLE json_error GET "${run_output}" num_qubits)
if(json_error OR NOT num_qubits EQUAL 2)
  message(FATAL_ERROR "rocq-run returned an invalid qubit count: ${run_output}")
endif()
string(JSON amplitude_count ERROR_VARIABLE json_error
  LENGTH "${run_output}" amplitudes)
if(json_error OR NOT amplitude_count EQUAL 4)
  message(FATAL_ERROR "rocq-run returned an invalid state dimension: ${run_output}")
endif()
string(JSON amplitude_00 ERROR_VARIABLE json_error
  GET "${run_output}" amplitudes 0 real)
if(json_error OR NOT amplitude_00 MATCHES "^0\\.707106")
  message(FATAL_ERROR "rocq-run returned an invalid |00> amplitude: ${run_output}")
endif()
string(JSON amplitude_11 ERROR_VARIABLE json_error
  GET "${run_output}" amplitudes 3 real)
if(json_error OR NOT amplitude_11 MATCHES "^0\\.707106")
  message(FATAL_ERROR "rocq-run returned an invalid |11> amplitude: ${run_output}")
endif()

execute_process(
  COMMAND "${RUNNER}" -
  INPUT_FILE "${INPUT}"
  RESULT_VARIABLE stdin_status
  OUTPUT_VARIABLE stdin_output
  ERROR_VARIABLE stdin_error
)
if(NOT stdin_status EQUAL 0 OR NOT stdin_output STREQUAL run_output)
  message(FATAL_ERROR
    "rocq-run stdin execution diverged (${stdin_status}): ${stdin_error}")
endif()

execute_process(
  COMMAND "${RUNNER}" "${BASE_INPUT}"
  RESULT_VARIABLE base_status
  OUTPUT_VARIABLE base_output
  ERROR_VARIABLE base_error
)
if(base_status EQUAL 0)
  message(FATAL_ERROR
    "rocq-run unexpectedly executed qir-v2-base source: ${base_output}")
endif()
if(NOT base_error MATCHES "qir-v2-base")
  message(FATAL_ERROR
    "rocq-run Base rejection was not actionable: ${base_error}")
endif()

execute_process(
  COMMAND "${RUNNER}" --unknown-option
  RESULT_VARIABLE usage_status
  OUTPUT_VARIABLE usage_output
  ERROR_VARIABLE usage_error
)
if(NOT usage_status EQUAL 2)
  message(FATAL_ERROR
    "rocq-run usage error returned ${usage_status}, expected 2")
endif()
if(NOT usage_error MATCHES "unknown option")
  message(FATAL_ERROR
    "rocq-run usage error was not actionable: ${usage_error}")
endif()
