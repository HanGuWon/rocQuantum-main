foreach(required_variable OPT INPUT OUTPUT)
  if(NOT DEFINED ${required_variable} OR "${${required_variable}}" STREQUAL "")
    message(FATAL_ERROR "${required_variable} is required")
  endif()
endforeach()

execute_process(
  COMMAND "${OPT}"
    "--pass-pipeline=builtin.module(rocq-qir-static-pipeline)"
    "${INPUT}" -o "${OUTPUT}"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE captured_stdout
  ERROR_VARIABLE captured_stderr
)
if("${result}" STREQUAL "0")
  message(FATAL_ERROR
    "static-custom pipeline unexpectedly accepted Base Profile measurement IR")
endif()
string(FIND "${captured_stderr}"
  "quantum.mz requires profile='qir-v2-base'" diagnostic_position)
if(diagnostic_position EQUAL -1)
  message(FATAL_ERROR
    "static-custom rejection did not report the profile diagnostic\n"
    "stdout:\n${captured_stdout}\n"
    "stderr:\n${captured_stderr}")
endif()
