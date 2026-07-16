foreach(required_variable
    TRANSLATOR INPUT BASE_INPUT INVALID_INPUT WORK_DIR LLVM_AS LLVM_OPT
    LLVM_DIS LLVM_READOBJ LLVM_NM)
  if(NOT DEFINED ${required_variable} OR "${${required_variable}}" STREQUAL "")
    message(FATAL_ERROR "${required_variable} is required")
  endif()
endforeach()

function(expect_exit expected label)
  execute_process(
    COMMAND ${ARGN}
    RESULT_VARIABLE actual
    OUTPUT_VARIABLE captured_stdout
    ERROR_VARIABLE captured_stderr
  )
  if(NOT "${actual}" STREQUAL "${expected}")
    message(FATAL_ERROR
      "${label}: expected exit ${expected}, got ${actual}\n"
      "stdout:\n${captured_stdout}\n"
      "stderr:\n${captured_stderr}")
  endif()
  set(LAST_STDOUT "${captured_stdout}" PARENT_SCOPE)
  set(LAST_STDERR "${captured_stderr}" PARENT_SCOPE)
endfunction()

function(require_contains text needle label)
  string(FIND "${text}" "${needle}" position)
  if(position EQUAL -1)
    message(FATAL_ERROR "${label}: expected text to contain '${needle}'")
  endif()
endfunction()

function(require_equal_files first second label)
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E compare_files "${first}" "${second}"
    RESULT_VARIABLE compare_result
  )
  if(NOT compare_result EQUAL 0)
    message(FATAL_ERROR "${label}: files differ: ${first} and ${second}")
  endif()
endfunction()

file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}")

expect_exit(0 "version"
  "${TRANSLATOR}" --version)
require_contains("${LAST_STDOUT}" "rocq-translate" "version output")
require_contains("${LAST_STDOUT}" "LLVM 22.1" "version output")

# Textual LLVM IR: optimize, verify with the selected LLVM, and require
# byte-for-byte repeatability.
set(ir_one "${WORK_DIR}/artifact.one.ll")
set(ir_two "${WORK_DIR}/artifact.two.ll")
expect_exit(0 "first LLVM IR emission"
  "${TRANSLATOR}" --emit=llvm-ir -O2 -o "${ir_one}" "${INPUT}")
expect_exit(0 "repeated LLVM IR emission"
  "${TRANSLATOR}" --emit=llvm-ir -O2 -o "${ir_two}" "${INPUT}")
require_equal_files("${ir_one}" "${ir_two}" "deterministic LLVM IR")
expect_exit(0 "llvm-as verification"
  "${LLVM_AS}" "${ir_one}" -o "${WORK_DIR}/artifact.from-ir.bc")
expect_exit(0 "opt verification of LLVM IR"
  "${LLVM_OPT}" -passes=verify -disable-output "${ir_one}")

# Binary LLVM bitcode: preserve NUL bytes, carry the raw bitcode magic, and
# round-trip through llvm-dis/opt.
set(bc_one "${WORK_DIR}/artifact.one.bc")
set(bc_two "${WORK_DIR}/artifact.two.bc")
expect_exit(0 "first bitcode emission"
  "${TRANSLATOR}" --emit llvm-bc -O3 -o "${bc_one}" "${INPUT}")
expect_exit(0 "repeated bitcode emission"
  "${TRANSLATOR}" --emit llvm-bc -O3 -o "${bc_two}" "${INPUT}")
require_equal_files("${bc_one}" "${bc_two}" "deterministic bitcode")
file(READ "${bc_one}" bitcode_magic LIMIT 4 HEX)
if(NOT bitcode_magic STREQUAL "4243c0de")
  message(FATAL_ERROR "bitcode payload has wrong magic '${bitcode_magic}'")
endif()
expect_exit(0 "llvm-dis round trip"
  "${LLVM_DIS}" "${bc_one}" -o "${WORK_DIR}/artifact.dis.ll")
expect_exit(0 "opt verification of bitcode"
  "${LLVM_OPT}" -passes=verify -disable-output "${bc_one}")

# Host-native PIC relocatable object: verify the format and that the entry is
# defined while QIS functions remain unresolved for the runtime linker.
set(object_one "${WORK_DIR}/artifact.one.o")
set(object_two "${WORK_DIR}/artifact.two.o")
expect_exit(0 "first object emission"
  "${TRANSLATOR}" --emit=object -O2 -o "${object_one}" "${INPUT}")
expect_exit(0 "repeated object emission"
  "${TRANSLATOR}" --emit=object -O2 -o "${object_two}" "${INPUT}")
require_equal_files("${object_one}" "${object_two}" "deterministic object")
expect_exit(0 "llvm-readobj verification"
  "${LLVM_READOBJ}" --file-headers "${object_one}")
require_contains("${LAST_STDOUT}" "Relocatable" "object file header")
expect_exit(0 "llvm-nm verification"
  "${LLVM_NM}" "${object_one}")
require_contains("${LAST_STDOUT}" "artifact_fixture" "object symbol table")
require_contains("${LAST_STDOUT}" "U __quantum__qis__h__body"
  "object unresolved QIS symbol")

# Base Profile QIR interchange artifacts are profile-preserving only at O0.
# Object files use a separate native-linker contract and may be optimized.
set(base_ir "${WORK_DIR}/base.ll")
expect_exit(0 "Base Profile O0 IR"
  "${TRANSLATOR}" --profile=qir-v2-base --emit=llvm-ir -O0
  -o "${base_ir}" "${BASE_INPUT}")
file(READ "${base_ir}" base_ir_text)
foreach(required_block "entry:" "body:" "measurements:" "output:")
  require_contains("${base_ir_text}" "${required_block}"
    "Base Profile O0 IR")
endforeach()
set(base_bc "${WORK_DIR}/base.bc")
set(base_dis "${WORK_DIR}/base.from-bc.ll")
expect_exit(0 "Base Profile O0 bitcode"
  "${TRANSLATOR}" --profile=qir-v2-base --emit=llvm-bc -O0
  -o "${base_bc}" "${BASE_INPUT}")
expect_exit(0 "Base Profile bitcode round trip"
  "${LLVM_DIS}" "${base_bc}" -o "${base_dis}")
file(READ "${base_dis}" base_dis_text)
foreach(required_block "entry:" "body:" "measurements:" "output:")
  require_contains("${base_dis_text}" "${required_block}"
    "Base Profile bitcode round trip")
endforeach()
expect_exit(2 "Base Profile O1 IR rejection"
  "${TRANSLATOR}" --profile=qir-v2-base --emit=llvm-ir -O1
  -o "${WORK_DIR}/base.invalid.ll" "${BASE_INPUT}")
require_contains("${LAST_STDERR}" "require -O0"
  "Base Profile O1 IR diagnostic")
expect_exit(2 "Base Profile O2 bitcode rejection"
  "${TRANSLATOR}" --profile=qir-v2-base --emit=llvm-bc -O2
  -o "${WORK_DIR}/base.invalid.bc" "${BASE_INPUT}")
require_contains("${LAST_STDERR}" "require -O0"
  "Base Profile O2 bitcode diagnostic")
expect_exit(0 "Base Profile optimized object"
  "${TRANSLATOR}" --profile=qir-v2-base --emit=object -O2
  -o "${WORK_DIR}/base.o" "${BASE_INPUT}")
expect_exit(0 "Base Profile object symbols"
  "${LLVM_NM}" "${WORK_DIR}/base.o")
foreach(required_symbol
    "U __quantum__rt__initialize"
    "U __quantum__qis__mz__body"
    "U __quantum__rt__result_record_output")
  require_contains("${LAST_STDOUT}" "${required_symbol}"
    "Base Profile optimized object")
endforeach()

# Content-addressed cache: miss, hit, identical bytes, then fail closed when
# the self-validating envelope is corrupted.
set(cache_directory "${WORK_DIR}/cache")
set(cache_one "${WORK_DIR}/cached.one.bc")
set(cache_two "${WORK_DIR}/cached.two.bc")
expect_exit(0 "artifact cache miss"
  "${TRANSLATOR}" --emit=llvm-bc -O2 --cache-dir "${cache_directory}"
  --verbose -o "${cache_one}" "${INPUT}")
require_contains("${LAST_STDERR}" "cache miss" "cache miss diagnostic")
expect_exit(0 "artifact cache hit"
  "${TRANSLATOR}" --emit=llvm-bc -O2 --cache-dir "${cache_directory}"
  --verbose -o "${cache_two}" "${INPUT}")
require_contains("${LAST_STDERR}" "cache hit" "cache hit diagnostic")
require_equal_files("${cache_one}" "${cache_two}" "cached artifact bytes")
file(GLOB_RECURSE cache_entries LIST_DIRECTORIES false
  "${cache_directory}/v1/*.cache")
list(LENGTH cache_entries cache_entry_count)
if(NOT cache_entry_count EQUAL 1)
  message(FATAL_ERROR
    "expected one content-addressed cache entry, found ${cache_entry_count}")
endif()
list(GET cache_entries 0 cache_entry)
file(WRITE "${cache_entry}" "corrupt")
expect_exit(1 "corrupt cache rejection"
  "${TRANSLATOR}" --emit=llvm-bc -O2 --cache-dir "${cache_directory}"
  -o "${WORK_DIR}/cached.corrupt.bc" "${INPUT}")
require_contains("${LAST_STDERR}" "corrupt entry"
  "corrupt cache diagnostic")

# Strict argument and I/O contracts. Usage errors are 2; compilation/cache/I/O
# failures are 1. A failed compile must not replace an existing output.
expect_exit(2 "malformed qubit count"
  "${TRANSLATOR}" --num-qubits=2junk "${INPUT}")
expect_exit(2 "unknown option"
  "${TRANSLATOR}" --definitely-unknown "${INPUT}")
expect_exit(2 "binary stdout rejection"
  "${TRANSLATOR}" --emit=llvm-bc "${INPUT}")
expect_exit(2 "invalid profile before cache lookup"
  "${TRANSLATOR}" --profile=not-a-profile --cache-dir "${cache_directory}"
  "${INPUT}")

set(sentinel "${WORK_DIR}/sentinel.ll")
file(WRITE "${sentinel}" "do-not-replace\n")
expect_exit(1 "invalid MLIR compilation"
  "${TRANSLATOR}" --emit=llvm-ir -o "${sentinel}" "${INVALID_INPUT}")
file(READ "${sentinel}" sentinel_after_failure)
if(NOT sentinel_after_failure STREQUAL "do-not-replace\n")
  message(FATAL_ERROR "failed compilation replaced the existing output")
endif()
expect_exit(0 "successful atomic replacement"
  "${TRANSLATOR}" --emit=llvm-ir -O0 -o "${sentinel}" "${INPUT}")
expect_exit(0 "verification after atomic replacement"
  "${LLVM_AS}" "${sentinel}" -o "${WORK_DIR}/sentinel.bc")

file(GLOB_RECURSE leaked_temporaries LIST_DIRECTORIES false
  "${WORK_DIR}/*.tmp-*")
if(leaked_temporaries)
  message(FATAL_ERROR "artifact tests leaked temporary files: ${leaked_temporaries}")
endif()
