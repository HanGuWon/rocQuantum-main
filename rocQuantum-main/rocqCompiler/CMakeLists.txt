cmake_minimum_required(VERSION 3.24)

# This directory is both the canonical compiler subdirectory used by
# rocQuantum and a small, CPU-only project for validating the MLIR dialects.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(rocqCompiler LANGUAGES CXX)
  set(ROCQ_COMPILER_STANDALONE ON)
else()
  set(ROCQ_COMPILER_STANDALONE OFF)
endif()

option(ROCQUANTUM_ALLOW_UNSUPPORTED_MLIR
  "Allow non-22.1 LLVM/MLIR for development-only compiler smoke tests" OFF)

if(NOT MLIR_FOUND)
  find_package(MLIR REQUIRED CONFIG)
endif()

# CUDA-Q 0.14/main develops against the LLVM/MLIR 22.1 API.  Do not silently
# accept another major/minor: generated ODS APIs and LLVM dialect lowering APIs
# are not source-compatible across arbitrary LLVM releases.
if(NOT DEFINED LLVM_VERSION_MAJOR OR NOT DEFINED LLVM_VERSION_MINOR)
  message(FATAL_ERROR
    "rocqCompiler requires LLVM/MLIR 22.1.x, but the package did not expose "
    "LLVM_VERSION_MAJOR and LLVM_VERSION_MINOR")
endif()
if(NOT LLVM_VERSION_MAJOR EQUAL 22 OR NOT LLVM_VERSION_MINOR EQUAL 1)
  if(ROCQUANTUM_ALLOW_UNSUPPORTED_MLIR)
    message(WARNING
      "Development-only compiler smoke build with unsupported LLVM/MLIR "
      "${LLVM_PACKAGE_VERSION}; release builds require 22.1.x")
  else()
    message(FATAL_ERROR
      "rocqCompiler requires LLVM/MLIR 22.1.x; found ${LLVM_PACKAGE_VERSION}. "
      "Set ROCQUANTUM_ALLOW_UNSUPPORTED_MLIR=ON only for development smoke tests.")
  endif()
endif()

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(SYSTEM ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

# ---------------------------------------------------------------------------
# Canonical quantum dialect, operations, and types.
# ---------------------------------------------------------------------------
set(LLVM_TARGET_DEFINITIONS QuantumOps.td)
mlir_tablegen(QuantumDialect.h.inc -gen-dialect-decls -dialect=quantum)
mlir_tablegen(QuantumDialect.cpp.inc -gen-dialect-defs -dialect=quantum)
mlir_tablegen(QuantumOps.h.inc -gen-op-decls)
mlir_tablegen(QuantumOps.cpp.inc -gen-op-defs)
mlir_tablegen(QuantumTypes.h.inc -gen-typedef-decls -typedefs-dialect=quantum)
mlir_tablegen(QuantumTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=quantum)
add_public_tablegen_target(RocqQuantumIncGen)

add_mlir_dialect_library(QuantumDialect
  QuantumDialect.cpp

  DISABLE_INSTALL
  PARTIAL_SOURCES_INTENDED

  DEPENDS
  RocqQuantumIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
)
if(TARGET MLIRBytecodeOpInterface)
  target_link_libraries(QuantumDialect PUBLIC MLIRBytecodeOpInterface)
endif()
target_include_directories(QuantumDialect PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)
target_compile_features(QuantumDialect PUBLIC cxx_std_20)

# ---------------------------------------------------------------------------
# Declarative pass API.  The implementation target is owned by the parent
# project; this target only generates the stable base/registration contract.
# ---------------------------------------------------------------------------
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/passes")
set(LLVM_TARGET_DEFINITIONS passes/Passes.td)
mlir_tablegen(passes/Passes.h.inc -gen-pass-decls -name RocqCompiler)
add_public_tablegen_target(RocqCompilerPassIncGen)

# ---------------------------------------------------------------------------
# Direct Quantum -> LLVM/QIR lowering, offline compiler API, and tools.
# ---------------------------------------------------------------------------
add_mlir_conversion_library(RocqQuantumToQIR
  passes/QuantumToQIRPass.cpp
  passes/Pipelines.cpp

  DISABLE_INSTALL
  PARTIAL_SOURCES_INTENDED

  DEPENDS
  RocqQuantumIncGen
  RocqCompilerPassIncGen

  LINK_LIBS PUBLIC
  QuantumDialect
  MLIRFuncDialect
  MLIRFuncToLLVM
  MLIRIR
  MLIRLLVMDialect
  MLIRPass
  MLIRTransforms
)
target_include_directories(RocqQuantumToQIR PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)

find_package(Threads REQUIRED)

add_library(rocqCompiler STATIC
  CompilerArtifacts.cpp
  MLIRCompiler.cpp
  QIRExecutionEngine.cpp
  QIRProfile.cpp
  ReferenceStateVecBackend.cpp
)
add_library(rocquantum::compiler ALIAS rocqCompiler)
add_dependencies(rocqCompiler RocqQuantumIncGen RocqCompilerPassIncGen)
target_include_directories(rocqCompiler PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)
target_link_libraries(rocqCompiler
  PUBLIC
  RocqQuantumToQIR
  PRIVATE
  MLIRIR
  MLIRExecutionEngine
  MLIRLLVMToLLVMIRTranslation
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRTargetLLVMIRExport
  Threads::Threads
)
if(TARGET LLVM)
  target_link_libraries(rocqCompiler PRIVATE LLVM)
else()
  llvm_map_components_to_libnames(ROCQ_COMPILER_LLVM_LIBS
    bitwriter
    core
    executionengine
    jitlink
    nativecodegen
    object
    orcjit
    orcshared
    orctargetprocess
    passes
    runtimedyld
    support
    target
    targetparser
  )
  target_link_libraries(rocqCompiler PRIVATE ${ROCQ_COMPILER_LLVM_LIBS})
endif()
if(TARGET MLIRBuiltinToLLVMIRTranslation)
  target_link_libraries(rocqCompiler PRIVATE MLIRBuiltinToLLVMIRTranslation)
endif()
target_compile_features(rocqCompiler PUBLIC cxx_std_20)

# The artifact cache key must change whenever source that can alter emitted
# bytes changes.  CMAKE_CONFIGURE_DEPENDS makes edits trigger reconfiguration;
# the resulting lowercase SHA-256 is compiled into the tooling library.
set(ROCQ_COMPILER_FINGERPRINT_INPUTS
  ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt
  ${CMAKE_CURRENT_SOURCE_DIR}/QuantumDialect.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/QuantumOps.td
  ${CMAKE_CURRENT_SOURCE_DIR}/CompilerArtifacts.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/CompilerArtifacts.h
  ${CMAKE_CURRENT_SOURCE_DIR}/MLIRCompiler.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/MLIRCompiler.h
  ${CMAKE_CURRENT_SOURCE_DIR}/QIRProfile.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/QIRProfile.h
  ${CMAKE_CURRENT_SOURCE_DIR}/passes/Passes.td
  ${CMAKE_CURRENT_SOURCE_DIR}/passes/Pipelines.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/passes/Pipelines.h
  ${CMAKE_CURRENT_SOURCE_DIR}/passes/QuantumToQIRPass.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/passes/QuantumToQIRPass.h
  ${CMAKE_CURRENT_SOURCE_DIR}/tools/ArtifactCache.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/tools/ArtifactCache.h
)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  ${ROCQ_COMPILER_FINGERPRINT_INPUTS})
set(_rocq_compiler_fingerprint_material "")
foreach(_rocq_compiler_input IN LISTS ROCQ_COMPILER_FINGERPRINT_INPUTS)
  file(SHA256 "${_rocq_compiler_input}" _rocq_compiler_input_sha256)
  file(RELATIVE_PATH _rocq_compiler_input_relative
    "${CMAKE_CURRENT_SOURCE_DIR}" "${_rocq_compiler_input}")
  string(APPEND _rocq_compiler_fingerprint_material
    "${_rocq_compiler_input_relative}=${_rocq_compiler_input_sha256}\n")
endforeach()
string(SHA256 ROCQ_COMPILER_FINGERPRINT
  "${_rocq_compiler_fingerprint_material}")
unset(_rocq_compiler_fingerprint_material)
unset(_rocq_compiler_input)
unset(_rocq_compiler_input_relative)
unset(_rocq_compiler_input_sha256)

add_library(rocqCompilerTooling STATIC tools/ArtifactCache.cpp)
target_include_directories(rocqCompilerTooling PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)
target_compile_definitions(rocqCompilerTooling PRIVATE
  ROCQ_COMPILER_FINGERPRINT="${ROCQ_COMPILER_FINGERPRINT}"
)
target_compile_features(rocqCompilerTooling PUBLIC cxx_std_20)
target_link_libraries(rocqCompilerTooling PUBLIC rocqCompiler)

add_executable(rocq-opt tools/rocq-opt.cpp)
target_include_directories(rocq-opt PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)
target_link_libraries(rocq-opt PRIVATE
  RocqQuantumToQIR
  QuantumDialect
  MLIROptLib
)

add_executable(rocq-translate tools/rocq-translate.cpp)
target_include_directories(rocq-translate PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)
if(PROJECT_VERSION)
  set(ROCQ_TRANSLATE_VERSION "${PROJECT_VERSION}")
else()
  set(ROCQ_TRANSLATE_VERSION "0.1.0")
endif()
target_compile_definitions(rocq-translate PRIVATE
  ROCQ_VERSION="${ROCQ_TRANSLATE_VERSION}"
)
target_link_libraries(rocq-translate PRIVATE rocqCompilerTooling)

add_executable(rocq-run tools/rocq-run.cpp)
target_include_directories(rocq-run PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)
target_compile_definitions(rocq-run PRIVATE
  ROCQ_VERSION="${ROCQ_TRANSLATE_VERSION}"
)
target_link_libraries(rocq-run PRIVATE rocqCompiler)

include(GNUInstallDirs)
install(TARGETS rocq-opt rocq-translate rocq-run
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

include(CTest)
if(BUILD_TESTING)
  add_executable(rocq-compiler-smoke tests/CompilerSmoke.cpp)
  target_include_directories(rocq-compiler-smoke PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
  )
  target_link_libraries(rocq-compiler-smoke PRIVATE rocqCompiler)
  add_test(NAME rocq.compiler.smoke COMMAND rocq-compiler-smoke)

  add_executable(rocq-reference-statevec-smoke
    tests/ReferenceStateVecBackendSmoke.cpp)
  target_include_directories(rocq-reference-statevec-smoke PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
  )
  target_link_libraries(rocq-reference-statevec-smoke PRIVATE rocqCompiler)
  add_test(NAME rocq.compiler.reference-statevec-smoke
    COMMAND rocq-reference-statevec-smoke)

  add_test(
    NAME rocq.compiler.run-cli
    COMMAND ${CMAKE_COMMAND}
      -DRUNNER=$<TARGET_FILE:rocq-run>
      -DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/tests/bell_state.mlir
      -DBASE_INPUT=${CMAKE_CURRENT_SOURCE_DIR}/tests/base_profile.mlir
      -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/CheckRocqRun.cmake
  )

  add_executable(rocq-compiler-artifacts-smoke
    tests/CompilerArtifactsSmoke.cpp)
  target_include_directories(rocq-compiler-artifacts-smoke PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
  )
  target_link_libraries(rocq-compiler-artifacts-smoke PRIVATE
    rocqCompiler
    Threads::Threads
  )
  add_test(NAME rocq.compiler.artifacts-smoke
    COMMAND rocq-compiler-artifacts-smoke)

  add_executable(rocq-artifact-cache-smoke tests/ArtifactCacheSmoke.cpp)
  target_include_directories(rocq-artifact-cache-smoke PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
  )
  target_link_libraries(rocq-artifact-cache-smoke PRIVATE
    rocqCompilerTooling
    Threads::Threads
  )
  add_test(NAME rocq.compiler.artifact-cache-smoke
    COMMAND rocq-artifact-cache-smoke)
  add_test(
    NAME rocq.opt.quantum-to-qir
    COMMAND $<TARGET_FILE:rocq-opt>
      --rocq-quantum-to-qir
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/bell.mlir
      -o ${CMAKE_CURRENT_BINARY_DIR}/bell.lowered.mlir
  )
  add_test(
    NAME rocq.opt.static-pipeline
    COMMAND $<TARGET_FILE:rocq-opt>
      "--pass-pipeline=builtin.module(rocq-qir-static-pipeline)"
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/bell.mlir
      -o ${CMAKE_CURRENT_BINARY_DIR}/bell.static-pipeline.mlir
  )
  add_test(
    NAME rocq.opt.base-pipeline
    COMMAND $<TARGET_FILE:rocq-opt>
      "--pass-pipeline=builtin.module(rocq-qir-base-pipeline)"
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/base_profile.mlir
      -o ${CMAKE_CURRENT_BINARY_DIR}/base.pipeline.mlir
  )
  add_test(
    NAME rocq.opt.static-pipeline-rejects-base
    COMMAND ${CMAKE_COMMAND}
      -DOPT=$<TARGET_FILE:rocq-opt>
      -DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/tests/base_profile.mlir
      -DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/base.wrong-pipeline.mlir
      -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/CheckPipelineRejection.cmake
  )

  find_program(ROCQ_LLVM_AS
    NAMES llvm-as llvm-as-${LLVM_VERSION_MAJOR}
    HINTS "${LLVM_TOOLS_BINARY_DIR}"
  )
  find_program(ROCQ_LLVM_OPT
    NAMES opt opt-${LLVM_VERSION_MAJOR}
    HINTS "${LLVM_TOOLS_BINARY_DIR}"
  )
  find_program(ROCQ_LLVM_DIS
    NAMES llvm-dis llvm-dis-${LLVM_VERSION_MAJOR}
    HINTS "${LLVM_TOOLS_BINARY_DIR}"
  )
  find_program(ROCQ_LLVM_READOBJ
    NAMES llvm-readobj llvm-readobj-${LLVM_VERSION_MAJOR}
    HINTS "${LLVM_TOOLS_BINARY_DIR}"
  )
  find_program(ROCQ_LLVM_NM
    NAMES llvm-nm llvm-nm-${LLVM_VERSION_MAJOR}
    HINTS "${LLVM_TOOLS_BINARY_DIR}"
  )
  if(NOT ROCQ_LLVM_AS OR NOT ROCQ_LLVM_OPT OR NOT ROCQ_LLVM_DIS OR
     NOT ROCQ_LLVM_READOBJ OR NOT ROCQ_LLVM_NM)
    message(FATAL_ERROR
      "Compiler artifact tests require llvm-as, opt, llvm-dis, llvm-readobj, "
      "and llvm-nm from the selected LLVM toolchain")
  endif()
  add_test(
    NAME rocq.qir.llvm-verify
    COMMAND ${CMAKE_COMMAND}
      -DTRANSLATOR=$<TARGET_FILE:rocq-translate>
      -DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/tests/bell.mlir
      -DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/bell.qir.ll
      -DLLVM_AS=${ROCQ_LLVM_AS}
      -DLLVM_OPT=${ROCQ_LLVM_OPT}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/CheckQIR.cmake
  )
  add_test(
    NAME rocq.qir.base-profile-verify
    COMMAND ${CMAKE_COMMAND}
      -DTRANSLATOR=$<TARGET_FILE:rocq-translate>
      -DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/tests/base_profile.mlir
      -DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/base_profile.qir.ll
      -DLLVM_AS=${ROCQ_LLVM_AS}
      -DLLVM_OPT=${ROCQ_LLVM_OPT}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/CheckBaseQIR.cmake
  )
  add_test(
    NAME rocq.compiler.artifact-cli
    COMMAND ${CMAKE_COMMAND}
      -DTRANSLATOR=$<TARGET_FILE:rocq-translate>
      -DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/tests/artifact.mlir
      -DBASE_INPUT=${CMAKE_CURRENT_SOURCE_DIR}/tests/base_profile.mlir
      -DINVALID_INPUT=${CMAKE_CURRENT_SOURCE_DIR}/tests/invalid-artifact.mlir
      -DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/artifact-cli-test
      -DLLVM_AS=${ROCQ_LLVM_AS}
      -DLLVM_OPT=${ROCQ_LLVM_OPT}
      -DLLVM_DIS=${ROCQ_LLVM_DIS}
      -DLLVM_READOBJ=${ROCQ_LLVM_READOBJ}
      -DLLVM_NM=${ROCQ_LLVM_NM}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/CheckCompilerArtifacts.cmake
  )
endif()

# The simulator dialect is retained as an explicit experimental compatibility
# surface.  The supported release lowering goes directly Quantum -> LLVM/QIR.
option(ROCQ_COMPILER_BUILD_EXPERIMENTAL_SIMULATOR_DIALECT
  "Build the non-release simulator intermediate dialect" OFF)
if(ROCQ_COMPILER_BUILD_EXPERIMENTAL_SIMULATOR_DIALECT)
  set(LLVM_TARGET_DEFINITIONS SimulatorOps.td)
  mlir_tablegen(SimulatorDialect.h.inc -gen-dialect-decls -dialect=sim)
  mlir_tablegen(SimulatorDialect.cpp.inc -gen-dialect-defs -dialect=sim)
  mlir_tablegen(SimulatorOps.h.inc -gen-op-decls)
  mlir_tablegen(SimulatorOps.cpp.inc -gen-op-defs)
  add_public_tablegen_target(RocqSimulatorIncGen)

  add_mlir_dialect_library(SimulatorDialect
    SimulatorDialect.cpp

    DISABLE_INSTALL
    PARTIAL_SOURCES_INTENDED

    DEPENDS
    RocqQuantumIncGen
    RocqSimulatorIncGen

    LINK_LIBS PUBLIC
    QuantumDialect
    MLIRIR
    MLIRSideEffectInterfaces
  )
  if(TARGET MLIRBytecodeOpInterface)
    target_link_libraries(SimulatorDialect PUBLIC MLIRBytecodeOpInterface)
  endif()
  target_include_directories(SimulatorDialect PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )
  target_compile_features(SimulatorDialect PUBLIC cxx_std_20)
endif()
