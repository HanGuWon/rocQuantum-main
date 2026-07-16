cmake_minimum_required(VERSION 3.21)
option(ROCQUANTUM_BUILD_NATIVE
       "Build the native ROCm libraries and Python extension modules." ON)
option(ROCQUANTUM_ENABLE_MLIR_COMPILER
       "Build the native MLIR/LLVM/QIR compiler (LLVM/MLIR 22.1.x)." OFF)
option(ROCQUANTUM_ALLOW_UNSUPPORTED_MLIR
       "Allow a non-22.1 LLVM/MLIR for development-only compiler smoke tests." OFF)

set(_rocq_native_wheel_requested "$ENV{ROCQ_BUILD_NATIVE}")
if(SKBUILD AND ROCQUANTUM_BUILD_NATIVE AND NOT _rocq_native_wheel_requested)
    message(FATAL_ERROR
        "Native scikit-build wheels require ROCQ_BUILD_NATIVE=1 so wheel tags and "
        "CMake native mode are selected together. For a direct native CMake build, "
        "configure outside scikit-build with -DROCQUANTUM_BUILD_NATIVE=ON.")
endif()
unset(_rocq_native_wheel_requested)

if(ROCQUANTUM_BUILD_NATIVE AND
   (NOT DEFINED CMAKE_HIP_ARCHITECTURES OR CMAKE_HIP_ARCHITECTURES STREQUAL ""))
    # CMake selects HIP architectures while project(... LANGUAGES HIP) enables
    # the compiler, so a compile-only host needs a conservative value in advance.
    # Release workflows override this for gfx942/gfx950 coverage.
    set(CMAKE_HIP_ARCHITECTURES "gfx90a" CACHE STRING
        "Default HIP architecture for rocQuantum" FORCE)
endif()

if(ROCQUANTUM_BUILD_NATIVE)
    project(rocQuantum VERSION 0.1.0 LANGUAGES CXX HIP)
elseif(ROCQUANTUM_ENABLE_MLIR_COMPILER)
    project(rocQuantum VERSION 0.1.0 LANGUAGES CXX)
else()
    project(rocQuantum VERSION 0.1.0 LANGUAGES NONE)
endif()

if(NOT ROCQUANTUM_BUILD_NATIVE AND NOT ROCQUANTUM_ENABLE_MLIR_COMPILER)
    message(STATUS "Building the host-only Python package (native ROCm build disabled)")
    install(
        FILES rocq_cli.py
        DESTINATION .
    )
    return()
endif()

include(GNUInstallDirs)
include(CTest)
include(CMakePackageConfigHelpers)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(ROCQUANTUM_BUILD_NATIVE)
    set(CMAKE_HIP_STANDARD 17)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(ROCQUANTUM_ENABLE_MLIR_COMPILER)
    add_subdirectory(rocqCompiler)
endif()

if(NOT ROCQUANTUM_BUILD_NATIVE)
    message(STATUS "Building the CPU-only MLIR/LLVM/QIR compiler (ROCm runtime disabled)")
    install(FILES rocq_cli.py DESTINATION .)
    return()
endif()

option(ROCQUANTUM_BUILD_BINDINGS "Build Python bindings." ON)
option(ROCQUANTUM_BUILD_COMPONENTS_SHARED "Build component libraries as shared libraries." ON)
option(ROCQ_PRECISION_DOUBLE
       "Build StateVec and TensorNet public APIs/kernels with complex128 precision." OFF)
option(ROCQUANTUM_ENABLE_SYNC_DEBUG "Enable debug stream synchronization in enqueue APIs." OFF)
if(ROCQUANTUM_BUILD_COMPONENTS_SHARED)
    set(ROCQUANTUM_COMPONENT_LIBRARY_TYPE SHARED)
else()
    set(ROCQUANTUM_COMPONENT_LIBRARY_TYPE STATIC)
endif()
set(ROCQUANTUM_EXPORT_SET_NAME rocQuantumTargets)

find_package(hip CONFIG REQUIRED)
find_package(hiprand REQUIRED)
find_package(rocblas REQUIRED)
find_package(rocsolver REQUIRED)

if(ROCQUANTUM_BUILD_BINDINGS)
    find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(pybind11 2.10 CONFIG REQUIRED)
endif()

set(ROCQUANTUM_PUBLIC_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(ROCQUANTUM_COMPONENT_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/rocquantum/include")
set(ROCQUANTUM_BUILT_WITH_METIS OFF)

add_subdirectory(rocquantum/src/hipStateVec)
add_subdirectory(rocquantum/src/hipTensorNet)
add_subdirectory(rocquantum/src/hipDensityMat)

if(ROCQ_PRECISION_DOUBLE)
    foreach(_rocq_precision_target
            hipStateVec
            rocqsim_tensornet)
        target_compile_definitions(
            ${_rocq_precision_target}
            PUBLIC ROCQ_PRECISION_DOUBLE=1
        )
    endforeach()
endif()

option(ROCQUANTUM_BUILD_LEGACY_KERNELS
       "Include legacy kernels.hip.cpp in the core library (experimental)." OFF)

add_library(rocquantum STATIC
    rocquantum/src/simulator.cpp
    rocqCompiler/HipStateVecBackend.cpp
)

if(ROCQUANTUM_BUILD_LEGACY_KERNELS)
    target_sources(rocquantum PRIVATE rocquantum/src/kernels.hip.cpp)
    set_source_files_properties(rocquantum/src/kernels.hip.cpp PROPERTIES LANGUAGE HIP)
    message(STATUS "Legacy kernels.hip.cpp included (experimental)")
endif()

target_include_directories(rocquantum
    PUBLIC
        $<BUILD_INTERFACE:${ROCQUANTUM_PUBLIC_INCLUDE_DIR}>
        $<BUILD_INTERFACE:${ROCQUANTUM_COMPONENT_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(rocquantum
    PUBLIC
        hipStateVec
        rocqsim_tensornet
        rocq_hip_density_mat
)

add_library(rocquantum::core ALIAS rocquantum)

if(ROCQUANTUM_BUILD_BINDINGS)
    add_subdirectory(python/rocq)

    pybind11_add_module(rocq_hip rocquantum/src/python/py_hip_density_mat.cpp)
    target_include_directories(rocq_hip PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${ROCQUANTUM_PUBLIC_INCLUDE_DIR}
        ${ROCQUANTUM_COMPONENT_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/rocquantum/src/hipDensityMat
    )
    target_link_libraries(rocq_hip
        PRIVATE
            rocq_hip_density_mat
            pybind11::module
    )

    pybind11_add_module(rocquantum_bind bindings.cpp)

    target_include_directories(rocquantum_bind PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${ROCQUANTUM_PUBLIC_INCLUDE_DIR}
        ${ROCQUANTUM_COMPONENT_INCLUDE_DIR}
    )

    target_link_libraries(rocquantum_bind
        PRIVATE
            rocquantum
            pybind11::module
    )

    if(ROCQUANTUM_ENABLE_MLIR_COMPILER)
        target_link_libraries(rocquantum_bind PRIVATE rocqCompiler)
        target_compile_definitions(
            rocquantum_bind
            PRIVATE ROCQUANTUM_ENABLE_MLIR_COMPILER=1
        )
    endif()

    if(UNIX AND NOT APPLE)
        set_target_properties(
            _rocq_hip_backend
            rocq_hip
            rocquantum_bind
            PROPERTIES
                BUILD_RPATH_USE_ORIGIN TRUE
                INSTALL_RPATH "$ORIGIN/${CMAKE_INSTALL_LIBDIR}"
        )
    endif()
endif()

install(
    DIRECTORY include/rocquantum
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
    DIRECTORY rocquantum/include/rocquantum
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    PATTERN "Compiler" EXCLUDE
    PATTERN "Dialect" EXCLUDE
)

install(
    TARGETS rocquantum
    EXPORT ${ROCQUANTUM_EXPORT_SET_NAME}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if(ROCQUANTUM_BUILD_BINDINGS)
    install(
        TARGETS rocq_hip rocquantum_bind
        EXPORT ${ROCQUANTUM_EXPORT_SET_NAME}
        LIBRARY DESTINATION .
        RUNTIME DESTINATION .
    )
endif()

install(
    FILES rocq_cli.py
    DESTINATION .
)

install(
    EXPORT ${ROCQUANTUM_EXPORT_SET_NAME}
    FILE rocQuantumTargets.cmake
    NAMESPACE rocquantum::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rocQuantum
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/rocQuantumConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/rocQuantumConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rocQuantum
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/rocQuantumConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/rocQuantumConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/rocQuantumConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rocQuantum
)

