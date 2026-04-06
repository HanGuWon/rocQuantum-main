cmake_minimum_required(VERSION 3.18)
project(rocQuantum VERSION 0.1.0 LANGUAGES CXX HIP)

include(GNUInstallDirs)
include(CTest)
include(CMakePackageConfigHelpers)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED CMAKE_HIP_ARCHITECTURES OR CMAKE_HIP_ARCHITECTURES STREQUAL "")
    set(CMAKE_HIP_ARCHITECTURES "gfx950;gfx942;gfx90a" CACHE STRING
        "Default HIP architectures for rocQuantum" FORCE)
endif()

option(ROCQUANTUM_BUILD_BINDINGS "Build Python bindings." ON)
option(ROCQUANTUM_BUILD_COMPONENTS_SHARED "Build component libraries as shared libraries." ON)
option(ROCQUANTUM_ENABLE_SYNC_DEBUG "Enable debug stream synchronization in enqueue APIs." OFF)

if(ROCQUANTUM_BUILD_COMPONENTS_SHARED)
    set(ROCQUANTUM_COMPONENT_LIBRARY_TYPE SHARED)
else()
    set(ROCQUANTUM_COMPONENT_LIBRARY_TYPE STATIC)
endif()
set(ROCQUANTUM_EXPORT_SET_NAME rocQuantumTargets)

find_package(HIP REQUIRED)
find_package(hiprand REQUIRED)
find_package(rocblas REQUIRED)
find_package(rocsolver REQUIRED)

if(ROCQUANTUM_BUILD_BINDINGS)
    find_package(Python3 COMPONENTS Development.Module REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)
endif()

set(ROCQUANTUM_PUBLIC_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(ROCQUANTUM_COMPONENT_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/rocquantum/include")

add_subdirectory(rocquantum/src/hipStateVec)
add_subdirectory(rocquantum/src/hipTensorNet)
add_subdirectory(rocquantum/src/hipDensityMat)

option(ROCQUANTUM_BUILD_LEGACY_KERNELS
       "Include legacy kernels.hip.cpp in the core library (experimental)." OFF)

add_library(rocquantum STATIC
    rocquantum/src/simulator.cpp
    rocqCompiler/HipStateVecBackend.cpp
)

if(ROCQUANTUM_BUILD_LEGACY_KERNELS)
    target_sources(rocquantum PRIVATE rocquantum/src/kernels.hip.cpp)
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
    pybind11_add_module(_rocq_hip_backend python/rocq/bindings.cpp)
    target_include_directories(_rocq_hip_backend PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${ROCQUANTUM_PUBLIC_INCLUDE_DIR}
        ${ROCQUANTUM_COMPONENT_INCLUDE_DIR}
    )
    target_link_libraries(_rocq_hip_backend
        PRIVATE
            hipStateVec
            rocqsim_tensornet
            pybind11::module
    )

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
endif()

install(
    DIRECTORY include/rocquantum
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
    DIRECTORY rocquantum/include/rocquantum
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
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
        TARGETS _rocq_hip_backend rocq_hip rocquantum_bind
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

