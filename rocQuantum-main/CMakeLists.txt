cmake_minimum_required(VERSION 3.18)
project(rocQuantum VERSION 0.1.0 LANGUAGES CXX HIP)

include(GNUInstallDirs)
include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

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
    find_package(Python3 COMPONENTS Development REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)
endif()

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
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/rocquantum/include
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
    pybind11_add_module(rocquantum_bind bindings.cpp)

    target_include_directories(rocquantum_bind PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/rocquantum/include
    )

    target_link_libraries(rocquantum_bind
        PRIVATE
            rocquantum
            pybind11::module
            Python3::Python
    )
endif()

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

install(
    EXPORT ${ROCQUANTUM_EXPORT_SET_NAME}
    FILE rocQuantumTargets.cmake
    NAMESPACE rocquantum::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rocQuantum
)

