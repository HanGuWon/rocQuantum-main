cmake_minimum_required(VERSION 3.18)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    message(FATAL_ERROR "Configure from the repository root: rocQuantum-main/CMakeLists.txt")
endif()

# Legacy shim: keep this file as a module layer but do not expose a standalone
# project entrypoint here.
add_subdirectory(src/hipStateVec)
add_subdirectory(src/hipTensorNet)
add_subdirectory(src/hipDensityMat)

