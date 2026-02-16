set(HIPSTATEVEC_SRC
    hipStateVec.cpp
    GateFusion.cpp
    single_qubit_kernels.hip
    two_qubit_kernels.hip
    measurement_kernels.hip
    multi_qubit_kernels.hip
    swap_kernels.hip
)

include(GNUInstallDirs)

if(NOT DEFINED ROCQUANTUM_COMPONENT_LIBRARY_TYPE)
    set(ROCQUANTUM_COMPONENT_LIBRARY_TYPE SHARED)
endif()
if(NOT DEFINED ROCQUANTUM_EXPORT_SET_NAME)
    set(ROCQUANTUM_EXPORT_SET_NAME rocQuantumTargets)
endif()

add_library(hipStateVec ${ROCQUANTUM_COMPONENT_LIBRARY_TYPE} ${HIPSTATEVEC_SRC})
add_library(rocquantum::statevec ALIAS hipStateVec)

set_source_files_properties(${HIPSTATEVEC_SRC} PROPERTIES HIP_SOURCE_TYPE "kernels")

target_include_directories(hipStateVec
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
)

target_link_libraries(hipStateVec
    PUBLIC
        HIP::hip_runtime
        roc::rocblas
        hiprand::hiprand
)

if(ROCQUANTUM_ENABLE_SYNC_DEBUG)
    target_compile_definitions(hipStateVec PRIVATE ROCQUANTUM_ENABLE_SYNC_DEBUG=1)
endif()

find_package(rccl QUIET)
if(rccl_FOUND AND TARGET rccl::rccl)
    target_link_libraries(hipStateVec PUBLIC rccl::rccl)
elseif(rccl_FOUND AND DEFINED rccl_LIBRARIES)
    target_link_libraries(hipStateVec PUBLIC ${rccl_LIBRARIES})
endif()

install(
    TARGETS hipStateVec
    EXPORT ${ROCQUANTUM_EXPORT_SET_NAME}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if(BUILD_TESTING)
    add_executable(test_hipStateVec_multi_gpu test_hipStateVec_multi_gpu.cpp)
    target_link_libraries(test_hipStateVec_multi_gpu PRIVATE hipStateVec HIP::hip_runtime)
    add_test(NAME MultiGPUTests COMMAND test_hipStateVec_multi_gpu)
endif()
