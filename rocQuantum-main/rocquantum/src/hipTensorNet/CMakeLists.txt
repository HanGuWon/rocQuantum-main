if(NOT DEFINED ROCQUANTUM_COMPONENT_LIBRARY_TYPE)
    set(ROCQUANTUM_COMPONENT_LIBRARY_TYPE STATIC)
endif()
if(NOT DEFINED ROCQUANTUM_EXPORT_SET_NAME)
    set(ROCQUANTUM_EXPORT_SET_NAME rocQuantumTargets)
endif()

include(GNUInstallDirs)

add_library(rocqsim_tensornet ${ROCQUANTUM_COMPONENT_LIBRARY_TYPE}
    rocTensorUtil.cpp
    hipTensorNet.cpp
    rocTensorUtil_kernels.hip
    PermutationKernels.hip
)
add_library(rocquantum::tensornet ALIAS rocqsim_tensornet)

set_source_files_properties(rocTensorUtil_kernels.hip PROPERTIES LANGUAGE HIP)
set_source_files_properties(PermutationKernels.hip PROPERTIES LANGUAGE HIP)

target_include_directories(rocqsim_tensornet
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
)

target_link_libraries(rocqsim_tensornet
    PUBLIC
        HIP::hip_runtime
        roc::rocblas
        roc::rocsolver
)

install(
    TARGETS rocqsim_tensornet
    EXPORT ${ROCQUANTUM_EXPORT_SET_NAME}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if(BUILD_TESTING)
    add_executable(test_rocTensorUtil test_hipTensorNet_rocTensorUtil.cpp)
    target_link_libraries(test_rocTensorUtil PRIVATE rocqsim_tensornet)

    add_executable(test_slicing test_hipTensorNet_slicing.cpp)
    target_link_libraries(test_slicing PRIVATE rocqsim_tensornet)

    add_executable(test_contraction_regression test_hipTensorNet_contraction_regression.cpp)
    target_link_libraries(test_contraction_regression PRIVATE rocqsim_tensornet)

    add_executable(test_permutation ../../tests/hipTensorNet/test_PermutationKernels.cpp)
    target_link_libraries(test_permutation PRIVATE rocqsim_tensornet)

    add_test(NAME RocTensorUtilTest COMMAND test_rocTensorUtil)
    add_test(NAME SlicingLogicTest COMMAND test_slicing)
    add_test(NAME HipTensorNetContractionRegression COMMAND test_contraction_regression)
    add_test(NAME PermutationKernelTest COMMAND test_permutation)
endif()
