add_executable(test_hipDensityMat test_hipDensityMat.cpp)
target_link_libraries(test_hipDensityMat PRIVATE
    rocq_hip_density_mat
    hip::host
)
add_test(NAME HipDensityMatRegression COMMAND test_hipDensityMat)
set_tests_properties(HipDensityMatRegression PROPERTIES
    LABELS "native;single-gpu;densitymat"
    SKIP_RETURN_CODE 77
)
