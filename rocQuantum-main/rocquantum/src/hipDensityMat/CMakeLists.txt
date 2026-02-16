if(NOT DEFINED ROCQUANTUM_COMPONENT_LIBRARY_TYPE)
    set(ROCQUANTUM_COMPONENT_LIBRARY_TYPE STATIC)
endif()
if(NOT DEFINED ROCQUANTUM_EXPORT_SET_NAME)
    set(ROCQUANTUM_EXPORT_SET_NAME rocQuantumTargets)
endif()

include(GNUInstallDirs)

add_library(rocq_hip_density_mat ${ROCQUANTUM_COMPONENT_LIBRARY_TYPE}
    hipDensityMat.cpp
    rocDensityMatCompat.cpp
)
add_library(rocquantum::densitymat ALIAS rocq_hip_density_mat)

target_include_directories(rocq_hip_density_mat
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(rocq_hip_density_mat
    PUBLIC
        HIP::hip_runtime
)

install(
    TARGETS rocq_hip_density_mat
    EXPORT ${ROCQUANTUM_EXPORT_SET_NAME}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(
    FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/hipDensityMat.hpp"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocquantum
)
