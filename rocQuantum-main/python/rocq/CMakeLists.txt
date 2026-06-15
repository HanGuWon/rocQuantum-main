if(NOT TARGET hipStateVec)
    message(FATAL_ERROR "python/rocq requires the hipStateVec target; configure from the repository root.")
endif()

if(NOT TARGET rocqsim_tensornet)
    message(FATAL_ERROR "python/rocq requires the rocqsim_tensornet target; configure from the repository root.")
endif()

if(NOT TARGET pybind11::module)
    find_package(pybind11 CONFIG REQUIRED)
endif()

if(NOT DEFINED ROCQUANTUM_PUBLIC_INCLUDE_DIR)
    set(ROCQUANTUM_PUBLIC_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")
endif()

if(NOT DEFINED ROCQUANTUM_COMPONENT_INCLUDE_DIR)
    set(ROCQUANTUM_COMPONENT_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/rocquantum/include")
endif()

pybind11_add_module(_rocq_hip_backend bindings.cpp)

target_include_directories(_rocq_hip_backend PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${ROCQUANTUM_PUBLIC_INCLUDE_DIR}
    ${ROCQUANTUM_COMPONENT_INCLUDE_DIR}
)

target_link_libraries(_rocq_hip_backend
    PRIVATE
        hipStateVec
        rocqsim_tensornet
        pybind11::module
)

install(
    TARGETS _rocq_hip_backend
    EXPORT ${ROCQUANTUM_EXPORT_SET_NAME}
    LIBRARY DESTINATION .
    RUNTIME DESTINATION .
)

message(STATUS "rocQuantum legacy Python bindings configured: _rocq_hip_backend")
