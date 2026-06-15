cmake_minimum_required(VERSION 3.21)
project(rocQuantumInstallConsumerSmoke LANGUAGES CXX HIP)

find_package(rocQuantum CONFIG REQUIRED)

foreach(required_target
        rocquantum::rocquantum
        rocquantum::hipStateVec
        rocquantum::rocqsim_tensornet
        rocquantum::rocq_hip_density_mat)
    if(NOT TARGET ${required_target})
        message(FATAL_ERROR "Missing installed rocQuantum target: ${required_target}")
    endif()
endforeach()

add_executable(rocquantum_install_consumer_smoke main.cpp)
target_compile_features(rocquantum_install_consumer_smoke PRIVATE cxx_std_17)
target_link_libraries(rocquantum_install_consumer_smoke PRIVATE rocquantum::rocquantum)
