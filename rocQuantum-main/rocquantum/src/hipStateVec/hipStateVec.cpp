#include "rocquantum/hipStateVec.h"

#include <hip/hip_runtime.h>
#include <hiprand/hiprand.h>
#include <hiprand/hiprand_kernel.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef ROCQ_HAVE_RCCL
#if __has_include(<rccl/rccl.h>)
#include <rccl/rccl.h>
#else
#include <nccl.h>
#endif
#endif

// Kernels implemented in the corresponding .hip translation units.
__global__ void apply_single_qubit_matrix_kernel(rocComplex* state,
                                                 unsigned numQubits,
                                                 unsigned targetQubit,
                                                 rocComplex m00,
                                                 rocComplex m01,
                                                 rocComplex m10,
                                                 rocComplex m11,
                                                 size_t batchSize);

__global__ void apply_single_qubit_matrix_batch_kernel(rocComplex* state,
                                                       unsigned numQubits,
                                                       unsigned targetQubit,
                                                       const rocComplex* matrices,
                                                       size_t batchSize);

__global__ void apply_controlled_single_qubit_matrix_kernel(rocComplex* state,
                                                            unsigned numQubits,
                                                            unsigned controlQubit,
                                                            unsigned targetQubit,
                                                            rocComplex m00,
                                                            rocComplex m01,
                                                            rocComplex m10,
                                                            rocComplex m11,
                                                            size_t batchSize);

__global__ void apply_controlled_single_qubit_matrix_batch_kernel(rocComplex* state,
                                                                  unsigned numQubits,
                                                                  unsigned controlQubit,
                                                                  unsigned targetQubit,
                                                                  const rocComplex* matrices,
                                                                  size_t batchSize);

__global__ void apply_CNOT_kernel(rocComplex* state,
                                  unsigned numQubits,
                                  unsigned controlQubit,
                                  unsigned targetQubit,
                                  size_t batchSize);

__global__ void apply_CZ_kernel(rocComplex* state,
                                unsigned numQubits,
                                unsigned controlQubit,
                                unsigned targetQubit,
                                size_t batchSize);

__global__ void apply_SWAP_kernel(rocComplex* state,
                                  unsigned numQubits,
                                  unsigned qubitA,
                                  unsigned qubitB,
                                  size_t batchSize);

__global__ void apply_multi_controlled_x_kernel(rocComplex* state,
                                                unsigned numQubits,
                                                unsigned long long controlMask,
                                                unsigned targetQubit,
                                                size_t batchSize);

__global__ void apply_CSWAP_kernel(rocComplex* state,
                                   unsigned numQubits,
                                   unsigned controlQubit,
                                   unsigned targetQubit1,
                                   unsigned targetQubit2,
                                   size_t batchSize);

__global__ void apply_two_qubit_generic_matrix_kernel(rocComplex* state,
                                                       unsigned numQubits,
                                                       const unsigned* targetQubitIndices_gpu,
                                                       const rocComplex* matrixDevice);

__global__ void apply_three_qubit_generic_matrix_kernel(rocComplex* state,
                                                        unsigned numQubits,
                                                        const unsigned* targetQubitIndices_gpu,
                                                        const rocComplex* matrixDevice);

__global__ void apply_four_qubit_generic_matrix_kernel(rocComplex* state,
                                                       unsigned numQubits,
                                                       const unsigned* targetQubitIndices_gpu,
                                                       const rocComplex* matrixDevice);

__global__ void reduce_expectation_z_kernel(const rocComplex* state,
                                            size_t numElements,
                                            unsigned targetQubit,
                                            double* blockSums);

__global__ void reduce_expectation_x_kernel(const rocComplex* state,
                                            size_t numElements,
                                            unsigned targetQubit,
                                            double* blockSums);

__global__ void reduce_expectation_y_kernel(const rocComplex* state,
                                            size_t numElements,
                                            unsigned targetQubit,
                                            double* blockSums);

__global__ void reduce_expectation_z_product_kernel(const rocComplex* state,
                                                    size_t numElements,
                                                    const unsigned* targetQubits,
                                                    unsigned numTargetQubits,
                                                    double* blockSums);

__global__ void reduce_expectation_pauli_string_kernel(const rocComplex* state,
                                                       size_t numElements,
                                                       const char* pauliString,
                                                       const unsigned* targetQubits,
                                                       unsigned numTargetQubits,
                                                       double* blockSums);

__global__ void accumulate_sample_probabilities_kernel(const rocComplex* state,
                                                       size_t numElements,
                                                       const unsigned* measuredQubits,
                                                       unsigned numMeasuredQubits,
                                                       double* outcomeProbs);

__global__ void reduce_double_sum_kernel(const double* input,
                                         size_t numElements,
                                         double* output);

__global__ void build_sampling_cdf_kernel(double* outcomeProbs,
                                          size_t numOutcomes);

__global__ void sample_from_cdf_kernel(const double* cdf,
                                       size_t numOutcomes,
                                       uint64_t* results,
                                       unsigned numShots,
                                       unsigned long long seed);

__global__ void reduce_measure_prob0_kernel(const rocComplex* state,
                                            size_t numElements,
                                            unsigned targetQubit,
                                            double* blockSums);

__global__ void collapse_and_renorm_measure_kernel(rocComplex* state,
                                                   size_t numElements,
                                                   unsigned targetQubit,
                                                   int measuredOutcome,
                                                   double invNorm);

__global__ void renormalize_state_kernel(rocComplex* state,
                                         unsigned numQubits,
                                         real_t d_sum_sq_mag_inv_sqrt);

__global__ void local_bit_swap_permutation_kernel(rocComplex* d_local_slice,
                                                  rocComplex* d_temp_buffer_for_slice,
                                                  size_t local_slice_num_elements,
                                                  unsigned local_qubit_idx1,
                                                  unsigned local_qubit_idx2);

struct rocsvInternalHandle {
    hipStream_t streams[1];
    bool ownsPrimaryStream = false;
    size_t batchSize = 1;
    unsigned numQubits = 0;
    rocComplex* d_state = nullptr;
    bool ownsState = false;
    void* pinnedBuffer = nullptr;
    size_t pinnedBufferBytes = 0;
    rocqDeviceMemHandler_t memHandler{nullptr, nullptr, nullptr};
    std::mt19937_64 rng{0x5deece66dULL};

    bool distributedMode = false;
    int distributedGpuCount = 0;
    unsigned globalNumQubits = 0;
    unsigned numLocalQubitsPerGpu = 0;
    unsigned numGlobalSliceQubits = 0;
    size_t localSliceElements = 0;

    std::vector<int> distributedDeviceIds;
    std::vector<hipStream_t> distributedStreams;
    std::vector<rocComplex*> distributedSlices;
    std::vector<rocComplex*> distributedSwapBuffers;
    std::vector<unsigned*> distributedTargetScratch;
    std::vector<rocComplex*> distributedMatrixScratch;
#ifdef ROCQ_HAVE_RCCL
    std::vector<ncclComm_t> distributedComms;
    bool distributedRcclReady = false;
#endif
};

namespace {

constexpr unsigned kMaxDistributedMatrixTargetQubits = 4;
constexpr size_t kMaxDistributedMatrixDim = 1ULL << kMaxDistributedMatrixTargetQubits;
constexpr size_t kMaxDistributedMatrixElements =
    kMaxDistributedMatrixDim * kMaxDistributedMatrixDim;

__host__ __device__ inline rocComplex make_complex(double real, double imag) {
#ifdef ROCQ_PRECISION_DOUBLE
    return rocComplex{real, imag};
#else
    return rocComplex{static_cast<float>(real), static_cast<float>(imag)};
#endif
}

inline rocComplex* resolve_state_pointer(rocsvInternalHandle* handle, rocComplex* external) {
    if (external) {
        return external;
    }
    return handle ? handle->d_state : nullptr;
}

inline rocqStatus_t check_last_hip_error() {
    const hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "hipStateVec kernel launch failed: " << hipGetErrorString(err) << '\n';
        return ROCQ_STATUS_HIP_ERROR;
    }
#ifdef ROCQUANTUM_ENABLE_SYNC_DEBUG
    if (hipDeviceSynchronize() != hipSuccess) {
        return ROCQ_STATUS_HIP_ERROR;
    }
#endif
    return ROCQ_STATUS_SUCCESS;
}

inline bool validate_qubit_index(unsigned qubit, unsigned numQubits) {
    return qubit < numQubits;
}

inline bool compute_power_of_two(unsigned exponent, size_t* out) {
    if (!out) {
        return false;
    }
    if (exponent >= static_cast<unsigned>(sizeof(size_t) * 8)) {
        return false;
    }
    *out = size_t{1} << exponent;
    return true;
}

inline int compute_reduction_blocks(size_t elements, int threadsPerBlock);

inline std::complex<double> to_std_complex(const rocComplex& value) {
    return {static_cast<double>(value.x), static_cast<double>(value.y)};
}

inline rocComplex from_std_complex(const std::complex<double>& value) {
    return make_complex(value.real(), value.imag());
}

inline bool env_flag_enabled(const char* name) {
    const char* value = std::getenv(name);
    if (!value) {
        return false;
    }
    std::string normalized(value);
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return normalized == "1" || normalized == "true" || normalized == "yes" || normalized == "on";
}

inline bool allow_host_matrix_fallback() {
    return env_flag_enabled("ROCQ_ALLOW_HOST_MATRIX_FALLBACK");
}

inline bool env_mode_equals(const char* name, const char* expected) {
    const char* value = std::getenv(name);
    if (!value) {
        return false;
    }
    std::string normalized(value);
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return normalized == expected;
}

inline bool distributed_host_fallback_enabled() {
    return env_mode_equals("ROCQ_DISTRIBUTED_FALLBACK_MODE", "host") ||
           env_mode_equals("ROCQ_DISTRIBUTED_FALLBACK_MODE", "host_fallback") ||
           env_flag_enabled("ROCQ_ENABLE_DISTRIBUTED_HOST_FALLBACK");
}

inline bool distributed_rccl_required() {
    return env_mode_equals("ROCQ_DISTRIBUTED_COMM", "rccl") ||
           env_flag_enabled("ROCQ_REQUIRE_RCCL");
}

inline bool distributed_rccl_disabled() {
    return env_mode_equals("ROCQ_DISTRIBUTED_COMM", "host") ||
           env_mode_equals("ROCQ_DISTRIBUTED_COMM", "none") ||
           env_flag_enabled("ROCQ_DISABLE_RCCL");
}

inline rocqStatus_t default_device_malloc(void** ptr, size_t bytes) {
    if (!ptr) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (bytes == 0) {
        *ptr = nullptr;
        return ROCQ_STATUS_SUCCESS;
    }
    return (hipMalloc(ptr, bytes) == hipSuccess) ? ROCQ_STATUS_SUCCESS : ROCQ_STATUS_ALLOCATION_FAILED;
}

inline rocqStatus_t default_device_free(void* ptr) {
    if (!ptr) {
        return ROCQ_STATUS_SUCCESS;
    }
    return (hipFree(ptr) == hipSuccess) ? ROCQ_STATUS_SUCCESS : ROCQ_STATUS_HIP_ERROR;
}

inline rocqStatus_t device_malloc(rocsvInternalHandle* handle, void** ptr, size_t bytes) {
    if (!handle) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (handle->memHandler.device_malloc) {
        return handle->memHandler.device_malloc(ptr, bytes, handle->memHandler.user_data);
    }
    return default_device_malloc(ptr, bytes);
}

inline rocqStatus_t device_free(rocsvInternalHandle* handle, void* ptr) {
    if (!handle) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (handle->memHandler.device_free) {
        return handle->memHandler.device_free(ptr, handle->memHandler.user_data);
    }
    return default_device_free(ptr);
}

inline rocqStatus_t copy_device_to_host(void* hostPtr,
                                        const void* devicePtr,
                                        size_t bytes,
                                        hipStream_t stream) {
    if (bytes == 0) {
        return ROCQ_STATUS_SUCCESS;
    }
    if (hipMemcpyAsync(hostPtr, devicePtr, bytes, hipMemcpyDeviceToHost, stream) != hipSuccess) {
        return ROCQ_STATUS_HIP_ERROR;
    }
    if (hipStreamSynchronize(stream) != hipSuccess) { // ROCQ_ASYNC_ALLOWED_SYNC
        return ROCQ_STATUS_HIP_ERROR;
    }
    return ROCQ_STATUS_SUCCESS;
}

inline rocqStatus_t copy_host_to_device(void* devicePtr,
                                        const void* hostPtr,
                                        size_t bytes,
                                        hipStream_t stream) {
    if (bytes == 0) {
        return ROCQ_STATUS_SUCCESS;
    }
    if (hipMemcpyAsync(devicePtr, hostPtr, bytes, hipMemcpyHostToDevice, stream) != hipSuccess) {
        return ROCQ_STATUS_HIP_ERROR;
    }
    return ROCQ_STATUS_SUCCESS;
}

inline rocqStatus_t validate_unique_qubits(const unsigned* qubits,
                                           unsigned count,
                                           unsigned numQubits,
                                           std::vector<unsigned>* out) {
    if (!out || count == 0 || !qubits) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    out->clear();
    out->reserve(count);
    std::vector<char> seen(numQubits, 0);
    for (unsigned i = 0; i < count; ++i) {
        const unsigned q = qubits[i];
        if (q >= numQubits || seen[q]) {
            return ROCQ_STATUS_INVALID_VALUE;
        }
        seen[q] = 1;
        out->push_back(q);
    }
    return ROCQ_STATUS_SUCCESS;
}

inline size_t effective_batch_size(rocsvInternalHandle* handle, rocComplex* state) {
    if (!handle) {
        return 1;
    }
    if (state && state != handle->d_state) {
        return 1;
    }
    return handle->batchSize > 0 ? handle->batchSize : 1;
}

inline double clamp_probability(double p) {
    if (p < 0.0) {
        return 0.0;
    }
    if (p > 1.0) {
        return 1.0;
    }
    return p;
}

inline bool is_effectively_zero(double x) {
    return std::abs(x) <= static_cast<double>(REAL_EPSILON);
}

inline bool is_power_of_two_int(int value) {
    return value > 0 && ((value & (value - 1)) == 0);
}

inline unsigned integer_log2(unsigned value) {
    unsigned result = 0;
    while (value > 1) {
        value >>= 1;
        ++result;
    }
    return result;
}

inline bool uses_distributed_state(const rocsvInternalHandle* handle, const rocComplex* external_state) {
    return handle && handle->distributedMode && external_state == nullptr;
}

inline bool distributed_qubit_local(const rocsvInternalHandle* handle, unsigned qubit) {
    return handle && qubit < handle->numLocalQubitsPerGpu;
}

inline bool distributed_all_qubits_local(const rocsvInternalHandle* handle,
                                         const std::vector<unsigned>& qubits) {
    if (!handle) {
        return false;
    }
    for (unsigned q : qubits) {
        if (!distributed_qubit_local(handle, q)) {
            return false;
        }
    }
    return true;
}

inline void reset_distributed_metadata(rocsvInternalHandle* handle) {
    if (!handle) {
        return;
    }
    handle->distributedMode = false;
    handle->distributedGpuCount = 0;
    handle->globalNumQubits = 0;
    handle->numLocalQubitsPerGpu = 0;
    handle->numGlobalSliceQubits = 0;
    handle->localSliceElements = 0;
    handle->distributedDeviceIds.clear();
    handle->distributedStreams.clear();
    handle->distributedSlices.clear();
    handle->distributedSwapBuffers.clear();
    handle->distributedTargetScratch.clear();
    handle->distributedMatrixScratch.clear();
#ifdef ROCQ_HAVE_RCCL
    handle->distributedComms.clear();
    handle->distributedRcclReady = false;
#endif
}

inline rocqStatus_t sync_distributed_streams(rocsvInternalHandle* handle) {
    if (!handle) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    const bool have_restore_device = !handle->distributedDeviceIds.empty();
    const int restore_device = have_restore_device ? handle->distributedDeviceIds[0] : 0;
    for (size_t rank = 0; rank < handle->distributedStreams.size(); ++rank) {
        if (hipSetDevice(handle->distributedDeviceIds[rank]) != hipSuccess) {
            return ROCQ_STATUS_HIP_ERROR;
        }
        if (hipStreamSynchronize(handle->distributedStreams[rank]) != hipSuccess) { // ROCQ_ASYNC_ALLOWED_SYNC
            return ROCQ_STATUS_HIP_ERROR;
        }
    }
    if (have_restore_device) {
        (void)hipSetDevice(restore_device);
    }
    return ROCQ_STATUS_SUCCESS;
}

inline rocqStatus_t clear_distributed_rccl_comms(rocsvInternalHandle* handle) {
#ifdef ROCQ_HAVE_RCCL
    if (!handle) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    int original_device = 0;
    const bool have_original_device = (hipGetDevice(&original_device) == hipSuccess);
    for (size_t rank = 0; rank < handle->distributedComms.size(); ++rank) {
        if (rank < handle->distributedDeviceIds.size() &&
            hipSetDevice(handle->distributedDeviceIds[rank]) != hipSuccess) {
            if (have_original_device) {
                (void)hipSetDevice(original_device);
            }
            return ROCQ_STATUS_HIP_ERROR;
        }
        ncclComm_t comm = handle->distributedComms[rank];
        if (comm && ncclCommDestroy(comm) != ncclSuccess) {
            if (have_original_device) {
                (void)hipSetDevice(original_device);
            }
            return ROCQ_STATUS_RCCL_ERROR;
        }
    }
    if (have_original_device) {
        (void)hipSetDevice(original_device);
    }
    handle->distributedComms.clear();
    handle->distributedRcclReady = false;
#else
    (void)handle;
#endif
    return ROCQ_STATUS_SUCCESS;
}

inline rocqStatus_t clear_distributed_state_storage(rocsvInternalHandle* handle) {
    if (!handle) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    rocqStatus_t comm_status = clear_distributed_rccl_comms(handle);
    if (comm_status != ROCQ_STATUS_SUCCESS) {
        return comm_status;
    }

    int original_device = 0;
    const bool have_original_device = (hipGetDevice(&original_device) == hipSuccess);

    for (size_t rank = 0; rank < handle->distributedSlices.size(); ++rank) {
        if (hipSetDevice(handle->distributedDeviceIds[rank]) != hipSuccess) {
            return ROCQ_STATUS_HIP_ERROR;
        }
        if (handle->distributedSlices[rank]) {
            rocqStatus_t free_status = device_free(handle, handle->distributedSlices[rank]);
            if (free_status != ROCQ_STATUS_SUCCESS) {
                return free_status;
            }
        }
        if (handle->distributedSwapBuffers.size() > rank && handle->distributedSwapBuffers[rank]) {
            rocqStatus_t free_status = device_free(handle, handle->distributedSwapBuffers[rank]);
            if (free_status != ROCQ_STATUS_SUCCESS) {
                return free_status;
            }
        }
        if (handle->distributedTargetScratch.size() > rank && handle->distributedTargetScratch[rank]) {
            rocqStatus_t free_status = device_free(handle, handle->distributedTargetScratch[rank]);
            if (free_status != ROCQ_STATUS_SUCCESS) {
                return free_status;
            }
        }
        if (handle->distributedMatrixScratch.size() > rank && handle->distributedMatrixScratch[rank]) {
            rocqStatus_t free_status = device_free(handle, handle->distributedMatrixScratch[rank]);
            if (free_status != ROCQ_STATUS_SUCCESS) {
                return free_status;
            }
        }
    }

    for (size_t rank = 0; rank < handle->distributedStreams.size(); ++rank) {
        if (hipSetDevice(handle->distributedDeviceIds[rank]) != hipSuccess) {
            return ROCQ_STATUS_HIP_ERROR;
        }
        if (handle->distributedStreams[rank]) {
            if (hipStreamDestroy(handle->distributedStreams[rank]) != hipSuccess) {
                return ROCQ_STATUS_HIP_ERROR;
            }
        }
    }

    if (have_original_device) {
        (void)hipSetDevice(original_device);
    }
    reset_distributed_metadata(handle);
    return ROCQ_STATUS_SUCCESS;
}

inline bool distributed_rccl_ready(const rocsvInternalHandle* handle) {
#ifdef ROCQ_HAVE_RCCL
    return handle &&
           handle->distributedRcclReady &&
           handle->distributedComms.size() == static_cast<size_t>(handle->distributedGpuCount);
#else
    (void)handle;
    return false;
#endif
}

inline rocsvDistributedBackend_t distributed_active_backend(const rocsvInternalHandle* handle) {
    if (!handle || !handle->distributedMode) {
        return ROCSV_DISTRIBUTED_BACKEND_NONE;
    }
    if (distributed_rccl_ready(handle)) {
        return ROCSV_DISTRIBUTED_BACKEND_RCCL;
    }
    if (distributed_host_fallback_enabled()) {
        return ROCSV_DISTRIBUTED_BACKEND_HOST_FALLBACK;
    }
    return ROCSV_DISTRIBUTED_BACKEND_NONE;
}

inline rocqStatus_t initialize_distributed_rccl_comms(rocsvInternalHandle* handle) {
#ifdef ROCQ_HAVE_RCCL
    if (!handle || handle->distributedGpuCount <= 1) {
        return ROCQ_STATUS_SUCCESS;
    }
    if (distributed_rccl_disabled()) {
        return distributed_rccl_required() ? ROCQ_STATUS_RCCL_ERROR : ROCQ_STATUS_SUCCESS;
    }

    handle->distributedComms.assign(static_cast<size_t>(handle->distributedGpuCount), nullptr);
    std::vector<int> devices = handle->distributedDeviceIds;
    const ncclResult_t init_status =
        ncclCommInitAll(handle->distributedComms.data(), handle->distributedGpuCount, devices.data());
    if (init_status != ncclSuccess) {
        (void)clear_distributed_rccl_comms(handle);
        return distributed_rccl_required() ? ROCQ_STATUS_RCCL_ERROR : ROCQ_STATUS_SUCCESS;
    }
    handle->distributedRcclReady = true;
#else
    if (distributed_rccl_required()) {
        return ROCQ_STATUS_RCCL_ERROR;
    }
    (void)handle;
#endif
    return ROCQ_STATUS_SUCCESS;
}

inline rocqStatus_t gather_distributed_state_to_host(rocsvInternalHandle* handle,
                                                     std::vector<rocComplex>* host_state) {
    if (!handle || !host_state || !handle->distributedMode) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (handle->distributedGpuCount <= 0 || handle->localSliceElements == 0) {
        host_state->clear();
        return ROCQ_STATUS_SUCCESS;
    }

    const size_t local_elements = handle->localSliceElements;
    const size_t total_elements = static_cast<size_t>(handle->distributedGpuCount) * local_elements;
    host_state->assign(total_elements, make_complex(0.0, 0.0));

    int original_device = 0;
    if (hipGetDevice(&original_device) != hipSuccess) {
        return ROCQ_STATUS_HIP_ERROR;
    }

    for (int rank = 0; rank < handle->distributedGpuCount; ++rank) {
        if (hipSetDevice(handle->distributedDeviceIds[rank]) != hipSuccess) {
            (void)hipSetDevice(original_device);
            return ROCQ_STATUS_HIP_ERROR;
        }
        const rocComplex* src = handle->distributedSlices[rank];
        if (!src) {
            continue;
        }
        const size_t bytes = local_elements * sizeof(rocComplex);
        if (hipMemcpy(host_state->data() + static_cast<size_t>(rank) * local_elements,
                      src,
                      bytes,
                      hipMemcpyDeviceToHost) != hipSuccess) {
            (void)hipSetDevice(original_device);
            return ROCQ_STATUS_HIP_ERROR;
        }
    }

    if (hipSetDevice(original_device) != hipSuccess) {
        return ROCQ_STATUS_HIP_ERROR;
    }
    return ROCQ_STATUS_SUCCESS;
}

inline rocqStatus_t scatter_host_state_to_distributed(rocsvInternalHandle* handle,
                                                      const std::vector<rocComplex>& host_state) {
    if (!handle || !handle->distributedMode) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (handle->distributedGpuCount <= 0 || handle->localSliceElements == 0) {
        return ROCQ_STATUS_SUCCESS;
    }

    const size_t local_elements = handle->localSliceElements;
    const size_t expected_elements = static_cast<size_t>(handle->distributedGpuCount) * local_elements;
    if (host_state.size() != expected_elements) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    int original_device = 0;
    if (hipGetDevice(&original_device) != hipSuccess) {
        return ROCQ_STATUS_HIP_ERROR;
    }

    for (int rank = 0; rank < handle->distributedGpuCount; ++rank) {
        if (hipSetDevice(handle->distributedDeviceIds[rank]) != hipSuccess) {
            (void)hipSetDevice(original_device);
            return ROCQ_STATUS_HIP_ERROR;
        }

        rocComplex* dst = handle->distributedSlices[rank];
        if (!dst) {
            continue;
        }

        const size_t bytes = local_elements * sizeof(rocComplex);
        if (hipMemcpy(dst,
                      host_state.data() + static_cast<size_t>(rank) * local_elements,
                      bytes,
                      hipMemcpyHostToDevice) != hipSuccess) {
            (void)hipSetDevice(original_device);
            return ROCQ_STATUS_HIP_ERROR;
        }
    }

    if (hipSetDevice(original_device) != hipSuccess) {
        return ROCQ_STATUS_HIP_ERROR;
    }
    return ROCQ_STATUS_SUCCESS;
}

inline size_t swap_bits_host(size_t value, unsigned bit_pos1, unsigned bit_pos2) {
    const size_t bit1 = (value >> bit_pos1) & size_t{1};
    const size_t bit2 = (value >> bit_pos2) & size_t{1};
    if (bit1 != bit2) {
        value ^= (size_t{1} << bit_pos1);
        value ^= (size_t{1} << bit_pos2);
    }
    return value;
}

inline rocqStatus_t distributed_swap_bits_host_remap(rocsvInternalHandle* handle,
                                                     unsigned qubit_idx1,
                                                     unsigned qubit_idx2) {
    if (!handle || !handle->distributedMode) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (handle->localSliceElements == 0 || handle->distributedGpuCount <= 0) {
        return ROCQ_STATUS_SUCCESS;
    }

    std::vector<rocComplex> host_input;
    rocqStatus_t status = gather_distributed_state_to_host(handle, &host_input);
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }

    const size_t local_elements = handle->localSliceElements;
    const size_t local_mask = local_elements - 1;
    std::vector<rocComplex> host_output(host_input.size(), make_complex(0.0, 0.0));

    for (int rank = 0; rank < handle->distributedGpuCount; ++rank) {
        const size_t rank_offset = static_cast<size_t>(rank) * local_elements;
        for (size_t local_idx = 0; local_idx < local_elements; ++local_idx) {
            const size_t global_idx =
                (static_cast<size_t>(rank) << handle->numLocalQubitsPerGpu) | local_idx;
            const size_t swapped_idx = swap_bits_host(global_idx, qubit_idx1, qubit_idx2);
            const int dst_rank = static_cast<int>(swapped_idx >> handle->numLocalQubitsPerGpu);
            const size_t dst_local = swapped_idx & local_mask;
            host_output[static_cast<size_t>(dst_rank) * local_elements + dst_local] =
                host_input[rank_offset + local_idx];
        }
    }

    return scatter_host_state_to_distributed(handle, host_output);
}

inline rocqStatus_t reduce_prob0_on_slice(rocsvInternalHandle* handle,
                                          const rocComplex* d_slice,
                                          size_t slice_elements,
                                          unsigned target_qubit,
                                          hipStream_t stream,
                                          double* prob0_out) {
    if (!handle || !d_slice || !prob0_out || slice_elements == 0) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    constexpr int threads_per_block = 256;
    const int blocks = compute_reduction_blocks(slice_elements, threads_per_block);
    if (blocks <= 0) {
        *prob0_out = 0.0;
        return ROCQ_STATUS_SUCCESS;
    }

    void* d_block_sums_void = nullptr;
    rocqStatus_t status = device_malloc(handle,
                                        &d_block_sums_void,
                                        static_cast<size_t>(blocks) * sizeof(double));
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }
    double* d_block_sums = static_cast<double*>(d_block_sums_void);

    hipLaunchKernelGGL(reduce_measure_prob0_kernel,
                       dim3(blocks),
                       dim3(threads_per_block),
                       threads_per_block * sizeof(double),
                       stream,
                       d_slice,
                       slice_elements,
                       target_qubit,
                       d_block_sums);
    status = check_last_hip_error();
    if (status != ROCQ_STATUS_SUCCESS) {
        device_free(handle, d_block_sums);
        return status;
    }

    double* current_in = d_block_sums;
    size_t current_count = static_cast<size_t>(blocks);
    bool owns_current = false;

    while (current_count > 1) {
        const int next_blocks = compute_reduction_blocks(current_count, threads_per_block);
        void* d_next_void = nullptr;
        status = device_malloc(handle,
                               &d_next_void,
                               static_cast<size_t>(next_blocks) * sizeof(double));
        if (status != ROCQ_STATUS_SUCCESS) {
            if (owns_current) {
                device_free(handle, current_in);
            }
            device_free(handle, d_block_sums);
            return status;
        }
        double* d_next = static_cast<double*>(d_next_void);
        hipLaunchKernelGGL(reduce_double_sum_kernel,
                           dim3(next_blocks),
                           dim3(threads_per_block),
                           threads_per_block * sizeof(double),
                           stream,
                           current_in,
                           current_count,
                           d_next);
        status = check_last_hip_error();
        if (status != ROCQ_STATUS_SUCCESS) {
            device_free(handle, d_next);
            if (owns_current) {
                device_free(handle, current_in);
            }
            device_free(handle, d_block_sums);
            return status;
        }

        if (owns_current) {
            device_free(handle, current_in);
        }
        current_in = d_next;
        current_count = static_cast<size_t>(next_blocks);
        owns_current = true;
    }

    if (hipMemcpyAsync(prob0_out, current_in, sizeof(double), hipMemcpyDeviceToHost, stream) != hipSuccess) {
        if (owns_current) {
            device_free(handle, current_in);
        }
        device_free(handle, d_block_sums);
        return ROCQ_STATUS_HIP_ERROR;
    }
    if (hipStreamSynchronize(stream) != hipSuccess) { // ROCQ_ASYNC_ALLOWED_SYNC
        if (owns_current) {
            device_free(handle, current_in);
        }
        device_free(handle, d_block_sums);
        return ROCQ_STATUS_HIP_ERROR;
    }

    if (owns_current) {
        device_free(handle, current_in);
    }
    device_free(handle, d_block_sums);
    return ROCQ_STATUS_SUCCESS;
}

rocqStatus_t apply_matrix_distributed_host_fallback(rocsvInternalHandle* handle,
                                                    unsigned numQubits,
                                                    const std::vector<unsigned>& targetQubits,
                                                    const std::vector<unsigned>& controlQubits,
                                                    const std::vector<rocComplex>& matrixHost);

rocqStatus_t launch_single_qubit_matrix(rocsvInternalHandle* handle,
                                        rocComplex* state,
                                        unsigned numQubits,
                                        unsigned targetQubit,
                                        const rocComplex& m00,
                                        const rocComplex& m01,
                                        const rocComplex& m10,
                                        const rocComplex& m11);

rocqStatus_t launch_controlled_single_qubit_matrix(rocsvInternalHandle* handle,
                                                   rocComplex* state,
                                                   unsigned numQubits,
                                                   unsigned controlQubit,
                                                   unsigned targetQubit,
                                                   const rocComplex& m00,
                                                   const rocComplex& m01,
                                                   const rocComplex& m10,
                                                   const rocComplex& m11);

inline rocqStatus_t restore_distributed_qubit_swaps(
    rocsvInternalHandle* handle,
    const std::vector<std::pair<unsigned, unsigned>>& swaps) {
    rocqStatus_t first_status = ROCQ_STATUS_SUCCESS;
    for (auto it = swaps.rbegin(); it != swaps.rend(); ++it) {
        rocqStatus_t status = rocsvSwapIndexBits(handle, it->first, it->second);
        if (first_status == ROCQ_STATUS_SUCCESS && status != ROCQ_STATUS_SUCCESS) {
            first_status = status;
        }
    }
    return first_status;
}

inline bool choose_distributed_local_slot(const rocsvInternalHandle* handle,
                                          const std::vector<unsigned>& reserved,
                                          unsigned* slot) {
    if (!handle || !slot) {
        return false;
    }
    for (unsigned q = 0; q < handle->numLocalQubitsPerGpu; ++q) {
        if (std::find(reserved.begin(), reserved.end(), q) == reserved.end()) {
            *slot = q;
            return true;
        }
    }
    return false;
}

inline rocqStatus_t launch_single_qubit_matrix_distributed_localized(
    rocsvInternalHandle* handle,
    rocComplex* state,
    unsigned numQubits,
    unsigned targetQubit,
    const rocComplex& m00,
    const rocComplex& m01,
    const rocComplex& m10,
    const rocComplex& m11) {
    unsigned local_target = targetQubit;
    std::vector<std::pair<unsigned, unsigned>> swaps;

    if (!distributed_qubit_local(handle, targetQubit)) {
        if (handle->numLocalQubitsPerGpu == 0) {
            const std::vector<rocComplex> matrix_host = {m00, m10, m01, m11};
            return apply_matrix_distributed_host_fallback(handle, numQubits, {targetQubit}, {}, matrix_host);
        }
        local_target = 0;
        rocqStatus_t status = rocsvSwapIndexBits(handle, targetQubit, local_target);
        if (status != ROCQ_STATUS_SUCCESS) {
            return status;
        }
        swaps.emplace_back(targetQubit, local_target);
    }

    rocqStatus_t status = launch_single_qubit_matrix(
        handle, state, numQubits, local_target, m00, m01, m10, m11);
    rocqStatus_t restore_status = restore_distributed_qubit_swaps(handle, swaps);
    return status != ROCQ_STATUS_SUCCESS ? status : restore_status;
}

inline rocqStatus_t launch_controlled_single_qubit_matrix_distributed_localized(
    rocsvInternalHandle* handle,
    rocComplex* state,
    unsigned numQubits,
    unsigned controlQubit,
    unsigned targetQubit,
    const rocComplex& m00,
    const rocComplex& m01,
    const rocComplex& m10,
    const rocComplex& m11) {
    std::vector<unsigned> reserved;
    unsigned local_control = controlQubit;
    unsigned local_target = targetQubit;

    if (distributed_qubit_local(handle, controlQubit)) {
        reserved.push_back(controlQubit);
    }
    if (distributed_qubit_local(handle, targetQubit)) {
        reserved.push_back(targetQubit);
    }

    std::vector<std::pair<unsigned, unsigned>> swaps_to_apply;
    if (!distributed_qubit_local(handle, controlQubit)) {
        if (!choose_distributed_local_slot(handle, reserved, &local_control)) {
            const std::vector<rocComplex> matrix_host = {m00, m10, m01, m11};
            return apply_matrix_distributed_host_fallback(
                handle, numQubits, {targetQubit}, {controlQubit}, matrix_host);
        }
        reserved.push_back(local_control);
        swaps_to_apply.emplace_back(controlQubit, local_control);
    }
    if (!distributed_qubit_local(handle, targetQubit)) {
        if (!choose_distributed_local_slot(handle, reserved, &local_target)) {
            const std::vector<rocComplex> matrix_host = {m00, m10, m01, m11};
            return apply_matrix_distributed_host_fallback(
                handle, numQubits, {targetQubit}, {controlQubit}, matrix_host);
        }
        reserved.push_back(local_target);
        swaps_to_apply.emplace_back(targetQubit, local_target);
    }

    std::vector<std::pair<unsigned, unsigned>> applied_swaps;
    for (const auto& swap : swaps_to_apply) {
        rocqStatus_t status = rocsvSwapIndexBits(handle, swap.first, swap.second);
        if (status != ROCQ_STATUS_SUCCESS) {
            (void)restore_distributed_qubit_swaps(handle, applied_swaps);
            return status;
        }
        applied_swaps.push_back(swap);
    }

    rocqStatus_t status = launch_controlled_single_qubit_matrix(
        handle, state, numQubits, local_control, local_target, m00, m01, m10, m11);
    rocqStatus_t restore_status = restore_distributed_qubit_swaps(handle, applied_swaps);
    return status != ROCQ_STATUS_SUCCESS ? status : restore_status;
}

rocqStatus_t launch_single_qubit_matrix(rocsvInternalHandle* handle,
                                        rocComplex* state,
                                        unsigned numQubits,
                                        unsigned targetQubit,
                                        const rocComplex& m00,
                                        const rocComplex& m01,
                                        const rocComplex& m10,
                                        const rocComplex& m11) {
    if (!validate_qubit_index(targetQubit, numQubits)) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    const bool distributed_call = handle && handle->distributedMode && state == handle->d_state;
    if (distributed_call) {
        if (numQubits != handle->globalNumQubits) {
            return ROCQ_STATUS_INVALID_VALUE;
        }
        if (!distributed_qubit_local(handle, targetQubit)) {
            return launch_single_qubit_matrix_distributed_localized(
                handle, state, numQubits, targetQubit, m00, m01, m10, m11);
        }

        if (handle->localSliceElements < 2) {
            return ROCQ_STATUS_SUCCESS;
        }
        const size_t local_pairs = handle->localSliceElements >> 1;
        constexpr int threads_per_block = 256;
        const int blocks =
            static_cast<int>((local_pairs + threads_per_block - 1) / threads_per_block);

        for (int rank = 0; rank < handle->distributedGpuCount; ++rank) {
            if (hipSetDevice(handle->distributedDeviceIds[rank]) != hipSuccess) {
                return ROCQ_STATUS_HIP_ERROR;
            }
            hipLaunchKernelGGL(apply_single_qubit_matrix_kernel,
                               dim3(blocks > 0 ? blocks : 1),
                               dim3(threads_per_block),
                               0,
                               handle->distributedStreams[rank],
                               handle->distributedSlices[rank],
                               handle->numLocalQubitsPerGpu,
                               targetQubit,
                               m00,
                               m01,
                               m10,
                               m11,
                               1);
            rocqStatus_t status = check_last_hip_error();
            if (status != ROCQ_STATUS_SUCCESS) {
                return status;
            }
        }
        return ROCQ_STATUS_SUCCESS;
    }

    const size_t elements_per_state = 1ULL << numQubits;
    if (elements_per_state < 2) {
        return ROCQ_STATUS_SUCCESS;
    }

    const size_t pairs_per_state = elements_per_state >> 1;
    const size_t total_pairs = handle->batchSize * pairs_per_state;
    if (total_pairs == 0) {
        return ROCQ_STATUS_SUCCESS;
    }

    constexpr int threads_per_block = 256;
    const int blocks = static_cast<int>((total_pairs + threads_per_block - 1) / threads_per_block);

    hipLaunchKernelGGL(apply_single_qubit_matrix_kernel,
                       dim3(blocks > 0 ? blocks : 1),
                       dim3(threads_per_block),
                       0,
                       handle->streams[0],
                       state,
                       numQubits,
                       targetQubit,
                       m00,
                       m01,
                       m10,
                       m11,
                       handle->batchSize);
    return check_last_hip_error();
}

rocqStatus_t launch_single_qubit_matrix_batch(rocsvInternalHandle* handle,
                                              rocComplex* state,
                                              unsigned numQubits,
                                              unsigned targetQubit,
                                              const std::vector<rocComplex>& matrices) {
    if (!handle || !state || !validate_qubit_index(targetQubit, numQubits)) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    const size_t batch_size = effective_batch_size(handle, state);
    if (matrices.size() != batch_size * 4ULL) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    if (handle->distributedMode && state == handle->d_state) {
        if (batch_size != 1 || matrices.size() != 4ULL) {
            return ROCQ_STATUS_NOT_IMPLEMENTED;
        }
        return launch_single_qubit_matrix(handle,
                                          state,
                                          numQubits,
                                          targetQubit,
                                          matrices[0],
                                          matrices[1],
                                          matrices[2],
                                          matrices[3]);
    }

    size_t elements_per_state = 0;
    if (!compute_power_of_two(numQubits, &elements_per_state)) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (elements_per_state < 2) {
        return ROCQ_STATUS_SUCCESS;
    }

    const size_t pairs_per_state = elements_per_state >> 1;
    if (batch_size > std::numeric_limits<size_t>::max() / pairs_per_state) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    const size_t total_pairs = batch_size * pairs_per_state;
    if (total_pairs == 0) {
        return ROCQ_STATUS_SUCCESS;
    }

    void* d_matrices_void = nullptr;
    rocqStatus_t status = device_malloc(handle, &d_matrices_void, matrices.size() * sizeof(rocComplex));
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }
    rocComplex* d_matrices = static_cast<rocComplex*>(d_matrices_void);
    status = copy_host_to_device(d_matrices,
                                 matrices.data(),
                                 matrices.size() * sizeof(rocComplex),
                                 handle->streams[0]);
    if (status != ROCQ_STATUS_SUCCESS) {
        device_free(handle, d_matrices);
        return status;
    }

    constexpr int threads_per_block = 256;
    const int blocks = static_cast<int>((total_pairs + threads_per_block - 1) / threads_per_block);

    hipLaunchKernelGGL(apply_single_qubit_matrix_batch_kernel,
                       dim3(blocks > 0 ? blocks : 1),
                       dim3(threads_per_block),
                       0,
                       handle->streams[0],
                       state,
                       numQubits,
                       targetQubit,
                       d_matrices,
                       batch_size);
    status = check_last_hip_error();
    rocqStatus_t free_status = device_free(handle, d_matrices);
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }
    return free_status;
}

rocqStatus_t launch_controlled_single_qubit_matrix(rocsvInternalHandle* handle,
                                                   rocComplex* state,
                                                   unsigned numQubits,
                                                   unsigned controlQubit,
                                                   unsigned targetQubit,
                                                   const rocComplex& m00,
                                                   const rocComplex& m01,
                                                   const rocComplex& m10,
                                                   const rocComplex& m11) {
    if (!validate_qubit_index(controlQubit, numQubits) ||
        !validate_qubit_index(targetQubit, numQubits) ||
        controlQubit == targetQubit) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    const bool distributed_call = handle && handle->distributedMode && state == handle->d_state;
    if (distributed_call) {
        if (numQubits != handle->globalNumQubits) {
            return ROCQ_STATUS_INVALID_VALUE;
        }
        if (!distributed_qubit_local(handle, controlQubit) ||
            !distributed_qubit_local(handle, targetQubit)) {
            return launch_controlled_single_qubit_matrix_distributed_localized(
                handle, state, numQubits, controlQubit, targetQubit, m00, m01, m10, m11);
        }
        if (handle->localSliceElements < 2) {
            return ROCQ_STATUS_SUCCESS;
        }

        const size_t local_pairs = handle->localSliceElements >> 1;
        constexpr int threads_per_block = 256;
        const int blocks =
            static_cast<int>((local_pairs + threads_per_block - 1) / threads_per_block);
        for (int rank = 0; rank < handle->distributedGpuCount; ++rank) {
            if (hipSetDevice(handle->distributedDeviceIds[rank]) != hipSuccess) {
                return ROCQ_STATUS_HIP_ERROR;
            }
            hipLaunchKernelGGL(apply_controlled_single_qubit_matrix_kernel,
                               dim3(blocks > 0 ? blocks : 1),
                               dim3(threads_per_block),
                               0,
                               handle->distributedStreams[rank],
                               handle->distributedSlices[rank],
                               handle->numLocalQubitsPerGpu,
                               controlQubit,
                               targetQubit,
                               m00,
                               m01,
                               m10,
                               m11,
                               1);
            rocqStatus_t status = check_last_hip_error();
            if (status != ROCQ_STATUS_SUCCESS) {
                return status;
            }
        }
        return ROCQ_STATUS_SUCCESS;
    }

    const size_t elements_per_state = 1ULL << numQubits;
    if (elements_per_state < 2) {
        return ROCQ_STATUS_SUCCESS;
    }

    const size_t pairs_per_state = elements_per_state >> 1;
    const size_t total_pairs = handle->batchSize * pairs_per_state;
    if (total_pairs == 0) {
        return ROCQ_STATUS_SUCCESS;
    }

    constexpr int threads_per_block = 256;
    const int blocks = static_cast<int>((total_pairs + threads_per_block - 1) / threads_per_block);

    hipLaunchKernelGGL(apply_controlled_single_qubit_matrix_kernel,
                       dim3(blocks > 0 ? blocks : 1),
                       dim3(threads_per_block),
                       0,
                       handle->streams[0],
                       state,
                       numQubits,
                       controlQubit,
                       targetQubit,
                       m00,
                       m01,
                       m10,
                       m11,
                       handle->batchSize);
    return check_last_hip_error();
}

rocqStatus_t launch_controlled_single_qubit_matrix_batch(rocsvInternalHandle* handle,
                                                         rocComplex* state,
                                                         unsigned numQubits,
                                                         unsigned controlQubit,
                                                         unsigned targetQubit,
                                                         const std::vector<rocComplex>& matrices) {
    if (!handle || !state ||
        !validate_qubit_index(controlQubit, numQubits) ||
        !validate_qubit_index(targetQubit, numQubits) ||
        controlQubit == targetQubit) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    const size_t batch_size = effective_batch_size(handle, state);
    if (matrices.size() != batch_size * 4ULL) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    if (handle->distributedMode && state == handle->d_state) {
        if (batch_size != 1 || matrices.size() != 4ULL) {
            return ROCQ_STATUS_NOT_IMPLEMENTED;
        }
        return launch_controlled_single_qubit_matrix(handle,
                                                     state,
                                                     numQubits,
                                                     controlQubit,
                                                     targetQubit,
                                                     matrices[0],
                                                     matrices[1],
                                                     matrices[2],
                                                     matrices[3]);
    }

    size_t elements_per_state = 0;
    if (!compute_power_of_two(numQubits, &elements_per_state)) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (elements_per_state < 2) {
        return ROCQ_STATUS_SUCCESS;
    }

    const size_t pairs_per_state = elements_per_state >> 1;
    if (batch_size > std::numeric_limits<size_t>::max() / pairs_per_state) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    const size_t total_pairs = batch_size * pairs_per_state;
    if (total_pairs == 0) {
        return ROCQ_STATUS_SUCCESS;
    }

    void* d_matrices_void = nullptr;
    rocqStatus_t status = device_malloc(handle, &d_matrices_void, matrices.size() * sizeof(rocComplex));
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }
    rocComplex* d_matrices = static_cast<rocComplex*>(d_matrices_void);
    status = copy_host_to_device(d_matrices,
                                 matrices.data(),
                                 matrices.size() * sizeof(rocComplex),
                                 handle->streams[0]);
    if (status != ROCQ_STATUS_SUCCESS) {
        device_free(handle, d_matrices);
        return status;
    }

    constexpr int threads_per_block = 256;
    const int blocks = static_cast<int>((total_pairs + threads_per_block - 1) / threads_per_block);
    hipLaunchKernelGGL(apply_controlled_single_qubit_matrix_batch_kernel,
                       dim3(blocks > 0 ? blocks : 1),
                       dim3(threads_per_block),
                       0,
                       handle->streams[0],
                       state,
                       numQubits,
                       controlQubit,
                       targetQubit,
                       d_matrices,
                       batch_size);
    status = check_last_hip_error();
    rocqStatus_t free_status = device_free(handle, d_matrices);
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }
    return free_status;
}

inline rocqStatus_t copy_matrix_from_device(const rocComplex* matrixDevice,
                                            size_t matrixElements,
                                            hipStream_t stream,
                                            std::vector<rocComplex>* hostMatrix) {
    if (!matrixDevice || !hostMatrix) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    hostMatrix->resize(matrixElements);
    return copy_device_to_host(hostMatrix->data(),
                               matrixDevice,
                               matrixElements * sizeof(rocComplex),
                               stream);
}

inline rocqStatus_t copy_sparse_matrix_from_device(const rocComplex* dataDevice,
                                                   const size_t* indicesDevice,
                                                   const size_t* indptrDevice,
                                                   size_t rows,
                                                   size_t nnz,
                                                   hipStream_t stream,
                                                   std::vector<rocComplex>* hostData,
                                                   std::vector<size_t>* hostIndices,
                                                   std::vector<size_t>* hostIndptr) {
    if (!indptrDevice || !hostData || !hostIndices || !hostIndptr ||
        rows == std::numeric_limits<size_t>::max()) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (nnz > 0 && (!dataDevice || !indicesDevice)) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    hostData->assign(nnz, make_complex(0.0, 0.0));
    hostIndices->assign(nnz, 0);
    hostIndptr->assign(rows + 1, 0);

    rocqStatus_t status = ROCQ_STATUS_SUCCESS;
    if (nnz > 0) {
        status = copy_device_to_host(hostData->data(),
                                     dataDevice,
                                     nnz * sizeof(rocComplex),
                                     stream);
        if (status != ROCQ_STATUS_SUCCESS) {
            return status;
        }

        status = copy_device_to_host(hostIndices->data(),
                                     indicesDevice,
                                     nnz * sizeof(size_t),
                                     stream);
        if (status != ROCQ_STATUS_SUCCESS) {
            return status;
        }
    }

    return copy_device_to_host(hostIndptr->data(),
                               indptrDevice,
                               hostIndptr->size() * sizeof(size_t),
                               stream);
}

inline rocqStatus_t validate_sparse_matrix_host_csr(const std::vector<rocComplex>& data,
                                                    const std::vector<size_t>& indices,
                                                    const std::vector<size_t>& indptr,
                                                    size_t rows,
                                                    size_t cols,
                                                    size_t nnz) {
    if (rows == 0 || cols == 0 || rows != cols ||
        data.size() != nnz || indices.size() != nnz || indptr.size() != rows + 1) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (indptr.empty() || indptr.front() != 0 || indptr.back() != nnz) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    for (size_t row = 0; row < rows; ++row) {
        if (indptr[row] > indptr[row + 1]) {
            return ROCQ_STATUS_INVALID_VALUE;
        }
    }
    for (size_t col : indices) {
        if (col >= cols) {
            return ROCQ_STATUS_INVALID_VALUE;
        }
    }
    return ROCQ_STATUS_SUCCESS;
}

inline size_t compose_basis_index(size_t baseIdx,
                                  size_t targetConfiguration,
                                  const std::vector<unsigned>& targetQubits) {
    size_t idx = baseIdx;
    for (size_t bit = 0; bit < targetQubits.size(); ++bit) {
        if ((targetConfiguration >> bit) & 1ULL) {
            idx |= (size_t{1} << targetQubits[bit]);
        }
    }
    return idx;
}

rocqStatus_t apply_matrix_host_impl(rocComplex* d_state,
                                    unsigned numQubits,
                                    const std::vector<unsigned>& targetQubits,
                                    const std::vector<unsigned>& controlQubits,
                                    const std::vector<rocComplex>& matrixHost,
                                    size_t batchSize,
                                    size_t stateStride,
                                    hipStream_t stream) {
    if (!d_state || targetQubits.empty()) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    size_t stateElements = 0;
    if (!compute_power_of_two(numQubits, &stateElements)) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (stateStride < stateElements) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    const size_t matrixDim = size_t{1} << targetQubits.size();
    if (matrixHost.size() != matrixDim * matrixDim) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    std::vector<char> isTarget(numQubits, 0);
    for (unsigned q : targetQubits) {
        isTarget[q] = 1;
    }
    for (unsigned q : controlQubits) {
        if (isTarget[q]) {
            return ROCQ_STATUS_INVALID_VALUE;
        }
    }

    std::vector<unsigned> nonTargetQubits;
    nonTargetQubits.reserve(numQubits - static_cast<unsigned>(targetQubits.size()));
    for (unsigned q = 0; q < numQubits; ++q) {
        if (!isTarget[q]) {
            nonTargetQubits.push_back(q);
        }
    }

    size_t nonTargetConfigs = 0;
    if (!compute_power_of_two(static_cast<unsigned>(nonTargetQubits.size()), &nonTargetConfigs)) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    std::vector<std::complex<double>> matrixStd(matrixHost.size());
    for (size_t i = 0; i < matrixHost.size(); ++i) {
        matrixStd[i] = to_std_complex(matrixHost[i]);
    }

    std::vector<rocComplex> batchState(stateElements);
    std::vector<std::complex<double>> inputAmps(matrixDim);
    std::vector<std::complex<double>> outputAmps(matrixDim);

    for (size_t batch = 0; batch < batchSize; ++batch) {
        rocComplex* batchPtr = d_state + batch * stateStride;
        rocqStatus_t status = copy_device_to_host(batchState.data(),
                                                  batchPtr,
                                                  stateElements * sizeof(rocComplex),
                                                  stream);
        if (status != ROCQ_STATUS_SUCCESS) {
            return status;
        }

        for (size_t config = 0; config < nonTargetConfigs; ++config) {
            size_t baseIdx = 0;
            for (size_t bit = 0; bit < nonTargetQubits.size(); ++bit) {
                if ((config >> bit) & 1ULL) {
                    baseIdx |= (size_t{1} << nonTargetQubits[bit]);
                }
            }

            bool controlsEnabled = true;
            for (unsigned ctrl : controlQubits) {
                if (((baseIdx >> ctrl) & 1ULL) == 0ULL) {
                    controlsEnabled = false;
                    break;
                }
            }
            if (!controlsEnabled) {
                continue;
            }

            for (size_t col = 0; col < matrixDim; ++col) {
                const size_t idx = compose_basis_index(baseIdx, col, targetQubits);
                inputAmps[col] = to_std_complex(batchState[idx]);
            }

            for (size_t row = 0; row < matrixDim; ++row) {
                std::complex<double> accum{0.0, 0.0};
                for (size_t col = 0; col < matrixDim; ++col) {
                    accum += matrixStd[row + col * matrixDim] * inputAmps[col];
                }
                outputAmps[row] = accum;
            }

            for (size_t row = 0; row < matrixDim; ++row) {
                const size_t idx = compose_basis_index(baseIdx, row, targetQubits);
                batchState[idx] = from_std_complex(outputAmps[row]);
            }
        }

        status = copy_host_to_device(batchPtr,
                                     batchState.data(),
                                     stateElements * sizeof(rocComplex),
                                     stream);
        if (status != ROCQ_STATUS_SUCCESS) {
            return status;
        }
    }

    return ROCQ_STATUS_SUCCESS;
}

rocqStatus_t apply_matrix_host_state_impl(std::vector<rocComplex>* host_state,
                                          unsigned numQubits,
                                          const std::vector<unsigned>& targetQubits,
                                          const std::vector<unsigned>& controlQubits,
                                          const std::vector<rocComplex>& matrixHost) {
    if (!host_state || targetQubits.empty()) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    size_t stateElements = 0;
    if (!compute_power_of_two(numQubits, &stateElements) || host_state->size() != stateElements) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    const size_t matrixDim = size_t{1} << targetQubits.size();
    if (matrixHost.size() != matrixDim * matrixDim) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    std::vector<char> isTarget(numQubits, 0);
    for (unsigned q : targetQubits) {
        if (!validate_qubit_index(q, numQubits)) {
            return ROCQ_STATUS_INVALID_VALUE;
        }
        isTarget[q] = 1;
    }
    for (unsigned q : controlQubits) {
        if (!validate_qubit_index(q, numQubits)) {
            return ROCQ_STATUS_INVALID_VALUE;
        }
        if (isTarget[q]) {
            return ROCQ_STATUS_INVALID_VALUE;
        }
    }

    std::vector<unsigned> nonTargetQubits;
    nonTargetQubits.reserve(numQubits - static_cast<unsigned>(targetQubits.size()));
    for (unsigned q = 0; q < numQubits; ++q) {
        if (!isTarget[q]) {
            nonTargetQubits.push_back(q);
        }
    }

    size_t nonTargetConfigs = 0;
    if (!compute_power_of_two(static_cast<unsigned>(nonTargetQubits.size()), &nonTargetConfigs)) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    std::vector<std::complex<double>> matrixStd(matrixHost.size());
    for (size_t i = 0; i < matrixHost.size(); ++i) {
        matrixStd[i] = to_std_complex(matrixHost[i]);
    }

    std::vector<std::complex<double>> inputAmps(matrixDim);
    std::vector<std::complex<double>> outputAmps(matrixDim);

    for (size_t config = 0; config < nonTargetConfigs; ++config) {
        size_t baseIdx = 0;
        for (size_t bit = 0; bit < nonTargetQubits.size(); ++bit) {
            if ((config >> bit) & 1ULL) {
                baseIdx |= (size_t{1} << nonTargetQubits[bit]);
            }
        }

        bool controlsEnabled = true;
        for (unsigned ctrl : controlQubits) {
            if (((baseIdx >> ctrl) & 1ULL) == 0ULL) {
                controlsEnabled = false;
                break;
            }
        }
        if (!controlsEnabled) {
            continue;
        }

        for (size_t col = 0; col < matrixDim; ++col) {
            const size_t idx = compose_basis_index(baseIdx, col, targetQubits);
            inputAmps[col] = to_std_complex((*host_state)[idx]);
        }

        for (size_t row = 0; row < matrixDim; ++row) {
            std::complex<double> accum{0.0, 0.0};
            for (size_t col = 0; col < matrixDim; ++col) {
                accum += matrixStd[row + col * matrixDim] * inputAmps[col];
            }
            outputAmps[row] = accum;
        }

        for (size_t row = 0; row < matrixDim; ++row) {
            const size_t idx = compose_basis_index(baseIdx, row, targetQubits);
            (*host_state)[idx] = from_std_complex(outputAmps[row]);
        }
    }

    return ROCQ_STATUS_SUCCESS;
}

rocqStatus_t compute_expectation_matrix_host_state(const std::vector<rocComplex>& host_state,
                                                   unsigned numQubits,
                                                   const std::vector<unsigned>& targetQubits,
                                                   const std::vector<rocComplex>& matrixHost,
                                                   size_t matrixDim,
                                                   rocComplex* result) {
    if (!result || targetQubits.empty()) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    size_t stateElements = 0;
    if (!compute_power_of_two(numQubits, &stateElements) || host_state.size() != stateElements) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    size_t expectedMatrixDim = 0;
    if (!compute_power_of_two(static_cast<unsigned>(targetQubits.size()), &expectedMatrixDim) ||
        matrixDim != expectedMatrixDim ||
        matrixHost.size() != matrixDim * matrixDim) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    for (unsigned q : targetQubits) {
        if (!validate_qubit_index(q, numQubits)) {
            return ROCQ_STATUS_INVALID_VALUE;
        }
    }

    double total_re = 0.0;
    double total_im = 0.0;
    for (size_t rowIndex = 0; rowIndex < stateElements; ++rowIndex) {
        size_t row_target = 0;
        size_t base_index = rowIndex;
        for (unsigned bit = 0; bit < targetQubits.size(); ++bit) {
            const unsigned qubit = targetQubits[bit];
            const size_t mask = size_t{1} << qubit;
            if ((rowIndex & mask) != 0ULL) {
                row_target |= (size_t{1} << bit);
            }
            base_index &= ~mask;
        }

        const rocComplex bra = host_state[rowIndex];
        for (size_t col_target = 0; col_target < matrixDim; ++col_target) {
            const size_t col_index = compose_basis_index(base_index, col_target, targetQubits);
            const rocComplex m = matrixHost[row_target + col_target * matrixDim];
            const rocComplex ket = host_state[col_index];
            const double mv_re = static_cast<double>(m.x) * static_cast<double>(ket.x) -
                                 static_cast<double>(m.y) * static_cast<double>(ket.y);
            const double mv_im = static_cast<double>(m.x) * static_cast<double>(ket.y) +
                                 static_cast<double>(m.y) * static_cast<double>(ket.x);

            total_re += static_cast<double>(bra.x) * mv_re + static_cast<double>(bra.y) * mv_im;
            total_im += static_cast<double>(bra.x) * mv_im - static_cast<double>(bra.y) * mv_re;
        }
    }

    *result = make_complex(total_re, total_im);
    return ROCQ_STATUS_SUCCESS;
}

rocqStatus_t apply_sparse_matrix_host_state_impl(std::vector<rocComplex>* host_state,
                                                 unsigned numQubits,
                                                 const std::vector<unsigned>& targetQubits,
                                                 const std::vector<rocComplex>& data,
                                                 const std::vector<size_t>& indices,
                                                 const std::vector<size_t>& indptr,
                                                 size_t rows,
                                                 size_t cols) {
    if (!host_state || targetQubits.empty()) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    size_t stateElements = 0;
    if (!compute_power_of_two(numQubits, &stateElements) || host_state->size() != stateElements) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    size_t matrixDim = 0;
    if (!compute_power_of_two(static_cast<unsigned>(targetQubits.size()), &matrixDim) ||
        rows != matrixDim || cols != matrixDim) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    rocqStatus_t status =
        validate_sparse_matrix_host_csr(data, indices, indptr, rows, cols, data.size());
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }

    std::vector<char> isTarget(numQubits, 0);
    for (unsigned q : targetQubits) {
        if (!validate_qubit_index(q, numQubits)) {
            return ROCQ_STATUS_INVALID_VALUE;
        }
        isTarget[q] = 1;
    }

    std::vector<unsigned> nonTargetQubits;
    nonTargetQubits.reserve(numQubits - static_cast<unsigned>(targetQubits.size()));
    for (unsigned q = 0; q < numQubits; ++q) {
        if (!isTarget[q]) {
            nonTargetQubits.push_back(q);
        }
    }

    size_t nonTargetConfigs = 0;
    if (!compute_power_of_two(static_cast<unsigned>(nonTargetQubits.size()), &nonTargetConfigs)) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    std::vector<rocComplex> output_state(stateElements, make_complex(0.0, 0.0));
    for (size_t config = 0; config < nonTargetConfigs; ++config) {
        size_t baseIdx = 0;
        for (size_t bit = 0; bit < nonTargetQubits.size(); ++bit) {
            if ((config >> bit) & 1ULL) {
                baseIdx |= (size_t{1} << nonTargetQubits[bit]);
            }
        }

        for (size_t local_row = 0; local_row < rows; ++local_row) {
            double accum_re = 0.0;
            double accum_im = 0.0;
            for (size_t offset = indptr[local_row]; offset < indptr[local_row + 1]; ++offset) {
                const size_t col_index = compose_basis_index(baseIdx, indices[offset], targetQubits);
                const rocComplex ket = (*host_state)[col_index];
                const rocComplex value = data[offset];
                accum_re += static_cast<double>(value.x) * static_cast<double>(ket.x) -
                            static_cast<double>(value.y) * static_cast<double>(ket.y);
                accum_im += static_cast<double>(value.x) * static_cast<double>(ket.y) +
                            static_cast<double>(value.y) * static_cast<double>(ket.x);
            }
            const size_t row_index = compose_basis_index(baseIdx, local_row, targetQubits);
            output_state[row_index] = make_complex(accum_re, accum_im);
        }
    }

    *host_state = std::move(output_state);
    return ROCQ_STATUS_SUCCESS;
}

rocqStatus_t compute_sparse_matrix_moments_host_state(const std::vector<rocComplex>& host_state,
                                                      unsigned numQubits,
                                                      const std::vector<rocComplex>& data,
                                                      const std::vector<size_t>& indices,
                                                      const std::vector<size_t>& indptr,
                                                      size_t rows,
                                                      size_t cols,
                                                      size_t nnz,
                                                      rocComplex* mean,
                                                      rocComplex* secondMoment) {
    if (!mean || !secondMoment) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    size_t stateElements = 0;
    if (!compute_power_of_two(numQubits, &stateElements) ||
        host_state.size() != stateElements || rows != stateElements || cols != stateElements) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    rocqStatus_t status = validate_sparse_matrix_host_csr(data, indices, indptr, rows, cols, nnz);
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }

    double mean_re = 0.0;
    double mean_im = 0.0;
    double second_re = 0.0;
    for (size_t row = 0; row < rows; ++row) {
        double accum_re = 0.0;
        double accum_im = 0.0;
        for (size_t offset = indptr[row]; offset < indptr[row + 1]; ++offset) {
            const rocComplex value = data[offset];
            const rocComplex ket = host_state[indices[offset]];
            accum_re += static_cast<double>(value.x) * static_cast<double>(ket.x) -
                        static_cast<double>(value.y) * static_cast<double>(ket.y);
            accum_im += static_cast<double>(value.x) * static_cast<double>(ket.y) +
                        static_cast<double>(value.y) * static_cast<double>(ket.x);
        }

        const rocComplex bra = host_state[row];
        mean_re += static_cast<double>(bra.x) * accum_re + static_cast<double>(bra.y) * accum_im;
        mean_im += static_cast<double>(bra.x) * accum_im - static_cast<double>(bra.y) * accum_re;
        second_re += accum_re * accum_re + accum_im * accum_im;
    }

    *mean = make_complex(mean_re, mean_im);
    *secondMoment = make_complex(second_re, 0.0);
    return ROCQ_STATUS_SUCCESS;
}

rocqStatus_t apply_matrix_distributed_host_fallback(rocsvInternalHandle* handle,
                                                    unsigned numQubits,
                                                    const std::vector<unsigned>& targetQubits,
                                                    const std::vector<unsigned>& controlQubits,
                                                    const std::vector<rocComplex>& matrixHost) {
    if (!distributed_host_fallback_enabled()) {
        return ROCQ_STATUS_NOT_IMPLEMENTED;
    }
    if (!handle || !handle->distributedMode || numQubits != handle->globalNumQubits) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    std::vector<rocComplex> host_full;
    rocqStatus_t status = gather_distributed_state_to_host(handle, &host_full);
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }

    status = apply_matrix_host_state_impl(&host_full,
                                          numQubits,
                                          targetQubits,
                                          controlQubits,
                                          matrixHost);
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }

    return scatter_host_state_to_distributed(handle, host_full);
}

rocqStatus_t expectation_matrix_distributed_host_fallback(rocsvInternalHandle* handle,
                                                          unsigned numQubits,
                                                          const std::vector<unsigned>& targetQubits,
                                                          const rocComplex* d_matrix,
                                                          size_t matrixDim,
                                                          rocComplex* result) {
    if (!distributed_host_fallback_enabled()) {
        return ROCQ_STATUS_NOT_IMPLEMENTED;
    }
    if (!handle || !handle->distributedMode || numQubits != handle->globalNumQubits ||
        handle->distributedDeviceIds.empty() || !d_matrix || !result) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (matrixDim == 0 || matrixDim > std::numeric_limits<size_t>::max() / matrixDim) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    int original_device = 0;
    if (hipGetDevice(&original_device) != hipSuccess) {
        return ROCQ_STATUS_HIP_ERROR;
    }
    auto restore_device = [&]() {
        (void)hipSetDevice(original_device);
    };

    if (hipSetDevice(handle->distributedDeviceIds[0]) != hipSuccess) {
        restore_device();
        return ROCQ_STATUS_HIP_ERROR;
    }

    std::vector<rocComplex> matrix_host;
    rocqStatus_t status = copy_matrix_from_device(d_matrix,
                                                  matrixDim * matrixDim,
                                                  handle->streams[0],
                                                  &matrix_host);
    restore_device();
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }

    std::vector<rocComplex> host_full;
    status = gather_distributed_state_to_host(handle, &host_full);
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }

    return compute_expectation_matrix_host_state(host_full,
                                                 numQubits,
                                                 targetQubits,
                                                 matrix_host,
                                                 matrixDim,
                                                 result);
}

rocqStatus_t expectation_matrix_host_fallback(rocsvInternalHandle* handle,
                                              rocComplex* state,
                                              unsigned numQubits,
                                              const std::vector<unsigned>& targetQubits,
                                              const rocComplex* d_matrix,
                                              size_t matrixDim,
                                              rocComplex* result) {
    if (!allow_host_matrix_fallback()) {
        return ROCQ_STATUS_NOT_IMPLEMENTED;
    }
    if (!handle || !state || !d_matrix || !result) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (effective_batch_size(handle, state) != 1) {
        return ROCQ_STATUS_NOT_IMPLEMENTED;
    }
    if (matrixDim == 0 || matrixDim > std::numeric_limits<size_t>::max() / matrixDim) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    std::vector<rocComplex> matrix_host;
    rocqStatus_t status = copy_matrix_from_device(d_matrix,
                                                  matrixDim * matrixDim,
                                                  handle->streams[0],
                                                  &matrix_host);
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }

    size_t state_elements = 0;
    if (!compute_power_of_two(numQubits, &state_elements)) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    std::vector<rocComplex> host_state(state_elements);
    status = copy_device_to_host(host_state.data(),
                                 state,
                                 state_elements * sizeof(rocComplex),
                                 handle->streams[0]);
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }

    return compute_expectation_matrix_host_state(host_state,
                                                 numQubits,
                                                 targetQubits,
                                                 matrix_host,
                                                 matrixDim,
                                                 result);
}

rocqStatus_t expectation_matrix_batch_host_fallback(rocsvInternalHandle* handle,
                                                    rocComplex* state,
                                                    unsigned numQubits,
                                                    const std::vector<unsigned>& targetQubits,
                                                    const rocComplex* d_matrix,
                                                    size_t matrixDim,
                                                    rocComplex* results) {
    if (!allow_host_matrix_fallback()) {
        return ROCQ_STATUS_NOT_IMPLEMENTED;
    }
    if (!handle || !state || !d_matrix || !results) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (matrixDim == 0 || matrixDim > std::numeric_limits<size_t>::max() / matrixDim) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    const size_t batch_size = effective_batch_size(handle, state);
    std::vector<rocComplex> matrix_host;
    rocqStatus_t status = copy_matrix_from_device(d_matrix,
                                                  matrixDim * matrixDim,
                                                  handle->streams[0],
                                                  &matrix_host);
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }

    size_t state_elements = 0;
    if (!compute_power_of_two(numQubits, &state_elements)) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (batch_size > std::numeric_limits<size_t>::max() / state_elements) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    std::vector<rocComplex> host_state(state_elements);
    for (size_t batch = 0; batch < batch_size; ++batch) {
        status = copy_device_to_host(host_state.data(),
                                     state + batch * state_elements,
                                     state_elements * sizeof(rocComplex),
                                     handle->streams[0]);
        if (status != ROCQ_STATUS_SUCCESS) {
            return status;
        }
        status = compute_expectation_matrix_host_state(host_state,
                                                       numQubits,
                                                       targetQubits,
                                                       matrix_host,
                                                       matrixDim,
                                                       results + batch);
        if (status != ROCQ_STATUS_SUCCESS) {
            return status;
        }
    }
    return ROCQ_STATUS_SUCCESS;
}

rocqStatus_t sparse_matrix_moments_distributed_host_fallback(rocsvInternalHandle* handle,
                                                             unsigned numQubits,
                                                             const rocComplex* d_data,
                                                             const size_t* d_indices,
                                                             const size_t* d_indptr,
                                                             size_t rows,
                                                             size_t cols,
                                                             size_t nnz,
                                                             rocComplex* mean,
                                                             rocComplex* secondMoment) {
    if (!distributed_host_fallback_enabled()) {
        return ROCQ_STATUS_NOT_IMPLEMENTED;
    }
    if (!handle || !handle->distributedMode || numQubits != handle->globalNumQubits ||
        handle->distributedDeviceIds.empty() || !mean || !secondMoment) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    int original_device = 0;
    if (hipGetDevice(&original_device) != hipSuccess) {
        return ROCQ_STATUS_HIP_ERROR;
    }
    auto restore_device = [&]() {
        (void)hipSetDevice(original_device);
    };

    if (hipSetDevice(handle->distributedDeviceIds[0]) != hipSuccess) {
        restore_device();
        return ROCQ_STATUS_HIP_ERROR;
    }

    std::vector<rocComplex> data;
    std::vector<size_t> indices;
    std::vector<size_t> indptr;
    rocqStatus_t status = copy_sparse_matrix_from_device(d_data,
                                                         d_indices,
                                                         d_indptr,
                                                         rows,
                                                         nnz,
                                                         handle->streams[0],
                                                         &data,
                                                         &indices,
                                                         &indptr);
    restore_device();
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }

    std::vector<rocComplex> host_full;
    status = gather_distributed_state_to_host(handle, &host_full);
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }

    return compute_sparse_matrix_moments_host_state(host_full,
                                                    numQubits,
                                                    data,
                                                    indices,
                                                    indptr,
                                                    rows,
                                                    cols,
                                                    nnz,
                                                    mean,
                                                    secondMoment);
}

rocqStatus_t apply_sparse_matrix_distributed_host_fallback(rocsvInternalHandle* handle,
                                                           unsigned numQubits,
                                                           const std::vector<unsigned>& targetQubits,
                                                           const rocComplex* d_data,
                                                           const size_t* d_indices,
                                                           const size_t* d_indptr,
                                                           size_t rows,
                                                           size_t cols,
                                                           size_t nnz) {
    if (!distributed_host_fallback_enabled()) {
        return ROCQ_STATUS_NOT_IMPLEMENTED;
    }
    if (!handle || !handle->distributedMode || numQubits != handle->globalNumQubits ||
        handle->distributedDeviceIds.empty()) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    int original_device = 0;
    if (hipGetDevice(&original_device) != hipSuccess) {
        return ROCQ_STATUS_HIP_ERROR;
    }
    auto restore_device = [&]() {
        (void)hipSetDevice(original_device);
    };

    if (hipSetDevice(handle->distributedDeviceIds[0]) != hipSuccess) {
        restore_device();
        return ROCQ_STATUS_HIP_ERROR;
    }

    std::vector<rocComplex> data;
    std::vector<size_t> indices;
    std::vector<size_t> indptr;
    rocqStatus_t status = copy_sparse_matrix_from_device(d_data,
                                                         d_indices,
                                                         d_indptr,
                                                         rows,
                                                         nnz,
                                                         handle->streams[0],
                                                         &data,
                                                         &indices,
                                                         &indptr);
    restore_device();
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }

    status = validate_sparse_matrix_host_csr(data, indices, indptr, rows, cols, nnz);
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }

    std::vector<rocComplex> host_full;
    status = gather_distributed_state_to_host(handle, &host_full);
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }

    status = apply_sparse_matrix_host_state_impl(&host_full,
                                                 numQubits,
                                                 targetQubits,
                                                 data,
                                                 indices,
                                                 indptr,
                                                 rows,
                                                 cols);
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }

    return scatter_host_state_to_distributed(handle, host_full);
}

__device__ inline double amp_abs2(const rocComplex& a) {
    return static_cast<double>(a.x) * static_cast<double>(a.x) +
           static_cast<double>(a.y) * static_cast<double>(a.y);
}

__device__ inline double amp_dot_real(const rocComplex& a, const rocComplex& b) {
    return static_cast<double>(a.x) * static_cast<double>(b.x) +
           static_cast<double>(a.y) * static_cast<double>(b.y);
}

__device__ inline double amp_cross_imag(const rocComplex& a, const rocComplex& b) {
    return static_cast<double>(a.x) * static_cast<double>(b.y) -
           static_cast<double>(a.y) * static_cast<double>(b.x);
}

__global__ void reduce_expectation_z_kernel(const rocComplex* state,
                                            size_t numElements,
                                            unsigned targetQubit,
                                            double* blockSums) {
    extern __shared__ double ssum[];
    const unsigned tid = threadIdx.x;
    const size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x + tid;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

    double local = 0.0;
    for (size_t idx = gid; idx < numElements; idx += stride) {
        const double mag = amp_abs2(state[idx]);
        local += (((idx >> targetQubit) & 1ULL) != 0ULL) ? -mag : mag;
    }

    ssum[tid] = local;
    __syncthreads();

    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            ssum[tid] += ssum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        blockSums[blockIdx.x] = ssum[0];
    }
}

__global__ void reduce_expectation_x_kernel(const rocComplex* state,
                                            size_t numElements,
                                            unsigned targetQubit,
                                            double* blockSums) {
    extern __shared__ double ssum[];
    const unsigned tid = threadIdx.x;
    const size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x + tid;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    const size_t targetMask = size_t{1} << targetQubit;

    double local = 0.0;
    for (size_t idx = gid; idx < numElements; idx += stride) {
        if ((idx & targetMask) != 0ULL) {
            continue;
        }
        const size_t partner = idx | targetMask;
        local += 2.0 * amp_dot_real(state[idx], state[partner]);
    }

    ssum[tid] = local;
    __syncthreads();

    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            ssum[tid] += ssum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        blockSums[blockIdx.x] = ssum[0];
    }
}

__global__ void reduce_expectation_y_kernel(const rocComplex* state,
                                            size_t numElements,
                                            unsigned targetQubit,
                                            double* blockSums) {
    extern __shared__ double ssum[];
    const unsigned tid = threadIdx.x;
    const size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x + tid;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    const size_t targetMask = size_t{1} << targetQubit;

    double local = 0.0;
    for (size_t idx = gid; idx < numElements; idx += stride) {
        if ((idx & targetMask) != 0ULL) {
            continue;
        }
        const size_t partner = idx | targetMask;
        local += 2.0 * amp_cross_imag(state[idx], state[partner]);
    }

    ssum[tid] = local;
    __syncthreads();

    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            ssum[tid] += ssum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        blockSums[blockIdx.x] = ssum[0];
    }
}

__global__ void reduce_expectation_z_product_kernel(const rocComplex* state,
                                                    size_t numElements,
                                                    const unsigned* targetQubits,
                                                    unsigned numTargetQubits,
                                                    double* blockSums) {
    extern __shared__ double ssum[];
    const unsigned tid = threadIdx.x;
    const size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x + tid;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

    double local = 0.0;
    for (size_t idx = gid; idx < numElements; idx += stride) {
        bool oddParity = false;
        for (unsigned i = 0; i < numTargetQubits; ++i) {
            oddParity ^= (((idx >> targetQubits[i]) & 1ULL) != 0ULL);
        }
        const double mag = amp_abs2(state[idx]);
        local += oddParity ? -mag : mag;
    }

    ssum[tid] = local;
    __syncthreads();

    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            ssum[tid] += ssum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        blockSums[blockIdx.x] = ssum[0];
    }
}

__global__ void reduce_expectation_pauli_string_kernel(const rocComplex* state,
                                                       size_t numElements,
                                                       const char* pauliString,
                                                       const unsigned* targetQubits,
                                                       unsigned numTargetQubits,
                                                       double* blockSums) {
    extern __shared__ double ssum[];
    const unsigned tid = threadIdx.x;
    const size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x + tid;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

    double local = 0.0;
    for (size_t idx = gid; idx < numElements; idx += stride) {
        size_t transformed = idx;
        double phaseRe = 1.0;
        double phaseIm = 0.0;

        for (unsigned k = 0; k < numTargetQubits; ++k) {
            const unsigned q = targetQubits[k];
            const bool bit = ((idx >> q) & 1ULL) != 0ULL;
            const char p = pauliString[k];
            const size_t mask = size_t{1} << q;
            if (p == 'X' || p == 'x') {
                transformed ^= mask;
            } else if (p == 'Y' || p == 'y') {
                transformed ^= mask;
                const double yRe = 0.0;
                const double yIm = bit ? -1.0 : 1.0;
                const double nextRe = phaseRe * yRe - phaseIm * yIm;
                const double nextIm = phaseRe * yIm + phaseIm * yRe;
                phaseRe = nextRe;
                phaseIm = nextIm;
            } else if (p == 'Z' || p == 'z') {
                if (bit) {
                    phaseRe = -phaseRe;
                    phaseIm = -phaseIm;
                }
            } else if (p == 'I' || p == 'i') {
                // no-op
            } else {
                phaseRe = 0.0;
                phaseIm = 0.0;
            }
        }

        const rocComplex a = state[idx];
        const rocComplex b = state[transformed];
        const double cbRe = static_cast<double>(a.x) * static_cast<double>(b.x) +
                            static_cast<double>(a.y) * static_cast<double>(b.y);
        const double cbIm = static_cast<double>(a.x) * static_cast<double>(b.y) -
                            static_cast<double>(a.y) * static_cast<double>(b.x);
        local += cbRe * phaseRe - cbIm * phaseIm;
    }

    ssum[tid] = local;
    __syncthreads();

    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            ssum[tid] += ssum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        blockSums[blockIdx.x] = ssum[0];
    }
}

__global__ void reduce_expectation_pauli_string_batch_kernel(const rocComplex* state,
                                                             size_t elementsPerState,
                                                             const char* pauliString,
                                                             const unsigned* targetQubits,
                                                             unsigned numTargetQubits,
                                                             double* blockSums) {
    extern __shared__ double ssum[];
    const unsigned tid = threadIdx.x;
    const size_t block = blockIdx.x;
    const size_t batch = blockIdx.y;
    const size_t gid = block * blockDim.x + tid;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    const rocComplex* batch_state = state + batch * elementsPerState;

    double local = 0.0;
    for (size_t idx = gid; idx < elementsPerState; idx += stride) {
        size_t transformed = idx;
        double phaseRe = 1.0;
        double phaseIm = 0.0;

        for (unsigned k = 0; k < numTargetQubits; ++k) {
            const unsigned q = targetQubits[k];
            const bool bit = ((idx >> q) & 1ULL) != 0ULL;
            const char p = pauliString[k];
            const size_t mask = size_t{1} << q;
            if (p == 'X' || p == 'x') {
                transformed ^= mask;
            } else if (p == 'Y' || p == 'y') {
                transformed ^= mask;
                const double yRe = 0.0;
                const double yIm = bit ? -1.0 : 1.0;
                const double nextRe = phaseRe * yRe - phaseIm * yIm;
                const double nextIm = phaseRe * yIm + phaseIm * yRe;
                phaseRe = nextRe;
                phaseIm = nextIm;
            } else if (p == 'Z' || p == 'z') {
                if (bit) {
                    phaseRe = -phaseRe;
                    phaseIm = -phaseIm;
                }
            } else if (p == 'I' || p == 'i') {
                // no-op
            } else {
                phaseRe = 0.0;
                phaseIm = 0.0;
            }
        }

        const rocComplex a = batch_state[idx];
        const rocComplex b = batch_state[transformed];
        const double cbRe = static_cast<double>(a.x) * static_cast<double>(b.x) +
                            static_cast<double>(a.y) * static_cast<double>(b.y);
        const double cbIm = static_cast<double>(a.x) * static_cast<double>(b.y) -
                            static_cast<double>(a.y) * static_cast<double>(b.x);
        local += cbRe * phaseRe - cbIm * phaseIm;
    }

    ssum[tid] = local;
    __syncthreads();

    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            ssum[tid] += ssum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        blockSums[batch * static_cast<size_t>(gridDim.x) + block] = ssum[0];
    }
}

__global__ void reduce_expectation_matrix_kernel(const rocComplex* state,
                                                 size_t numElements,
                                                 const unsigned* targetQubits,
                                                 unsigned numTargetQubits,
                                                 const rocComplex* matrix,
                                                 size_t matrixDim,
                                                 rocComplex* blockSums) {
    extern __shared__ double shared[];
    double* real_sums = shared;
    double* imag_sums = shared + blockDim.x;
    const unsigned tid = threadIdx.x;
    const size_t batch = static_cast<size_t>(blockIdx.y);
    const rocComplex* batch_state = state + batch * numElements;
    const size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x + tid;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

    double local_re = 0.0;
    double local_im = 0.0;
    for (size_t rowIndex = gid; rowIndex < numElements; rowIndex += stride) {
        size_t row_target = 0;
        size_t base_index = rowIndex;
        for (unsigned bit = 0; bit < numTargetQubits; ++bit) {
            const unsigned qubit = targetQubits[bit];
            const size_t mask = size_t{1} << qubit;
            if ((rowIndex & mask) != 0ULL) {
                row_target |= (size_t{1} << bit);
            }
            base_index &= ~mask;
        }

        const rocComplex bra = batch_state[rowIndex];
        for (size_t col_target = 0; col_target < matrixDim; ++col_target) {
            size_t col_index = base_index;
            for (unsigned bit = 0; bit < numTargetQubits; ++bit) {
                if ((col_target >> bit) & 1ULL) {
                    col_index |= (size_t{1} << targetQubits[bit]);
                }
            }

            const rocComplex m = matrix[row_target + col_target * matrixDim];
            const rocComplex ket = batch_state[col_index];
            const double mv_re = static_cast<double>(m.x) * static_cast<double>(ket.x) -
                                 static_cast<double>(m.y) * static_cast<double>(ket.y);
            const double mv_im = static_cast<double>(m.x) * static_cast<double>(ket.y) +
                                 static_cast<double>(m.y) * static_cast<double>(ket.x);

            local_re += static_cast<double>(bra.x) * mv_re + static_cast<double>(bra.y) * mv_im;
            local_im += static_cast<double>(bra.x) * mv_im - static_cast<double>(bra.y) * mv_re;
        }
    }

    real_sums[tid] = local_re;
    imag_sums[tid] = local_im;
    __syncthreads();

    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            real_sums[tid] += real_sums[tid + s];
            imag_sums[tid] += imag_sums[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        blockSums[batch * static_cast<size_t>(gridDim.x) + blockIdx.x] =
            make_complex(real_sums[0], imag_sums[0]);
    }
}

__global__ void reduce_sparse_matrix_moments_kernel(const rocComplex* state,
                                                    const rocComplex* data,
                                                    const size_t* indices,
                                                    const size_t* indptr,
                                                    size_t rows,
                                                    rocComplex* meanBlockSums,
                                                    rocComplex* secondBlockSums) {
    extern __shared__ double shared[];
    double* mean_re_sums = shared;
    double* mean_im_sums = shared + blockDim.x;
    double* second_re_sums = shared + 2 * blockDim.x;
    double* second_im_sums = shared + 3 * blockDim.x;
    const unsigned tid = threadIdx.x;
    const size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x + tid;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

    double mean_re = 0.0;
    double mean_im = 0.0;
    double second_re = 0.0;
    double second_im = 0.0;
    for (size_t row = gid; row < rows; row += stride) {
        double accum_re = 0.0;
        double accum_im = 0.0;
        for (size_t offset = indptr[row]; offset < indptr[row + 1]; ++offset) {
            const rocComplex value = data[offset];
            const rocComplex ket = state[indices[offset]];
            accum_re += static_cast<double>(value.x) * static_cast<double>(ket.x) -
                        static_cast<double>(value.y) * static_cast<double>(ket.y);
            accum_im += static_cast<double>(value.x) * static_cast<double>(ket.y) +
                        static_cast<double>(value.y) * static_cast<double>(ket.x);
        }

        const rocComplex bra = state[row];
        mean_re += static_cast<double>(bra.x) * accum_re + static_cast<double>(bra.y) * accum_im;
        mean_im += static_cast<double>(bra.x) * accum_im - static_cast<double>(bra.y) * accum_re;
        second_re += accum_re * accum_re + accum_im * accum_im;
    }

    mean_re_sums[tid] = mean_re;
    mean_im_sums[tid] = mean_im;
    second_re_sums[tid] = second_re;
    second_im_sums[tid] = second_im;
    __syncthreads();

    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            mean_re_sums[tid] += mean_re_sums[tid + s];
            mean_im_sums[tid] += mean_im_sums[tid + s];
            second_re_sums[tid] += second_re_sums[tid + s];
            second_im_sums[tid] += second_im_sums[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        meanBlockSums[blockIdx.x] = make_complex(mean_re_sums[0], mean_im_sums[0]);
        secondBlockSums[blockIdx.x] = make_complex(second_re_sums[0], second_im_sums[0]);
    }
}

__global__ void reduce_sparse_matrix_moments_batch_kernel(const rocComplex* state,
                                                          const rocComplex* data,
                                                          const size_t* indices,
                                                          const size_t* indptr,
                                                          size_t rows,
                                                          rocComplex* meanBlockSums,
                                                          rocComplex* secondBlockSums) {
    extern __shared__ double shared[];
    double* mean_re_sums = shared;
    double* mean_im_sums = shared + blockDim.x;
    double* second_re_sums = shared + 2 * blockDim.x;
    double* second_im_sums = shared + 3 * blockDim.x;
    const unsigned tid = threadIdx.x;
    const size_t batch = static_cast<size_t>(blockIdx.y);
    const size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x + tid;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    const rocComplex* batch_state = state + batch * rows;

    double mean_re = 0.0;
    double mean_im = 0.0;
    double second_re = 0.0;
    double second_im = 0.0;
    for (size_t row = gid; row < rows; row += stride) {
        double accum_re = 0.0;
        double accum_im = 0.0;
        for (size_t offset = indptr[row]; offset < indptr[row + 1]; ++offset) {
            const rocComplex value = data[offset];
            const rocComplex ket = batch_state[indices[offset]];
            accum_re += static_cast<double>(value.x) * static_cast<double>(ket.x) -
                        static_cast<double>(value.y) * static_cast<double>(ket.y);
            accum_im += static_cast<double>(value.x) * static_cast<double>(ket.y) +
                        static_cast<double>(value.y) * static_cast<double>(ket.x);
        }

        const rocComplex bra = batch_state[row];
        mean_re += static_cast<double>(bra.x) * accum_re + static_cast<double>(bra.y) * accum_im;
        mean_im += static_cast<double>(bra.x) * accum_im - static_cast<double>(bra.y) * accum_re;
        second_re += accum_re * accum_re + accum_im * accum_im;
    }

    mean_re_sums[tid] = mean_re;
    mean_im_sums[tid] = mean_im;
    second_re_sums[tid] = second_re;
    second_im_sums[tid] = second_im;
    __syncthreads();

    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            mean_re_sums[tid] += mean_re_sums[tid + s];
            mean_im_sums[tid] += mean_im_sums[tid + s];
            second_re_sums[tid] += second_re_sums[tid + s];
            second_im_sums[tid] += second_im_sums[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        const size_t offset = batch * static_cast<size_t>(gridDim.x) + blockIdx.x;
        meanBlockSums[offset] = make_complex(mean_re_sums[0], mean_im_sums[0]);
        secondBlockSums[offset] = make_complex(second_re_sums[0], second_im_sums[0]);
    }
}

__global__ void apply_sparse_matrix_kernel(const rocComplex* inputState,
                                           rocComplex* outputState,
                                           size_t numElements,
                                           const unsigned* targetQubits,
                                           unsigned numTargetQubits,
                                           const rocComplex* data,
                                           const size_t* indices,
                                           const size_t* indptr) {
    const size_t batch = static_cast<size_t>(blockIdx.y);
    const size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    const rocComplex* batch_input = inputState + batch * numElements;
    rocComplex* batch_output = outputState + batch * numElements;

    for (size_t row_index = gid; row_index < numElements; row_index += stride) {
        size_t local_row = 0;
        size_t base_index = row_index;
        for (unsigned bit = 0; bit < numTargetQubits; ++bit) {
            const unsigned qubit = targetQubits[bit];
            const size_t mask = size_t{1} << qubit;
            if ((row_index & mask) != 0ULL) {
                local_row |= size_t{1} << bit;
            }
            base_index &= ~mask;
        }

        double accum_re = 0.0;
        double accum_im = 0.0;
        for (size_t offset = indptr[local_row]; offset < indptr[local_row + 1]; ++offset) {
            size_t col_index = base_index;
            const size_t local_col = indices[offset];
            for (unsigned bit = 0; bit < numTargetQubits; ++bit) {
                if (((local_col >> bit) & size_t{1}) != 0ULL) {
                    col_index |= size_t{1} << targetQubits[bit];
                }
            }

            const rocComplex value = data[offset];
            const rocComplex ket = batch_input[col_index];
            accum_re += static_cast<double>(value.x) * static_cast<double>(ket.x) -
                        static_cast<double>(value.y) * static_cast<double>(ket.y);
            accum_im += static_cast<double>(value.x) * static_cast<double>(ket.y) +
                        static_cast<double>(value.y) * static_cast<double>(ket.x);
        }
        batch_output[row_index] = make_complex(accum_re, accum_im);
    }
}

__global__ void accumulate_sample_probabilities_kernel(const rocComplex* state,
                                                       size_t numElements,
                                                       const unsigned* measuredQubits,
                                                       unsigned numMeasuredQubits,
                                                       double* outcomeProbs) {
    const size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

    for (size_t idx = gid; idx < numElements; idx += stride) {
        const double prob = amp_abs2(state[idx]);
        if (prob <= 0.0) {
            continue;
        }

        unsigned outcome = 0;
        for (unsigned b = 0; b < numMeasuredQubits; ++b) {
            if ((idx >> measuredQubits[b]) & 1ULL) {
                outcome |= (1U << b);
            }
        }
        atomicAdd(&outcomeProbs[outcome], prob);
    }
}

__global__ void accumulate_sample_probabilities_batch_kernel(const rocComplex* state,
                                                             size_t elementsPerState,
                                                             size_t batchSize,
                                                             const unsigned* measuredQubits,
                                                             unsigned numMeasuredQubits,
                                                             size_t numOutcomes,
                                                             double* outcomeProbs) {
    const size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    const size_t total_elements = elementsPerState * batchSize;

    for (size_t idx = gid; idx < total_elements; idx += stride) {
        const size_t batch = idx / elementsPerState;
        const size_t basis = idx - batch * elementsPerState;
        const double prob = amp_abs2(state[idx]);
        if (prob <= 0.0) {
            continue;
        }

        unsigned outcome = 0;
        for (unsigned b = 0; b < numMeasuredQubits; ++b) {
            if ((basis >> measuredQubits[b]) & 1ULL) {
                outcome |= (1U << b);
            }
        }
        atomicAdd(&outcomeProbs[batch * numOutcomes + outcome], prob);
    }
}

__global__ void reduce_double_sum_kernel(const double* input,
                                         size_t numElements,
                                         double* output) {
    extern __shared__ double ssum[];
    const unsigned tid = threadIdx.x;
    const size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x + tid;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

    double local = 0.0;
    for (size_t i = gid; i < numElements; i += stride) {
        local += input[i];
    }

    ssum[tid] = local;
    __syncthreads();

    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            ssum[tid] += ssum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = ssum[0];
    }
}

__global__ void build_sampling_cdf_kernel(double* outcomeProbs,
                                          size_t numOutcomes) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || !outcomeProbs || numOutcomes == 0) {
        return;
    }

    double total = 0.0;
    for (size_t i = 0; i < numOutcomes; ++i) {
        total += outcomeProbs[i];
    }

    if (total <= 0.0) {
        outcomeProbs[numOutcomes - 1] = 0.0;
        return;
    }

    double running = 0.0;
    const double invTotal = 1.0 / total;
    for (size_t i = 0; i < numOutcomes; ++i) {
        running += outcomeProbs[i] * invTotal;
        outcomeProbs[i] = running;
    }
    outcomeProbs[numOutcomes - 1] = 1.0;
}

__global__ void sample_from_cdf_kernel(const double* cdf,
                                       size_t numOutcomes,
                                       uint64_t* results,
                                       unsigned numShots,
                                       unsigned long long seed) {
    const size_t shot = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (shot >= numShots || !cdf || !results || numOutcomes == 0) {
        return;
    }

    hiprandStatePhilox4_32_10_t rngState;
    hiprand_init(seed, static_cast<unsigned long long>(shot), 0ULL, &rngState);
    double r = hiprand_uniform_double(&rngState);
    if (r >= 1.0) {
        r = 0.9999999999999999;
    }

    size_t low = 0;
    size_t high = numOutcomes - 1;
    while (low < high) {
        const size_t mid = low + ((high - low) >> 1);
        if (r <= cdf[mid]) {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    results[shot] = static_cast<uint64_t>(low);
}

__global__ void reduce_measure_prob0_kernel(const rocComplex* state,
                                            size_t numElements,
                                            unsigned targetQubit,
                                            double* blockSums) {
    extern __shared__ double ssum[];
    const unsigned tid = threadIdx.x;
    const size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x + tid;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

    double local = 0.0;
    for (size_t idx = gid; idx < numElements; idx += stride) {
        if (((idx >> targetQubit) & 1ULL) == 0ULL) {
            local += amp_abs2(state[idx]);
        }
    }

    ssum[tid] = local;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            ssum[tid] += ssum[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        blockSums[blockIdx.x] = ssum[0];
    }
}

__global__ void collapse_and_renorm_measure_kernel(rocComplex* state,
                                                   size_t numElements,
                                                   unsigned targetQubit,
                                                   int measuredOutcome,
                                                   double invNorm) {
    const size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

    for (size_t idx = gid; idx < numElements; idx += stride) {
        const int bit = ((idx >> targetQubit) & 1ULL) ? 1 : 0;
        if (bit != measuredOutcome) {
            state[idx] = make_complex(0.0, 0.0);
            continue;
        }
        state[idx].x = static_cast<real_t>(static_cast<double>(state[idx].x) * invNorm);
        state[idx].y = static_cast<real_t>(static_cast<double>(state[idx].y) * invNorm);
    }
}

inline int compute_reduction_blocks(size_t elements, int threadsPerBlock) {
    if (elements == 0 || threadsPerBlock <= 0) {
        return 1;
    }
    const size_t raw = (elements + static_cast<size_t>(threadsPerBlock) - 1) / static_cast<size_t>(threadsPerBlock);
    const size_t capped = std::min<size_t>(raw, 65535);
    return static_cast<int>(std::max<size_t>(1, capped));
}

inline rocqStatus_t reduce_blocks_to_scalar(rocsvInternalHandle* handle,
                                            double* d_blockSums,
                                            int numBlocks,
                                            double* outResult) {
    if (!handle || !d_blockSums || !outResult || numBlocks <= 0) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    constexpr int threads_per_block = 256;
    double* current_in = d_blockSums;
    size_t current_count = static_cast<size_t>(numBlocks);
    bool owns_current_in = false;

    while (current_count > 1) {
        const int next_blocks = compute_reduction_blocks(current_count, threads_per_block);
        void* next_out_void = nullptr;
        rocqStatus_t alloc_status = device_malloc(handle,
                                                  &next_out_void,
                                                  static_cast<size_t>(next_blocks) * sizeof(double));
        if (alloc_status != ROCQ_STATUS_SUCCESS) {
            if (owns_current_in) {
                device_free(handle, current_in);
            }
            return alloc_status;
        }

        double* next_out = static_cast<double*>(next_out_void);
        hipLaunchKernelGGL(reduce_double_sum_kernel,
                           dim3(next_blocks),
                           dim3(threads_per_block),
                           threads_per_block * sizeof(double),
                           handle->streams[0],
                           current_in,
                           current_count,
                           next_out);
        rocqStatus_t launch_status = check_last_hip_error();
        if (launch_status != ROCQ_STATUS_SUCCESS) {
            device_free(handle, next_out);
            if (owns_current_in) {
                device_free(handle, current_in);
            }
            return launch_status;
        }

        if (owns_current_in) {
            device_free(handle, current_in);
        }

        current_in = next_out;
        current_count = static_cast<size_t>(next_blocks);
        owns_current_in = true;
    }

    rocqStatus_t copy_status = copy_device_to_host(outResult,
                                                   current_in,
                                                   sizeof(double),
                                                   handle->streams[0]);
    if (owns_current_in) {
        device_free(handle, current_in);
    }
    return copy_status;
}

inline rocqStatus_t reduce_blocks_to_device_scalar(rocsvInternalHandle* handle,
                                                   hipStream_t stream,
                                                   double* d_blockSums,
                                                   int numBlocks,
                                                   double** d_scalar_out) {
    if (!handle || !d_blockSums || !d_scalar_out || numBlocks <= 0) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    void* scalar_void = nullptr;
    rocqStatus_t status = device_malloc(handle, &scalar_void, sizeof(double));
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }
    double* d_scalar = static_cast<double*>(scalar_void);

    constexpr int threads_per_block = 256;
    double* current_in = d_blockSums;
    size_t current_count = static_cast<size_t>(numBlocks);
    bool owns_current_in = false;

    while (current_count > 1) {
        const int next_blocks = compute_reduction_blocks(current_count, threads_per_block);
        void* next_out_void = nullptr;
        status = device_malloc(handle,
                               &next_out_void,
                               static_cast<size_t>(next_blocks) * sizeof(double));
        if (status != ROCQ_STATUS_SUCCESS) {
            if (owns_current_in) {
                device_free(handle, current_in);
            }
            device_free(handle, d_scalar);
            return status;
        }

        double* next_out = static_cast<double*>(next_out_void);
        hipLaunchKernelGGL(reduce_double_sum_kernel,
                           dim3(next_blocks),
                           dim3(threads_per_block),
                           threads_per_block * sizeof(double),
                           stream,
                           current_in,
                           current_count,
                           next_out);
        status = check_last_hip_error();
        if (status != ROCQ_STATUS_SUCCESS) {
            device_free(handle, next_out);
            if (owns_current_in) {
                device_free(handle, current_in);
            }
            device_free(handle, d_scalar);
            return status;
        }

        if (owns_current_in) {
            device_free(handle, current_in);
        }
        current_in = next_out;
        current_count = static_cast<size_t>(next_blocks);
        owns_current_in = true;
    }

    if (hipMemcpyAsync(d_scalar,
                       current_in,
                       sizeof(double),
                       hipMemcpyDeviceToDevice,
                       stream) != hipSuccess) {
        if (owns_current_in) {
            device_free(handle, current_in);
        }
        device_free(handle, d_scalar);
        return ROCQ_STATUS_HIP_ERROR;
    }
    if (owns_current_in) {
        device_free(handle, current_in);
    }

    *d_scalar_out = d_scalar;
    return ROCQ_STATUS_SUCCESS;
}

inline rocqStatus_t rccl_allreduce_double_sum_inplace(rocsvInternalHandle* handle,
                                                      const std::vector<double*>& rankBuffers,
                                                      size_t count) {
    if (!handle || rankBuffers.size() != static_cast<size_t>(handle->distributedGpuCount) || count == 0) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (!distributed_rccl_ready(handle)) {
        return ROCQ_STATUS_NOT_IMPLEMENTED;
    }

#ifdef ROCQ_HAVE_RCCL
    ncclResult_t status = ncclGroupStart();
    if (status != ncclSuccess) {
        return ROCQ_STATUS_RCCL_ERROR;
    }

    for (int rank = 0; rank < handle->distributedGpuCount; ++rank) {
        const size_t rank_idx = static_cast<size_t>(rank);
        if (hipSetDevice(handle->distributedDeviceIds[rank_idx]) != hipSuccess) {
            (void)ncclGroupEnd();
            return ROCQ_STATUS_HIP_ERROR;
        }
        status = ncclAllReduce(rankBuffers[rank_idx],
                               rankBuffers[rank_idx],
                               count,
                               ncclDouble,
                               ncclSum,
                               handle->distributedComms[rank_idx],
                               handle->distributedStreams[rank_idx]);
        if (status != ncclSuccess) {
            (void)ncclGroupEnd();
            return ROCQ_STATUS_RCCL_ERROR;
        }
    }

    status = ncclGroupEnd();
    return (status == ncclSuccess) ? ROCQ_STATUS_SUCCESS : ROCQ_STATUS_RCCL_ERROR;
#else
    (void)handle;
    (void)rankBuffers;
    (void)count;
    return ROCQ_STATUS_NOT_IMPLEMENTED;
#endif
}

enum class DistributedExpectationKind {
    SingleZ,
    SingleX,
    SingleY,
    ZProduct,
    PauliString,
};

inline rocqStatus_t distributed_expectation_rccl(rocsvInternalHandle* handle,
                                                unsigned numQubits,
                                                const std::vector<unsigned>& targets,
                                                DistributedExpectationKind kind,
                                                const std::string& pauli,
                                                double* result) {
    if (!handle || !result || targets.empty()) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (!distributed_rccl_ready(handle)) {
        return ROCQ_STATUS_NOT_IMPLEMENTED;
    }
    if (numQubits != handle->globalNumQubits) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (!distributed_all_qubits_local(handle, targets)) {
        return ROCQ_STATUS_NOT_IMPLEMENTED;
    }
    if (kind == DistributedExpectationKind::PauliString && pauli.size() != targets.size()) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    const int ranks = handle->distributedGpuCount;
    const size_t rank_count = static_cast<size_t>(ranks);
    const size_t local_elements = handle->localSliceElements;
    constexpr int threads_per_block = 256;
    const int blocks = compute_reduction_blocks(local_elements, threads_per_block);

    std::vector<double*> rank_block_sums(rank_count, nullptr);
    std::vector<double*> rank_scalars(rank_count, nullptr);
    std::vector<unsigned*> rank_targets(rank_count, nullptr);
    std::vector<char*> rank_pauli(rank_count, nullptr);

    auto cleanup = [&]() {
        for (int rank = 0; rank < ranks; ++rank) {
            const size_t idx = static_cast<size_t>(rank);
            if (hipSetDevice(handle->distributedDeviceIds[idx]) != hipSuccess) {
                continue;
            }
            if (rank_pauli[idx]) {
                (void)device_free(handle, rank_pauli[idx]);
            }
            if (rank_targets[idx]) {
                (void)device_free(handle, rank_targets[idx]);
            }
            if (rank_scalars[idx]) {
                (void)device_free(handle, rank_scalars[idx]);
            }
            if (rank_block_sums[idx]) {
                (void)device_free(handle, rank_block_sums[idx]);
            }
        }
    };

    for (int rank = 0; rank < ranks; ++rank) {
        const size_t idx = static_cast<size_t>(rank);
        if (hipSetDevice(handle->distributedDeviceIds[idx]) != hipSuccess) {
            cleanup();
            return ROCQ_STATUS_HIP_ERROR;
        }
        hipStream_t stream = handle->distributedStreams[idx];

        void* block_sums_void = nullptr;
        rocqStatus_t status = device_malloc(handle,
                                            &block_sums_void,
                                            static_cast<size_t>(blocks) * sizeof(double));
        if (status != ROCQ_STATUS_SUCCESS) {
            cleanup();
            return status;
        }
        rank_block_sums[idx] = static_cast<double*>(block_sums_void);

        if (kind == DistributedExpectationKind::ZProduct ||
            kind == DistributedExpectationKind::PauliString) {
            void* targets_void = nullptr;
            status = device_malloc(handle, &targets_void, targets.size() * sizeof(unsigned));
            if (status != ROCQ_STATUS_SUCCESS) {
                cleanup();
                return status;
            }
            rank_targets[idx] = static_cast<unsigned*>(targets_void);
            status = copy_host_to_device(rank_targets[idx],
                                         targets.data(),
                                         targets.size() * sizeof(unsigned),
                                         stream);
            if (status != ROCQ_STATUS_SUCCESS) {
                cleanup();
                return status;
            }
        }

        if (kind == DistributedExpectationKind::PauliString) {
            void* pauli_void = nullptr;
            status = device_malloc(handle, &pauli_void, pauli.size() * sizeof(char));
            if (status != ROCQ_STATUS_SUCCESS) {
                cleanup();
                return status;
            }
            rank_pauli[idx] = static_cast<char*>(pauli_void);
            status = copy_host_to_device(rank_pauli[idx],
                                         pauli.data(),
                                         pauli.size() * sizeof(char),
                                         stream);
            if (status != ROCQ_STATUS_SUCCESS) {
                cleanup();
                return status;
            }
        }

        switch (kind) {
            case DistributedExpectationKind::SingleZ:
                hipLaunchKernelGGL(reduce_expectation_z_kernel,
                                   dim3(blocks),
                                   dim3(threads_per_block),
                                   threads_per_block * sizeof(double),
                                   stream,
                                   handle->distributedSlices[idx],
                                   local_elements,
                                   targets[0],
                                   rank_block_sums[idx]);
                break;
            case DistributedExpectationKind::SingleX:
                hipLaunchKernelGGL(reduce_expectation_x_kernel,
                                   dim3(blocks),
                                   dim3(threads_per_block),
                                   threads_per_block * sizeof(double),
                                   stream,
                                   handle->distributedSlices[idx],
                                   local_elements,
                                   targets[0],
                                   rank_block_sums[idx]);
                break;
            case DistributedExpectationKind::SingleY:
                hipLaunchKernelGGL(reduce_expectation_y_kernel,
                                   dim3(blocks),
                                   dim3(threads_per_block),
                                   threads_per_block * sizeof(double),
                                   stream,
                                   handle->distributedSlices[idx],
                                   local_elements,
                                   targets[0],
                                   rank_block_sums[idx]);
                break;
            case DistributedExpectationKind::ZProduct:
                hipLaunchKernelGGL(reduce_expectation_z_product_kernel,
                                   dim3(blocks),
                                   dim3(threads_per_block),
                                   threads_per_block * sizeof(double),
                                   stream,
                                   handle->distributedSlices[idx],
                                   local_elements,
                                   rank_targets[idx],
                                   static_cast<unsigned>(targets.size()),
                                   rank_block_sums[idx]);
                break;
            case DistributedExpectationKind::PauliString:
                hipLaunchKernelGGL(reduce_expectation_pauli_string_kernel,
                                   dim3(blocks),
                                   dim3(threads_per_block),
                                   threads_per_block * sizeof(double),
                                   stream,
                                   handle->distributedSlices[idx],
                                   local_elements,
                                   rank_pauli[idx],
                                   rank_targets[idx],
                                   static_cast<unsigned>(targets.size()),
                                   rank_block_sums[idx]);
                break;
        }

        status = check_last_hip_error();
        if (status != ROCQ_STATUS_SUCCESS) {
            cleanup();
            return status;
        }

        status = reduce_blocks_to_device_scalar(handle,
                                                stream,
                                                rank_block_sums[idx],
                                                blocks,
                                                &rank_scalars[idx]);
        if (status != ROCQ_STATUS_SUCCESS) {
            cleanup();
            return status;
        }
    }

    rocqStatus_t status = rccl_allreduce_double_sum_inplace(handle, rank_scalars, 1);
    if (status != ROCQ_STATUS_SUCCESS) {
        cleanup();
        return status;
    }

    status = sync_distributed_streams(handle);
    if (status != ROCQ_STATUS_SUCCESS) {
        cleanup();
        return status;
    }

    if (hipSetDevice(handle->distributedDeviceIds[0]) != hipSuccess) {
        cleanup();
        return ROCQ_STATUS_HIP_ERROR;
    }
    status = copy_device_to_host(result,
                                 rank_scalars[0],
                                 sizeof(double),
                                 handle->distributedStreams[0]);
    cleanup();
    return status;
}

inline rocqStatus_t host_pauli_expectation(const std::vector<rocComplex>& host_state,
                                           unsigned numQubits,
                                           const std::vector<unsigned>& targets,
                                           const std::string& pauli,
                                           double* result) {
    if (!result || targets.empty() || targets.size() != pauli.size()) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    size_t state_elements = 0;
    if (!compute_power_of_two(numQubits, &state_elements) || host_state.size() != state_elements) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    std::complex<double> total{0.0, 0.0};
    for (size_t idx = 0; idx < state_elements; ++idx) {
        size_t out_idx = idx;
        std::complex<double> phase{1.0, 0.0};

        for (size_t p = 0; p < pauli.size(); ++p) {
            const unsigned qubit = targets[p];
            if (!validate_qubit_index(qubit, numQubits)) {
                return ROCQ_STATUS_INVALID_VALUE;
            }
            const bool bit = ((idx >> qubit) & 1ULL) != 0ULL;
            const char op = static_cast<char>(std::toupper(static_cast<unsigned char>(pauli[p])));
            switch (op) {
                case 'I':
                    break;
                case 'X':
                    out_idx ^= (size_t{1} << qubit);
                    break;
                case 'Y':
                    out_idx ^= (size_t{1} << qubit);
                    phase *= bit ? std::complex<double>{0.0, -1.0} : std::complex<double>{0.0, 1.0};
                    break;
                case 'Z':
                    if (bit) {
                        phase = -phase;
                    }
                    break;
                default:
                    return ROCQ_STATUS_INVALID_VALUE;
            }
        }

        total += std::conj(to_std_complex(host_state[out_idx])) * phase * to_std_complex(host_state[idx]);
    }

    *result = total.real();
    return ROCQ_STATUS_SUCCESS;
}

inline rocqStatus_t distributed_expectation_host_fallback(rocsvInternalHandle* handle,
                                                          unsigned numQubits,
                                                          const std::vector<unsigned>& targets,
                                                          const std::string& pauli,
                                                          double* result) {
    if (!distributed_host_fallback_enabled()) {
        return ROCQ_STATUS_NOT_IMPLEMENTED;
    }
    if (!handle || !handle->distributedMode || numQubits != handle->globalNumQubits) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    std::vector<rocComplex> host_full;
    rocqStatus_t status = gather_distributed_state_to_host(handle, &host_full);
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }
    return host_pauli_expectation(host_full, numQubits, targets, pauli, result);
}

inline rocqStatus_t distributed_expectation_with_fallback(rocsvInternalHandle* handle,
                                                          unsigned numQubits,
                                                          const std::vector<unsigned>& targets,
                                                          DistributedExpectationKind kind,
                                                          const std::string& pauli,
                                                          double* result) {
    rocqStatus_t status = distributed_expectation_rccl(handle, numQubits, targets, kind, pauli, result);
    if (status != ROCQ_STATUS_NOT_IMPLEMENTED) {
        return status;
    }

    std::string fallback_pauli = pauli;
    if (fallback_pauli.empty()) {
        switch (kind) {
            case DistributedExpectationKind::SingleZ:
                fallback_pauli = "Z";
                break;
            case DistributedExpectationKind::SingleX:
                fallback_pauli = "X";
                break;
            case DistributedExpectationKind::SingleY:
                fallback_pauli = "Y";
                break;
            case DistributedExpectationKind::ZProduct:
                fallback_pauli.assign(targets.size(), 'Z');
                break;
            case DistributedExpectationKind::PauliString:
                break;
        }
    }

    return distributed_expectation_host_fallback(handle, numQubits, targets, fallback_pauli, result);
}

inline rocqStatus_t free_distributed_probability_buffers(rocsvInternalHandle* handle,
                                                         const std::vector<double*>& rank_probs) {
    if (!handle) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    rocqStatus_t first_status = ROCQ_STATUS_SUCCESS;
    const int ranks = handle->distributedGpuCount;
    for (int rank = 0; rank < ranks && static_cast<size_t>(rank) < rank_probs.size(); ++rank) {
        const size_t idx = static_cast<size_t>(rank);
        if (!rank_probs[idx]) {
            continue;
        }
        if (hipSetDevice(handle->distributedDeviceIds[idx]) != hipSuccess) {
            if (first_status == ROCQ_STATUS_SUCCESS) {
                first_status = ROCQ_STATUS_HIP_ERROR;
            }
            continue;
        }
        rocqStatus_t status = device_free(handle, rank_probs[idx]);
        if (first_status == ROCQ_STATUS_SUCCESS && status != ROCQ_STATUS_SUCCESS) {
            first_status = status;
        }
    }
    return first_status;
}

inline rocqStatus_t accumulate_distributed_sample_probabilities_rccl(
    rocsvInternalHandle* handle,
    unsigned numQubits,
    const std::vector<unsigned>& measured,
    std::vector<double*>* rank_probs_out) {
    if (!handle || measured.empty() || !rank_probs_out) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    rank_probs_out->clear();
    if (!distributed_rccl_ready(handle)) {
        return ROCQ_STATUS_NOT_IMPLEMENTED;
    }
    if (numQubits != handle->globalNumQubits) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (!distributed_all_qubits_local(handle, measured)) {
        return ROCQ_STATUS_NOT_IMPLEMENTED;
    }
    if (measured.size() >= static_cast<size_t>(sizeof(size_t) * 8)) {
        return ROCQ_STATUS_NOT_IMPLEMENTED;
    }

    const size_t num_outcomes = size_t{1} << measured.size();
    const int ranks = handle->distributedGpuCount;
    const size_t rank_count = static_cast<size_t>(ranks);
    const size_t local_elements = handle->localSliceElements;
    constexpr int threads_per_block = 256;
    const int blocks = compute_reduction_blocks(local_elements, threads_per_block);

    std::vector<unsigned*> rank_measured(rank_count, nullptr);
    rank_probs_out->assign(rank_count, nullptr);

    auto cleanup = [&]() {
        for (int rank = 0; rank < ranks; ++rank) {
            const size_t idx = static_cast<size_t>(rank);
            if (hipSetDevice(handle->distributedDeviceIds[idx]) != hipSuccess) {
                continue;
            }
            if ((*rank_probs_out)[idx]) {
                (void)device_free(handle, (*rank_probs_out)[idx]);
                (*rank_probs_out)[idx] = nullptr;
            }
            if (rank_measured[idx]) {
                (void)device_free(handle, rank_measured[idx]);
                rank_measured[idx] = nullptr;
            }
        }
    };

    for (int rank = 0; rank < ranks; ++rank) {
        const size_t idx = static_cast<size_t>(rank);
        if (hipSetDevice(handle->distributedDeviceIds[idx]) != hipSuccess) {
            cleanup();
            return ROCQ_STATUS_HIP_ERROR;
        }
        hipStream_t stream = handle->distributedStreams[idx];

        void* measured_void = nullptr;
        rocqStatus_t status = device_malloc(handle, &measured_void, measured.size() * sizeof(unsigned));
        if (status != ROCQ_STATUS_SUCCESS) {
            cleanup();
            return status;
        }
        rank_measured[idx] = static_cast<unsigned*>(measured_void);
        status = copy_host_to_device(rank_measured[idx],
                                     measured.data(),
                                     measured.size() * sizeof(unsigned),
                                     stream);
        if (status != ROCQ_STATUS_SUCCESS) {
            cleanup();
            return status;
        }

        void* probs_void = nullptr;
        status = device_malloc(handle, &probs_void, num_outcomes * sizeof(double));
        if (status != ROCQ_STATUS_SUCCESS) {
            cleanup();
            return status;
        }
        (*rank_probs_out)[idx] = static_cast<double*>(probs_void);
        if (hipMemsetAsync((*rank_probs_out)[idx], 0, num_outcomes * sizeof(double), stream) != hipSuccess) {
            cleanup();
            return ROCQ_STATUS_HIP_ERROR;
        }

        hipLaunchKernelGGL(accumulate_sample_probabilities_kernel,
                           dim3(blocks),
                           dim3(threads_per_block),
                           0,
                           stream,
                           handle->distributedSlices[idx],
                           local_elements,
                           rank_measured[idx],
                           static_cast<unsigned>(measured.size()),
                           (*rank_probs_out)[idx]);
        status = check_last_hip_error();
        if (status != ROCQ_STATUS_SUCCESS) {
            cleanup();
            return status;
        }
    }

    rocqStatus_t status = rccl_allreduce_double_sum_inplace(handle, *rank_probs_out, num_outcomes);
    if (status != ROCQ_STATUS_SUCCESS) {
        cleanup();
        return status;
    }

    status = sync_distributed_streams(handle);
    if (status != ROCQ_STATUS_SUCCESS) {
        cleanup();
        return status;
    }

    rocqStatus_t first_free_status = ROCQ_STATUS_SUCCESS;
    for (int rank = 0; rank < ranks; ++rank) {
        const size_t idx = static_cast<size_t>(rank);
        if (hipSetDevice(handle->distributedDeviceIds[idx]) != hipSuccess) {
            if (first_free_status == ROCQ_STATUS_SUCCESS) {
                first_free_status = ROCQ_STATUS_HIP_ERROR;
            }
            continue;
        }
        if (rank_measured[idx]) {
            rocqStatus_t free_status = device_free(handle, rank_measured[idx]);
            rank_measured[idx] = nullptr;
            if (first_free_status == ROCQ_STATUS_SUCCESS && free_status != ROCQ_STATUS_SUCCESS) {
                first_free_status = free_status;
            }
        }
    }
    if (first_free_status != ROCQ_STATUS_SUCCESS) {
        (void)free_distributed_probability_buffers(handle, *rank_probs_out);
        rank_probs_out->assign(rank_count, nullptr);
        return first_free_status;
    }

    return ROCQ_STATUS_SUCCESS;
}

inline rocqStatus_t distributed_sample_rccl(rocsvInternalHandle* handle,
                                            unsigned numQubits,
                                            const std::vector<unsigned>& measured,
                                            unsigned numShots,
                                            uint64_t* h_results) {
    if (!handle || measured.empty() || !h_results) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (numShots == 0) {
        return ROCQ_STATUS_SUCCESS;
    }

    std::vector<double*> rank_probs;
    rocqStatus_t status = accumulate_distributed_sample_probabilities_rccl(
        handle, numQubits, measured, &rank_probs);
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }

    const size_t num_outcomes = size_t{1} << measured.size();
    uint64_t* d_results = nullptr;

    auto cleanup = [&]() {
        (void)free_distributed_probability_buffers(handle, rank_probs);
        if (d_results) {
            if (hipSetDevice(handle->distributedDeviceIds[0]) == hipSuccess) {
                (void)device_free(handle, d_results);
            }
        }
    };

    if (hipSetDevice(handle->distributedDeviceIds[0]) != hipSuccess) {
        cleanup();
        return ROCQ_STATUS_HIP_ERROR;
    }
    hipStream_t stream0 = handle->distributedStreams[0];
    hipLaunchKernelGGL(build_sampling_cdf_kernel,
                       dim3(1),
                       dim3(1),
                       0,
                       stream0,
                       rank_probs[0],
                       num_outcomes);
    status = check_last_hip_error();
    if (status != ROCQ_STATUS_SUCCESS) {
        cleanup();
        return status;
    }

    double cdf_last = 0.0;
    if (hipMemcpyAsync(&cdf_last,
                       rank_probs[0] + (num_outcomes - 1),
                       sizeof(double),
                       hipMemcpyDeviceToHost,
                       stream0) != hipSuccess) {
        cleanup();
        return ROCQ_STATUS_HIP_ERROR;
    }
    if (hipStreamSynchronize(stream0) != hipSuccess) { // ROCQ_ASYNC_ALLOWED_SYNC
        cleanup();
        return ROCQ_STATUS_HIP_ERROR;
    }
    if (cdf_last <= 0.0) {
        cleanup();
        return ROCQ_STATUS_FAILURE;
    }

    void* d_results_void = nullptr;
    status = device_malloc(handle, &d_results_void, static_cast<size_t>(numShots) * sizeof(uint64_t));
    if (status != ROCQ_STATUS_SUCCESS) {
        cleanup();
        return status;
    }
    d_results = static_cast<uint64_t*>(d_results_void);

    const unsigned long long seed = handle->rng();
    constexpr int threads_per_block = 256;
    const int sample_blocks = compute_reduction_blocks(static_cast<size_t>(numShots), threads_per_block);
    hipLaunchKernelGGL(sample_from_cdf_kernel,
                       dim3(sample_blocks),
                       dim3(threads_per_block),
                       0,
                       stream0,
                       rank_probs[0],
                       num_outcomes,
                       d_results,
                       numShots,
                       seed);
    status = check_last_hip_error();
    if (status != ROCQ_STATUS_SUCCESS) {
        cleanup();
        return status;
    }

    status = copy_device_to_host(h_results,
                                 d_results,
                                 static_cast<size_t>(numShots) * sizeof(uint64_t),
                                 stream0);
    cleanup();
    return status;
}

inline rocqStatus_t host_sample_state(rocsvInternalHandle* handle,
                                      const std::vector<rocComplex>& host_state,
                                      unsigned numQubits,
                                      const std::vector<unsigned>& measured,
                                      unsigned numShots,
                                      uint64_t* h_results) {
    if (!handle || measured.empty() || !h_results) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    size_t state_elements = 0;
    if (!compute_power_of_two(numQubits, &state_elements) || host_state.size() != state_elements) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (measured.size() >= static_cast<size_t>(sizeof(size_t) * 8)) {
        return ROCQ_STATUS_NOT_IMPLEMENTED;
    }

    const size_t num_outcomes = size_t{1} << measured.size();
    std::vector<double> outcome_probs(num_outcomes, 0.0);
    for (size_t idx = 0; idx < state_elements; ++idx) {
        size_t outcome = 0;
        for (size_t m = 0; m < measured.size(); ++m) {
            const unsigned qubit = measured[m];
            if (!validate_qubit_index(qubit, numQubits)) {
                return ROCQ_STATUS_INVALID_VALUE;
            }
            if ((idx >> qubit) & 1ULL) {
                outcome |= (size_t{1} << m);
            }
        }
        const std::complex<double> amp = to_std_complex(host_state[idx]);
        outcome_probs[outcome] += std::norm(amp);
    }

    std::discrete_distribution<uint64_t> dist(outcome_probs.begin(), outcome_probs.end());
    for (unsigned shot = 0; shot < numShots; ++shot) {
        h_results[shot] = dist(handle->rng);
    }
    return ROCQ_STATUS_SUCCESS;
}

inline rocqStatus_t distributed_sample_host_fallback(rocsvInternalHandle* handle,
                                                     unsigned numQubits,
                                                     const std::vector<unsigned>& measured,
                                                     unsigned numShots,
                                                     uint64_t* h_results) {
    if (!distributed_host_fallback_enabled()) {
        return ROCQ_STATUS_NOT_IMPLEMENTED;
    }
    if (!handle || !handle->distributedMode || numQubits != handle->globalNumQubits) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    std::vector<rocComplex> host_full;
    rocqStatus_t status = gather_distributed_state_to_host(handle, &host_full);
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }
    return host_sample_state(handle, host_full, numQubits, measured, numShots, h_results);
}

inline rocqStatus_t distributed_sample_with_fallback(rocsvInternalHandle* handle,
                                                     unsigned numQubits,
                                                     const std::vector<unsigned>& measured,
                                                     unsigned numShots,
                                                     uint64_t* h_results) {
    rocqStatus_t status = distributed_sample_rccl(handle, numQubits, measured, numShots, h_results);
    if (status != ROCQ_STATUS_NOT_IMPLEMENTED) {
        return status;
    }
    return distributed_sample_host_fallback(handle, numQubits, measured, numShots, h_results);
}

inline rocqStatus_t accumulate_local_sample_probabilities(rocsvInternalHandle* handle,
                                                          rocComplex* state,
                                                          unsigned numQubits,
                                                          const std::vector<unsigned>& measured,
                                                          double** d_outcome_probs,
                                                          unsigned** d_measured_qubits) {
    if (!handle || !state || measured.empty() || !d_outcome_probs || !d_measured_qubits) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    *d_outcome_probs = nullptr;
    *d_measured_qubits = nullptr;
    if (measured.size() >= static_cast<size_t>(sizeof(size_t) * 8)) {
        return ROCQ_STATUS_NOT_IMPLEMENTED;
    }
    if (effective_batch_size(handle, state) != 1) {
        return ROCQ_STATUS_NOT_IMPLEMENTED;
    }

    size_t state_elements = 0;
    if (!compute_power_of_two(numQubits, &state_elements)) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    void* d_measured_void = nullptr;
    rocqStatus_t status = device_malloc(handle, &d_measured_void, measured.size() * sizeof(unsigned));
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }
    unsigned* d_measured = static_cast<unsigned*>(d_measured_void);
    status = copy_host_to_device(d_measured,
                                 measured.data(),
                                 measured.size() * sizeof(unsigned),
                                 handle->streams[0]);
    if (status != ROCQ_STATUS_SUCCESS) {
        device_free(handle, d_measured);
        return status;
    }

    const size_t num_outcomes = size_t{1} << measured.size();
    void* d_outcome_probs_void = nullptr;
    status = device_malloc(handle, &d_outcome_probs_void, num_outcomes * sizeof(double));
    if (status != ROCQ_STATUS_SUCCESS) {
        device_free(handle, d_measured);
        return status;
    }
    double* local_outcome_probs = static_cast<double*>(d_outcome_probs_void);
    if (hipMemsetAsync(local_outcome_probs, 0, num_outcomes * sizeof(double), handle->streams[0]) != hipSuccess) {
        device_free(handle, local_outcome_probs);
        device_free(handle, d_measured);
        return ROCQ_STATUS_HIP_ERROR;
    }

    constexpr int threads_per_block = 256;
    const int blocks = compute_reduction_blocks(state_elements, threads_per_block);
    hipLaunchKernelGGL(accumulate_sample_probabilities_kernel,
                       dim3(blocks),
                       dim3(threads_per_block),
                       0,
                       handle->streams[0],
                       state,
                       state_elements,
                       d_measured,
                       static_cast<unsigned>(measured.size()),
                       local_outcome_probs);
    status = check_last_hip_error();
    if (status != ROCQ_STATUS_SUCCESS) {
        device_free(handle, local_outcome_probs);
        device_free(handle, d_measured);
        return status;
    }

    *d_outcome_probs = local_outcome_probs;
    *d_measured_qubits = d_measured;
    return ROCQ_STATUS_SUCCESS;
}

inline rocqStatus_t normalize_host_probabilities(double* probabilities, size_t num_outcomes) {
    if (!probabilities || num_outcomes == 0) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    double total = 0.0;
    for (size_t i = 0; i < num_outcomes; ++i) {
        total += probabilities[i];
    }
    if (total <= 0.0 || !std::isfinite(total)) {
        return ROCQ_STATUS_FAILURE;
    }

    const double inv_total = 1.0 / total;
    for (size_t i = 0; i < num_outcomes; ++i) {
        probabilities[i] *= inv_total;
    }
    return ROCQ_STATUS_SUCCESS;
}

inline rocqStatus_t compute_local_sample_probabilities(rocsvInternalHandle* handle,
                                                       rocComplex* state,
                                                       unsigned numQubits,
                                                       const std::vector<unsigned>& measured,
                                                       double* h_probabilities) {
    if (!h_probabilities) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    double* d_outcome_probs = nullptr;
    unsigned* d_measured = nullptr;
    rocqStatus_t status = accumulate_local_sample_probabilities(
        handle, state, numQubits, measured, &d_outcome_probs, &d_measured);
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }

    const size_t num_outcomes = size_t{1} << measured.size();
    status = copy_device_to_host(h_probabilities,
                                 d_outcome_probs,
                                 num_outcomes * sizeof(double),
                                 handle->streams[0]);
    rocqStatus_t free_status = device_free(handle, d_outcome_probs);
    rocqStatus_t measured_free_status = device_free(handle, d_measured);
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }
    if (free_status != ROCQ_STATUS_SUCCESS) {
        return free_status;
    }
    if (measured_free_status != ROCQ_STATUS_SUCCESS) {
        return measured_free_status;
    }
    return normalize_host_probabilities(h_probabilities, num_outcomes);
}

inline rocqStatus_t compute_local_sample_probabilities_batch(rocsvInternalHandle* handle,
                                                             rocComplex* state,
                                                             unsigned numQubits,
                                                             const std::vector<unsigned>& measured,
                                                             double* h_probabilities) {
    if (!handle || !state || !h_probabilities || measured.empty()) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (measured.size() >= static_cast<size_t>(sizeof(size_t) * 8)) {
        return ROCQ_STATUS_NOT_IMPLEMENTED;
    }

    size_t elements_per_state = 0;
    if (!compute_power_of_two(numQubits, &elements_per_state)) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    const size_t batch_size = effective_batch_size(handle, state);
    if (batch_size > std::numeric_limits<size_t>::max() / elements_per_state) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    const size_t num_outcomes = size_t{1} << measured.size();
    if (batch_size > std::numeric_limits<size_t>::max() / num_outcomes) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    const size_t total_outcomes = batch_size * num_outcomes;
    const size_t total_elements = batch_size * elements_per_state;

    void* d_measured_void = nullptr;
    rocqStatus_t status = device_malloc(handle, &d_measured_void, measured.size() * sizeof(unsigned));
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }
    unsigned* d_measured = static_cast<unsigned*>(d_measured_void);
    status = copy_host_to_device(d_measured,
                                 measured.data(),
                                 measured.size() * sizeof(unsigned),
                                 handle->streams[0]);
    if (status != ROCQ_STATUS_SUCCESS) {
        device_free(handle, d_measured);
        return status;
    }

    void* d_outcome_probs_void = nullptr;
    status = device_malloc(handle, &d_outcome_probs_void, total_outcomes * sizeof(double));
    if (status != ROCQ_STATUS_SUCCESS) {
        device_free(handle, d_measured);
        return status;
    }
    double* d_outcome_probs = static_cast<double*>(d_outcome_probs_void);
    if (hipMemsetAsync(d_outcome_probs, 0, total_outcomes * sizeof(double), handle->streams[0]) != hipSuccess) {
        device_free(handle, d_outcome_probs);
        device_free(handle, d_measured);
        return ROCQ_STATUS_HIP_ERROR;
    }

    constexpr int threads_per_block = 256;
    const int blocks = compute_reduction_blocks(total_elements, threads_per_block);
    hipLaunchKernelGGL(accumulate_sample_probabilities_batch_kernel,
                       dim3(blocks),
                       dim3(threads_per_block),
                       0,
                       handle->streams[0],
                       state,
                       elements_per_state,
                       batch_size,
                       d_measured,
                       static_cast<unsigned>(measured.size()),
                       num_outcomes,
                       d_outcome_probs);
    status = check_last_hip_error();
    if (status != ROCQ_STATUS_SUCCESS) {
        device_free(handle, d_outcome_probs);
        device_free(handle, d_measured);
        return status;
    }

    status = copy_device_to_host(h_probabilities,
                                 d_outcome_probs,
                                 total_outcomes * sizeof(double),
                                 handle->streams[0]);
    rocqStatus_t free_status = device_free(handle, d_outcome_probs);
    rocqStatus_t measured_free_status = device_free(handle, d_measured);
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }
    if (free_status != ROCQ_STATUS_SUCCESS) {
        return free_status;
    }
    if (measured_free_status != ROCQ_STATUS_SUCCESS) {
        return measured_free_status;
    }

    for (size_t batch = 0; batch < batch_size; ++batch) {
        status = normalize_host_probabilities(h_probabilities + batch * num_outcomes, num_outcomes);
        if (status != ROCQ_STATUS_SUCCESS) {
            return status;
        }
    }
    return ROCQ_STATUS_SUCCESS;
}

inline rocqStatus_t compute_local_expectation_pauli_string_batch(rocsvInternalHandle* handle,
                                                                 rocComplex* state,
                                                                 unsigned numQubits,
                                                                 const std::vector<unsigned>& targets,
                                                                 const std::string& pauli,
                                                                 double* h_results) {
    if (!handle || !state || !h_results || targets.empty() || pauli.size() != targets.size()) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    size_t elements_per_state = 0;
    if (!compute_power_of_two(numQubits, &elements_per_state)) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    const size_t batch_size = effective_batch_size(handle, state);
    if (batch_size > 65535) {
        return ROCQ_STATUS_NOT_IMPLEMENTED;
    }

    constexpr int threads_per_block = 256;
    const int blocks = compute_reduction_blocks(elements_per_state, threads_per_block);
    if (batch_size > std::numeric_limits<size_t>::max() / static_cast<size_t>(blocks)) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    void* d_targets_void = nullptr;
    rocqStatus_t status = device_malloc(handle, &d_targets_void, targets.size() * sizeof(unsigned));
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }
    unsigned* d_targets = static_cast<unsigned*>(d_targets_void);
    status = copy_host_to_device(d_targets,
                                 targets.data(),
                                 targets.size() * sizeof(unsigned),
                                 handle->streams[0]);
    if (status != ROCQ_STATUS_SUCCESS) {
        device_free(handle, d_targets);
        return status;
    }

    void* d_pauli_void = nullptr;
    status = device_malloc(handle, &d_pauli_void, pauli.size() * sizeof(char));
    if (status != ROCQ_STATUS_SUCCESS) {
        device_free(handle, d_targets);
        return status;
    }
    char* d_pauli = static_cast<char*>(d_pauli_void);
    status = copy_host_to_device(d_pauli,
                                 pauli.data(),
                                 pauli.size() * sizeof(char),
                                 handle->streams[0]);
    if (status != ROCQ_STATUS_SUCCESS) {
        device_free(handle, d_pauli);
        device_free(handle, d_targets);
        return status;
    }

    void* d_block_sums_void = nullptr;
    const size_t total_blocks = batch_size * static_cast<size_t>(blocks);
    status = device_malloc(handle, &d_block_sums_void, total_blocks * sizeof(double));
    if (status != ROCQ_STATUS_SUCCESS) {
        device_free(handle, d_pauli);
        device_free(handle, d_targets);
        return status;
    }
    double* d_block_sums = static_cast<double*>(d_block_sums_void);

    hipLaunchKernelGGL(reduce_expectation_pauli_string_batch_kernel,
                       dim3(blocks, static_cast<unsigned>(batch_size)),
                       dim3(threads_per_block),
                       threads_per_block * sizeof(double),
                       handle->streams[0],
                       state,
                       elements_per_state,
                       d_pauli,
                       d_targets,
                       static_cast<unsigned>(targets.size()),
                       d_block_sums);
    status = check_last_hip_error();
    if (status == ROCQ_STATUS_SUCCESS) {
        for (size_t batch = 0; batch < batch_size; ++batch) {
            status = reduce_blocks_to_scalar(handle,
                                             d_block_sums + batch * static_cast<size_t>(blocks),
                                             blocks,
                                             h_results + batch);
            if (status != ROCQ_STATUS_SUCCESS) {
                break;
            }
        }
    }

    rocqStatus_t block_free_status = device_free(handle, d_block_sums);
    rocqStatus_t pauli_free_status = device_free(handle, d_pauli);
    rocqStatus_t targets_free_status = device_free(handle, d_targets);
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }
    if (block_free_status != ROCQ_STATUS_SUCCESS) {
        return block_free_status;
    }
    if (pauli_free_status != ROCQ_STATUS_SUCCESS) {
        return pauli_free_status;
    }
    return targets_free_status;
}

inline rocqStatus_t compute_distributed_sample_probabilities(rocsvInternalHandle* handle,
                                                             unsigned numQubits,
                                                             const std::vector<unsigned>& measured,
                                                             double* h_probabilities) {
    if (!h_probabilities) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    std::vector<double*> rank_probs;
    rocqStatus_t status = accumulate_distributed_sample_probabilities_rccl(
        handle, numQubits, measured, &rank_probs);
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }

    if (hipSetDevice(handle->distributedDeviceIds[0]) != hipSuccess) {
        (void)free_distributed_probability_buffers(handle, rank_probs);
        return ROCQ_STATUS_HIP_ERROR;
    }

    const size_t num_outcomes = size_t{1} << measured.size();
    status = copy_device_to_host(h_probabilities,
                                 rank_probs[0],
                                 num_outcomes * sizeof(double),
                                 handle->distributedStreams[0]);
    rocqStatus_t free_status = free_distributed_probability_buffers(handle, rank_probs);
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }
    if (free_status != ROCQ_STATUS_SUCCESS) {
        return free_status;
    }
    return normalize_host_probabilities(h_probabilities, num_outcomes);
}

inline rocqStatus_t apply_sparse_matrix_distributed_local(rocsvInternalHandle* handle,
                                                          unsigned numQubits,
                                                          const std::vector<unsigned>& targets,
                                                          const rocComplex* d_data,
                                                          const size_t* d_indices,
                                                          const size_t* d_indptr,
                                                          size_t rows,
                                                          size_t cols,
                                                          size_t nnz) {
    if (!handle || handle->distributedGpuCount <= 0 || handle->localSliceElements == 0) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    const size_t gpu_count = static_cast<size_t>(handle->distributedGpuCount);
    if (handle->distributedDeviceIds.size() < gpu_count ||
        handle->distributedStreams.size() < gpu_count ||
        handle->distributedSlices.size() < gpu_count ||
        handle->distributedSwapBuffers.size() < gpu_count) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (numQubits != handle->globalNumQubits) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (!distributed_all_qubits_local(handle, targets)) {
        return apply_sparse_matrix_distributed_host_fallback(handle,
                                                             numQubits,
                                                             targets,
                                                             d_data,
                                                             d_indices,
                                                             d_indptr,
                                                             rows,
                                                             cols,
                                                             nnz);
    }

    int original_device = 0;
    if (hipGetDevice(&original_device) != hipSuccess) {
        return ROCQ_STATUS_HIP_ERROR;
    }
    auto restore_device = [&]() {
        (void)hipSetDevice(original_device);
    };

    if (hipSetDevice(handle->distributedDeviceIds[0]) != hipSuccess) {
        restore_device();
        return ROCQ_STATUS_HIP_ERROR;
    }

    std::vector<rocComplex> h_data;
    std::vector<size_t> h_indices;
    std::vector<size_t> h_indptr;
    rocqStatus_t status = copy_sparse_matrix_from_device(d_data,
                                                         d_indices,
                                                         d_indptr,
                                                         rows,
                                                         nnz,
                                                         handle->streams[0],
                                                         &h_data,
                                                         &h_indices,
                                                         &h_indptr);
    if (status != ROCQ_STATUS_SUCCESS) {
        restore_device();
        return status;
    }

    constexpr int threads_per_block = 256;
    const int blocks = compute_reduction_blocks(handle->localSliceElements, threads_per_block);
    const size_t bytes_per_slice = handle->localSliceElements * sizeof(rocComplex);

    for (int rank = 0; rank < handle->distributedGpuCount; ++rank) {
        const size_t rank_idx = static_cast<size_t>(rank);
        if (hipSetDevice(handle->distributedDeviceIds[rank_idx]) != hipSuccess) {
            restore_device();
            return ROCQ_STATUS_HIP_ERROR;
        }
        if (!handle->distributedSlices[rank_idx] || !handle->distributedSwapBuffers[rank_idx]) {
            restore_device();
            return ROCQ_STATUS_INVALID_VALUE;
        }

        void* d_targets_void = nullptr;
        void* d_data_void = nullptr;
        void* d_indices_void = nullptr;
        void* d_indptr_void = nullptr;
        auto cleanup_rank = [&]() {
            if (d_targets_void) {
                (void)device_free(handle, d_targets_void);
            }
            if (d_data_void) {
                (void)device_free(handle, d_data_void);
            }
            if (d_indices_void) {
                (void)device_free(handle, d_indices_void);
            }
            if (d_indptr_void) {
                (void)device_free(handle, d_indptr_void);
            }
        };

        status = device_malloc(handle, &d_targets_void, targets.size() * sizeof(unsigned));
        if (status != ROCQ_STATUS_SUCCESS) {
            restore_device();
            return status;
        }
        status = copy_host_to_device(d_targets_void,
                                     targets.data(),
                                     targets.size() * sizeof(unsigned),
                                     handle->distributedStreams[rank_idx]);
        if (status != ROCQ_STATUS_SUCCESS) {
            cleanup_rank();
            restore_device();
            return status;
        }

        if (nnz > 0) {
            status = device_malloc(handle, &d_data_void, h_data.size() * sizeof(rocComplex));
            if (status != ROCQ_STATUS_SUCCESS) {
                cleanup_rank();
                restore_device();
                return status;
            }
            status = copy_host_to_device(d_data_void,
                                         h_data.data(),
                                         h_data.size() * sizeof(rocComplex),
                                         handle->distributedStreams[rank_idx]);
            if (status != ROCQ_STATUS_SUCCESS) {
                cleanup_rank();
                restore_device();
                return status;
            }

            status = device_malloc(handle, &d_indices_void, h_indices.size() * sizeof(size_t));
            if (status != ROCQ_STATUS_SUCCESS) {
                cleanup_rank();
                restore_device();
                return status;
            }
            status = copy_host_to_device(d_indices_void,
                                         h_indices.data(),
                                         h_indices.size() * sizeof(size_t),
                                         handle->distributedStreams[rank_idx]);
            if (status != ROCQ_STATUS_SUCCESS) {
                cleanup_rank();
                restore_device();
                return status;
            }
        }

        status = device_malloc(handle, &d_indptr_void, h_indptr.size() * sizeof(size_t));
        if (status != ROCQ_STATUS_SUCCESS) {
            cleanup_rank();
            restore_device();
            return status;
        }
        status = copy_host_to_device(d_indptr_void,
                                     h_indptr.data(),
                                     h_indptr.size() * sizeof(size_t),
                                     handle->distributedStreams[rank_idx]);
        if (status != ROCQ_STATUS_SUCCESS) {
            cleanup_rank();
            restore_device();
            return status;
        }

        hipLaunchKernelGGL(apply_sparse_matrix_kernel,
                           dim3(blocks),
                           dim3(threads_per_block),
                           0,
                           handle->distributedStreams[rank_idx],
                           handle->distributedSlices[rank_idx],
                           handle->distributedSwapBuffers[rank_idx],
                           handle->localSliceElements,
                           static_cast<unsigned*>(d_targets_void),
                           static_cast<unsigned>(targets.size()),
                           static_cast<rocComplex*>(d_data_void),
                           static_cast<size_t*>(d_indices_void),
                           static_cast<size_t*>(d_indptr_void));
        status = check_last_hip_error();
        if (status != ROCQ_STATUS_SUCCESS) {
            cleanup_rank();
            restore_device();
            return status;
        }

        if (hipMemcpyAsync(handle->distributedSlices[rank_idx],
                           handle->distributedSwapBuffers[rank_idx],
                           bytes_per_slice,
                           hipMemcpyDeviceToDevice,
                           handle->distributedStreams[rank_idx]) != hipSuccess) {
            cleanup_rank();
            restore_device();
            return ROCQ_STATUS_HIP_ERROR;
        }
        if (hipStreamSynchronize(handle->distributedStreams[rank_idx]) != hipSuccess) { // ROCQ_ASYNC_ALLOWED_SYNC
            cleanup_rank();
            restore_device();
            return ROCQ_STATUS_HIP_ERROR;
        }
        cleanup_rank();
    }

    restore_device();
    return ROCQ_STATUS_SUCCESS;
}

} // namespace

rocqStatus_t rocsvCreate(rocsvHandle_t* handle) {
    if (!handle) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    *handle = new rocsvInternalHandle();
    if (hipStreamCreate(&((*handle)->streams[0])) != hipSuccess) {
        delete *handle;
        *handle = nullptr;
        return ROCQ_STATUS_HIP_ERROR;
    }
    (*handle)->ownsPrimaryStream = true;
    (*handle)->batchSize = 1;
    (*handle)->numQubits = 0;
    (*handle)->d_state = nullptr;
    (*handle)->ownsState = false;
    (*handle)->pinnedBuffer = nullptr;
    (*handle)->pinnedBufferBytes = 0;
    (*handle)->memHandler = {nullptr, nullptr, nullptr};

    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess) {
        hipStreamDestroy((*handle)->streams[0]);
        delete *handle;
        *handle = nullptr;
        return ROCQ_STATUS_HIP_ERROR;
    }
    if (device_count <= 0) {
        hipStreamDestroy((*handle)->streams[0]);
        delete *handle;
        *handle = nullptr;
        return ROCQ_STATUS_HIP_ERROR;
    }
    (*handle)->distributedGpuCount = device_count;
    (*handle)->distributedDeviceIds.resize(static_cast<size_t>((*handle)->distributedGpuCount));
    for (int i = 0; i < (*handle)->distributedGpuCount; ++i) {
        (*handle)->distributedDeviceIds[static_cast<size_t>(i)] = i;
    }
    return ROCQ_STATUS_SUCCESS;
}

rocqStatus_t rocsvDestroy(rocsvHandle_t handle) {
    if (handle) {
        rocsvFreePinnedBuffer(handle);
        rocsvFreeState(handle);
        if (handle->ownsPrimaryStream && handle->streams[0]) {
            hipStreamDestroy(handle->streams[0]);
        }
        delete handle;
    }
    return ROCQ_STATUS_SUCCESS;
}

rocqStatus_t rocsvSetStream(rocsvHandle_t handle, hipStream_t stream) {
    if (!handle) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (handle->streams[0] == stream) {
        return ROCQ_STATUS_SUCCESS;
    }

    if (handle->ownsPrimaryStream && handle->streams[0]) {
        if (hipStreamDestroy(handle->streams[0]) != hipSuccess) {
            return ROCQ_STATUS_HIP_ERROR;
        }
    }

    handle->streams[0] = stream;
    handle->ownsPrimaryStream = false;
    return ROCQ_STATUS_SUCCESS;
}

rocqStatus_t rocsvGetStream(rocsvHandle_t handle, hipStream_t* stream) {
    if (!handle || !stream) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    *stream = handle->streams[0];
    return ROCQ_STATUS_SUCCESS;
}

rocqStatus_t rocsvSetDeviceMemHandler(rocsvHandle_t handle, const rocqDeviceMemHandler_t* handler) {
    if (!handle) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (!handler) {
        handle->memHandler = {nullptr, nullptr, nullptr};
        return ROCQ_STATUS_SUCCESS;
    }

    const bool hasMalloc = handler->device_malloc != nullptr;
    const bool hasFree = handler->device_free != nullptr;
    if (hasMalloc != hasFree) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    handle->memHandler = *handler;
    return ROCQ_STATUS_SUCCESS;
}

rocqStatus_t rocsvGetDeviceMemHandler(rocsvHandle_t handle, rocqDeviceMemHandler_t* handler) {
    if (!handle || !handler) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    *handler = handle->memHandler;
    return ROCQ_STATUS_SUCCESS;
}

rocqStatus_t rocsvGetNumGpus(rocsvHandle_t handle, int* num_gpus) {
    if (!handle || !num_gpus) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (hipGetDeviceCount(num_gpus) != hipSuccess) {
        return ROCQ_STATUS_HIP_ERROR;
    }
    return ROCQ_STATUS_SUCCESS;
}

rocqStatus_t rocsvGetDistributedInfo(rocsvHandle_t handle, rocsvDistributedInfo_t* info) {
    if (!handle || !info) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    info->distributed_mode = handle->distributedMode ? 1 : 0;
    info->gpu_count = handle->distributedMode ? handle->distributedGpuCount : 1;
    info->global_num_qubits = handle->distributedMode ? handle->globalNumQubits : handle->numQubits;
    info->local_num_qubits_per_gpu =
        handle->distributedMode ? handle->numLocalQubitsPerGpu : handle->numQubits;
    info->global_slice_qubits = handle->distributedMode ? handle->numGlobalSliceQubits : 0;
    info->local_slice_elements = handle->distributedMode ? handle->localSliceElements : 0;
    return ROCQ_STATUS_SUCCESS;
}

rocqStatus_t rocsvGetDistributedBackend(rocsvHandle_t handle, rocsvDistributedBackend_t* backend) {
    if (!handle || !backend) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    *backend = distributed_active_backend(handle);
    return ROCQ_STATUS_SUCCESS;
}

const char* rocsvDistributedBackendName(rocsvDistributedBackend_t backend) {
    switch (backend) {
        case ROCSV_DISTRIBUTED_BACKEND_NONE:
            return "none";
        case ROCSV_DISTRIBUTED_BACKEND_HOST_FALLBACK:
            return "host_fallback";
        case ROCSV_DISTRIBUTED_BACKEND_RCCL:
            return "rccl";
        default:
            return "unknown";
    }
}

rocqStatus_t rocsvSynchronize(rocsvHandle_t handle) {
    if (!handle) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (handle->distributedMode) {
        return sync_distributed_streams(handle); // ROCQ_ASYNC_ALLOWED_SYNC
    }
    if (hipStreamSynchronize(handle->streams[0]) != hipSuccess) { // ROCQ_ASYNC_ALLOWED_SYNC
        return ROCQ_STATUS_HIP_ERROR;
    }
    return ROCQ_STATUS_SUCCESS;
}

rocqStatus_t rocsvAllocateState(rocsvHandle_t handle,
                                unsigned numQubits,
                                rocComplex** d_state,
                                size_t batchSize) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;

    if (handle->distributedMode || !handle->distributedSlices.empty()) {
        rocqStatus_t clear_status = clear_distributed_state_storage(handle);
        if (clear_status != ROCQ_STATUS_SUCCESS) {
            return clear_status;
        }
    }

    size_t num_elements_per_state = 0;
    if (!compute_power_of_two(numQubits, &num_elements_per_state)) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    handle->batchSize = batchSize > 0 ? batchSize : 1;
    handle->numQubits = numQubits;

    if (handle->d_state && handle->ownsState) {
        rocqStatus_t free_status = device_free(handle, handle->d_state);
        if (free_status != ROCQ_STATUS_SUCCESS) {
            return free_status;
        }
        handle->d_state = nullptr;
        handle->ownsState = false;
    }

    const size_t total_elements = handle->batchSize * num_elements_per_state;
    void* allocated_ptr = nullptr;
    rocqStatus_t alloc_status = device_malloc(handle, &allocated_ptr, total_elements * sizeof(rocComplex));
    if (alloc_status != ROCQ_STATUS_SUCCESS) {
        return alloc_status;
    }

    if (d_state) {
        *d_state = static_cast<rocComplex*>(allocated_ptr);
    }

    handle->d_state = static_cast<rocComplex*>(allocated_ptr);
    handle->ownsState = true;
    return ROCQ_STATUS_SUCCESS;
}

rocqStatus_t rocsvFreeState(rocsvHandle_t handle) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    if (handle->distributedMode || !handle->distributedSlices.empty()) {
        rocqStatus_t clear_status = clear_distributed_state_storage(handle);
        if (clear_status != ROCQ_STATUS_SUCCESS) {
            return clear_status;
        }
    }
    if (handle->d_state && handle->ownsState) {
        rocqStatus_t free_status = device_free(handle, handle->d_state);
        if (free_status != ROCQ_STATUS_SUCCESS) {
            return free_status;
        }
    }
    handle->d_state = nullptr;
    handle->ownsState = false;
    handle->numQubits = 0;
    handle->batchSize = 1;
    return ROCQ_STATUS_SUCCESS;
}

rocqStatus_t rocsvInitializeState(rocsvHandle_t handle,
                                  rocComplex* d_state,
                                  unsigned numQubits) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    if (uses_distributed_state(handle, d_state)) {
        if (numQubits != handle->globalNumQubits) {
            return ROCQ_STATUS_INVALID_VALUE;
        }
        return rocsvInitializeDistributedState(handle);
    }
    rocComplex* target_state = resolve_state_pointer(handle, d_state);
    if (!target_state) return ROCQ_STATUS_INVALID_VALUE;

    size_t elements_per_state = 0;
    if (!compute_power_of_two(numQubits, &elements_per_state)) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    const size_t batch_size = effective_batch_size(handle, target_state);
    const size_t total_elements = batch_size * elements_per_state;

    if (hipMemsetAsync(target_state, 0, total_elements * sizeof(rocComplex), handle->streams[0]) != hipSuccess) {
        return ROCQ_STATUS_HIP_ERROR;
    }

    const rocComplex one = make_complex(1.0, 0.0);
    for (size_t batch = 0; batch < batch_size; ++batch) {
        rocComplex* amp0 = target_state + batch * elements_per_state;
        if (hipMemcpyAsync(amp0, &one, sizeof(rocComplex), hipMemcpyHostToDevice, handle->streams[0]) != hipSuccess) {
            return ROCQ_STATUS_HIP_ERROR;
        }
    }
    if (target_state == handle->d_state) {
        handle->numQubits = numQubits;
        handle->batchSize = batch_size;
    }
    return ROCQ_STATUS_SUCCESS;
}

// --- Single-GPU compatibility helpers for distributed APIs ------------------

rocqStatus_t rocsvAllocateDistributedState(rocsvHandle_t handle,
                                           unsigned totalNumQubits) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;

    int visible_devices = 0;
    if (hipGetDeviceCount(&visible_devices) != hipSuccess) {
        return ROCQ_STATUS_HIP_ERROR;
    }
    if (visible_devices <= 0) {
        return ROCQ_STATUS_HIP_ERROR;
    }

    int active_gpus = visible_devices;
    if (!is_power_of_two_int(active_gpus)) {
        int pow2 = 1;
        while ((pow2 << 1) <= active_gpus) {
            pow2 <<= 1;
        }
        active_gpus = pow2;
    }
    while (active_gpus > 1 &&
           integer_log2(static_cast<unsigned>(active_gpus)) > totalNumQubits) {
        active_gpus >>= 1;
    }

    const unsigned num_global_slice_qubits =
        (active_gpus > 1) ? integer_log2(static_cast<unsigned>(active_gpus)) : 0;
    if (totalNumQubits < num_global_slice_qubits) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    rocqStatus_t free_status = rocsvFreeState(handle);
    if (free_status != ROCQ_STATUS_SUCCESS) {
        return free_status;
    }

    unsigned num_local_qubits = totalNumQubits - num_global_slice_qubits;
    size_t local_elements = 0;
    if (!compute_power_of_two(num_local_qubits, &local_elements)) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    reset_distributed_metadata(handle);
    handle->distributedMode = true;
    handle->distributedGpuCount = active_gpus;
    handle->globalNumQubits = totalNumQubits;
    handle->numQubits = totalNumQubits;
    handle->numGlobalSliceQubits = num_global_slice_qubits;
    handle->numLocalQubitsPerGpu = num_local_qubits;
    handle->localSliceElements = local_elements;
    handle->batchSize = 1;

    handle->distributedDeviceIds.resize(static_cast<size_t>(active_gpus));
    handle->distributedStreams.resize(static_cast<size_t>(active_gpus), nullptr);
    handle->distributedSlices.resize(static_cast<size_t>(active_gpus), nullptr);
    handle->distributedSwapBuffers.resize(static_cast<size_t>(active_gpus), nullptr);
    handle->distributedTargetScratch.resize(static_cast<size_t>(active_gpus), nullptr);
    handle->distributedMatrixScratch.resize(static_cast<size_t>(active_gpus), nullptr);

    int original_device = 0;
    if (hipGetDevice(&original_device) != hipSuccess) {
        clear_distributed_state_storage(handle);
        return ROCQ_STATUS_HIP_ERROR;
    }

    const size_t bytes_per_slice = local_elements * sizeof(rocComplex);
    for (int rank = 0; rank < active_gpus; ++rank) {
        handle->distributedDeviceIds[static_cast<size_t>(rank)] = rank;
        if (hipSetDevice(rank) != hipSuccess) {
            (void)hipSetDevice(original_device);
            clear_distributed_state_storage(handle);
            return ROCQ_STATUS_HIP_ERROR;
        }

        if (hipStreamCreate(&handle->distributedStreams[static_cast<size_t>(rank)]) != hipSuccess) {
            (void)hipSetDevice(original_device);
            clear_distributed_state_storage(handle);
            return ROCQ_STATUS_HIP_ERROR;
        }

        void* slice_ptr = nullptr;
        rocqStatus_t alloc_status = device_malloc(handle, &slice_ptr, bytes_per_slice);
        if (alloc_status != ROCQ_STATUS_SUCCESS) {
            (void)hipSetDevice(original_device);
            clear_distributed_state_storage(handle);
            return alloc_status;
        }
        handle->distributedSlices[static_cast<size_t>(rank)] = static_cast<rocComplex*>(slice_ptr);

        void* swap_ptr = nullptr;
        alloc_status = device_malloc(handle, &swap_ptr, bytes_per_slice);
        if (alloc_status != ROCQ_STATUS_SUCCESS) {
            (void)hipSetDevice(original_device);
            clear_distributed_state_storage(handle);
            return alloc_status;
        }
        handle->distributedSwapBuffers[static_cast<size_t>(rank)] =
            static_cast<rocComplex*>(swap_ptr);

        void* target_scratch = nullptr;
        alloc_status = device_malloc(handle,
                                     &target_scratch,
                                     kMaxDistributedMatrixTargetQubits * sizeof(unsigned));
        if (alloc_status != ROCQ_STATUS_SUCCESS) {
            (void)hipSetDevice(original_device);
            clear_distributed_state_storage(handle);
            return alloc_status;
        }
        handle->distributedTargetScratch[static_cast<size_t>(rank)] =
            static_cast<unsigned*>(target_scratch);

        void* matrix_scratch = nullptr;
        alloc_status = device_malloc(handle,
                                     &matrix_scratch,
                                     kMaxDistributedMatrixElements * sizeof(rocComplex));
        if (alloc_status != ROCQ_STATUS_SUCCESS) {
            (void)hipSetDevice(original_device);
            clear_distributed_state_storage(handle);
            return alloc_status;
        }
        handle->distributedMatrixScratch[static_cast<size_t>(rank)] =
            static_cast<rocComplex*>(matrix_scratch);
    }

    if (hipSetDevice(original_device) != hipSuccess) {
        clear_distributed_state_storage(handle);
        return ROCQ_STATUS_HIP_ERROR;
    }

    rocqStatus_t rccl_status = initialize_distributed_rccl_comms(handle);
    if (rccl_status != ROCQ_STATUS_SUCCESS) {
        clear_distributed_state_storage(handle);
        return rccl_status;
    }

    handle->d_state = handle->distributedSlices[0];
    handle->ownsState = false;
    return ROCQ_STATUS_SUCCESS;
}

rocqStatus_t rocsvInitializeDistributedState(rocsvHandle_t handle) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    if (!handle->distributedMode || handle->distributedGpuCount <= 0) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (handle->localSliceElements == 0) {
        return ROCQ_STATUS_SUCCESS;
    }

    const size_t bytes_per_slice = handle->localSliceElements * sizeof(rocComplex);
    for (int rank = 0; rank < handle->distributedGpuCount; ++rank) {
        if (hipSetDevice(handle->distributedDeviceIds[static_cast<size_t>(rank)]) != hipSuccess) {
            return ROCQ_STATUS_HIP_ERROR;
        }
        if (hipMemsetAsync(handle->distributedSlices[static_cast<size_t>(rank)],
                           0,
                           bytes_per_slice,
                           handle->distributedStreams[static_cast<size_t>(rank)]) != hipSuccess) {
            return ROCQ_STATUS_HIP_ERROR;
        }
    }

    const rocComplex one = make_complex(1.0, 0.0);
    if (hipSetDevice(handle->distributedDeviceIds[0]) != hipSuccess) {
        return ROCQ_STATUS_HIP_ERROR;
    }
    if (hipMemcpyAsync(handle->distributedSlices[0],
                       &one,
                       sizeof(rocComplex),
                       hipMemcpyHostToDevice,
                       handle->distributedStreams[0]) != hipSuccess) {
        return ROCQ_STATUS_HIP_ERROR;
    }
    return ROCQ_STATUS_SUCCESS;
}

rocqStatus_t rocsvSwapIndexBits(rocsvHandle_t handle,
                                unsigned qubit_idx1,
                                unsigned qubit_idx2) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    if (qubit_idx1 == qubit_idx2) {
        return ROCQ_STATUS_SUCCESS;
    }
    if (!validate_qubit_index(qubit_idx1, handle->numQubits) ||
        !validate_qubit_index(qubit_idx2, handle->numQubits)) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    if (handle->distributedMode) {
        const bool local_local = distributed_qubit_local(handle, qubit_idx1) &&
                                 distributed_qubit_local(handle, qubit_idx2);
        if (local_local) {
            if (handle->localSliceElements < 2) {
                return ROCQ_STATUS_SUCCESS;
            }
            constexpr int threads_per_block = 256;
            const int blocks = static_cast<int>(
                (handle->localSliceElements + threads_per_block - 1) / threads_per_block);
            for (int rank = 0; rank < handle->distributedGpuCount; ++rank) {
                if (hipSetDevice(handle->distributedDeviceIds[static_cast<size_t>(rank)]) != hipSuccess) {
                    return ROCQ_STATUS_HIP_ERROR;
                }
                hipLaunchKernelGGL(local_bit_swap_permutation_kernel,
                                   dim3(blocks > 0 ? blocks : 1),
                                   dim3(threads_per_block),
                                   0,
                                   handle->distributedStreams[static_cast<size_t>(rank)],
                                   handle->distributedSlices[static_cast<size_t>(rank)],
                                   handle->distributedSwapBuffers[static_cast<size_t>(rank)],
                                   handle->localSliceElements,
                                   qubit_idx1,
                                   qubit_idx2);
                rocqStatus_t status = check_last_hip_error();
                if (status != ROCQ_STATUS_SUCCESS) {
                    return status;
                }
            }
            return ROCQ_STATUS_SUCCESS;
        }
        return distributed_swap_bits_host_remap(handle, qubit_idx1, qubit_idx2);
    }

    if (!handle->d_state || handle->numQubits == 0) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    return rocsvApplySWAP(handle, handle->d_state, handle->numQubits, qubit_idx1, qubit_idx2);
}

// --- Single-qubit named gates ------------------------------------------------

rocqStatus_t rocsvApplyH(rocsvHandle_t handle,
                         rocComplex* d_state,
                         unsigned numQubits,
                         unsigned targetQubit) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) return ROCQ_STATUS_INVALID_VALUE;

    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    const rocComplex m00 = make_complex(inv_sqrt2, 0.0);
    const rocComplex m01 = make_complex(inv_sqrt2, 0.0);
    const rocComplex m10 = make_complex(inv_sqrt2, 0.0);
    const rocComplex m11 = make_complex(-inv_sqrt2, 0.0);
    return launch_single_qubit_matrix(handle, state, numQubits, targetQubit, m00, m01, m10, m11);
}

rocqStatus_t rocsvApplyX(rocsvHandle_t handle,
                         rocComplex* d_state,
                         unsigned numQubits,
                         unsigned targetQubit) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) return ROCQ_STATUS_INVALID_VALUE;

    const rocComplex zero = make_complex(0.0, 0.0);
    const rocComplex one = make_complex(1.0, 0.0);
    return launch_single_qubit_matrix(handle, state, numQubits, targetQubit, zero, one, one, zero);
}

rocqStatus_t rocsvApplyY(rocsvHandle_t handle,
                         rocComplex* d_state,
                         unsigned numQubits,
                         unsigned targetQubit) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) return ROCQ_STATUS_INVALID_VALUE;

    const rocComplex zero = make_complex(0.0, 0.0);
    const rocComplex minus_i = make_complex(0.0, -1.0);
    const rocComplex plus_i = make_complex(0.0, 1.0);
    return launch_single_qubit_matrix(handle, state, numQubits, targetQubit, zero, minus_i, plus_i, zero);
}

rocqStatus_t rocsvApplyZ(rocsvHandle_t handle,
                         rocComplex* d_state,
                         unsigned numQubits,
                         unsigned targetQubit) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) return ROCQ_STATUS_INVALID_VALUE;

    const rocComplex one = make_complex(1.0, 0.0);
    const rocComplex minus_one = make_complex(-1.0, 0.0);
    const rocComplex zero = make_complex(0.0, 0.0);
    return launch_single_qubit_matrix(handle, state, numQubits, targetQubit, one, zero, zero, minus_one);
}

rocqStatus_t rocsvApplyS(rocsvHandle_t handle,
                         rocComplex* d_state,
                         unsigned numQubits,
                         unsigned targetQubit) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) return ROCQ_STATUS_INVALID_VALUE;

    const rocComplex one = make_complex(1.0, 0.0);
    const rocComplex zero = make_complex(0.0, 0.0);
    const rocComplex plus_i = make_complex(0.0, 1.0);
    return launch_single_qubit_matrix(handle, state, numQubits, targetQubit, one, zero, zero, plus_i);
}

rocqStatus_t rocsvApplySdg(rocsvHandle_t handle,
                           rocComplex* d_state,
                           unsigned numQubits,
                           unsigned targetQubit) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) return ROCQ_STATUS_INVALID_VALUE;

    const rocComplex one = make_complex(1.0, 0.0);
    const rocComplex zero = make_complex(0.0, 0.0);
    const rocComplex minus_i = make_complex(0.0, -1.0);
    return launch_single_qubit_matrix(handle, state, numQubits, targetQubit, one, zero, zero, minus_i);
}

rocqStatus_t rocsvApplyT(rocsvHandle_t handle,
                         rocComplex* d_state,
                         unsigned numQubits,
                         unsigned targetQubit) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) return ROCQ_STATUS_INVALID_VALUE;

    constexpr double pi = 3.14159265358979323846;
    const double phase = pi / 4.0;
    const rocComplex one = make_complex(1.0, 0.0);
    const rocComplex zero = make_complex(0.0, 0.0);
    const rocComplex t_phase = make_complex(std::cos(phase), std::sin(phase));
    return launch_single_qubit_matrix(handle, state, numQubits, targetQubit, one, zero, zero, t_phase);
}

rocqStatus_t rocsvApplyTdg(rocsvHandle_t handle,
                           rocComplex* d_state,
                           unsigned numQubits,
                           unsigned targetQubit) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) return ROCQ_STATUS_INVALID_VALUE;

    constexpr double pi = 3.14159265358979323846;
    const double phase = -pi / 4.0;
    const rocComplex one = make_complex(1.0, 0.0);
    const rocComplex zero = make_complex(0.0, 0.0);
    const rocComplex tdg_phase = make_complex(std::cos(phase), std::sin(phase));
    return launch_single_qubit_matrix(handle, state, numQubits, targetQubit, one, zero, zero, tdg_phase);
}

// --- Parametrised single-qubit rotations ------------------------------------

enum class RotationAxis {
    X,
    Y,
    Z,
};

inline std::vector<rocComplex> make_rotation_matrices(const double* h_thetas,
                                                      size_t thetaCount,
                                                      RotationAxis axis) {
    std::vector<rocComplex> matrices;
    matrices.reserve(thetaCount * 4ULL);
    for (size_t batch = 0; batch < thetaCount; ++batch) {
        const double half = h_thetas[batch] / 2.0;
        const double c = std::cos(half);
        const double s = std::sin(half);
        if (axis == RotationAxis::X) {
            const rocComplex cval = make_complex(c, 0.0);
            const rocComplex minus_i_s = make_complex(0.0, -s);
            matrices.push_back(cval);
            matrices.push_back(minus_i_s);
            matrices.push_back(minus_i_s);
            matrices.push_back(cval);
        } else if (axis == RotationAxis::Y) {
            matrices.push_back(make_complex(c, 0.0));
            matrices.push_back(make_complex(-s, 0.0));
            matrices.push_back(make_complex(s, 0.0));
            matrices.push_back(make_complex(c, 0.0));
        } else {
            matrices.push_back(make_complex(c, -s));
            matrices.push_back(make_complex(0.0, 0.0));
            matrices.push_back(make_complex(0.0, 0.0));
            matrices.push_back(make_complex(c, s));
        }
    }
    return matrices;
}

inline std::vector<rocComplex> make_phase_matrices(const double* h_thetas,
                                                   size_t thetaCount) {
    std::vector<rocComplex> matrices;
    matrices.reserve(thetaCount * 4ULL);
    const rocComplex one = make_complex(1.0, 0.0);
    const rocComplex zero = make_complex(0.0, 0.0);
    for (size_t batch = 0; batch < thetaCount; ++batch) {
        const double theta = h_thetas[batch];
        matrices.push_back(one);
        matrices.push_back(zero);
        matrices.push_back(zero);
        matrices.push_back(make_complex(std::cos(theta), std::sin(theta)));
    }
    return matrices;
}

inline rocqStatus_t rocsvApplyRotationBatch(rocsvHandle_t handle,
                                            rocComplex* d_state,
                                            unsigned numQubits,
                                            unsigned targetQubit,
                                            const double* h_thetas,
                                            size_t thetaCount,
                                            RotationAxis axis) {
    if (!handle || !h_thetas || thetaCount == 0) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    const size_t batch_size = effective_batch_size(handle, state);
    if (thetaCount != batch_size) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    std::vector<rocComplex> matrices = make_rotation_matrices(h_thetas, thetaCount, axis);
    return launch_single_qubit_matrix_batch(handle, state, numQubits, targetQubit, matrices);
}

rocqStatus_t rocsvApplyRx(rocsvHandle_t handle,
                          rocComplex* d_state,
                          unsigned numQubits,
                          unsigned targetQubit,
                          double theta) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) return ROCQ_STATUS_INVALID_VALUE;

    const double half = theta / 2.0;
    const rocComplex c = make_complex(std::cos(half), 0.0);
    const rocComplex minus_i_s = make_complex(0.0, -std::sin(half));
    return launch_single_qubit_matrix(handle, state, numQubits, targetQubit,
                                      c, minus_i_s, minus_i_s, c);
}

rocqStatus_t rocsvApplyRy(rocsvHandle_t handle,
                          rocComplex* d_state,
                          unsigned numQubits,
                          unsigned targetQubit,
                          double theta) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) return ROCQ_STATUS_INVALID_VALUE;

    const double half = theta / 2.0;
    const rocComplex c = make_complex(std::cos(half), 0.0);
    const rocComplex minus_s = make_complex(-std::sin(half), 0.0);
    const rocComplex plus_s = make_complex(std::sin(half), 0.0);
    return launch_single_qubit_matrix(handle, state, numQubits, targetQubit,
                                      c, minus_s, plus_s, c);
}

rocqStatus_t rocsvApplyRz(rocsvHandle_t handle,
                          rocComplex* d_state,
                          unsigned numQubits,
                          unsigned targetQubit,
                          double theta) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) return ROCQ_STATUS_INVALID_VALUE;

    const double half = theta / 2.0;
    const rocComplex phase_pos = make_complex(std::cos(half), std::sin(half));
    const rocComplex phase_neg = make_complex(std::cos(half), -std::sin(half));
    const rocComplex zero = make_complex(0.0, 0.0);
    return launch_single_qubit_matrix(handle, state, numQubits, targetQubit,
                                      phase_neg, zero, zero, phase_pos);
}

rocqStatus_t rocsvApplyP(rocsvHandle_t handle,
                         rocComplex* d_state,
                         unsigned numQubits,
                         unsigned targetQubit,
                         double theta) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) return ROCQ_STATUS_INVALID_VALUE;

    const rocComplex one = make_complex(1.0, 0.0);
    const rocComplex zero = make_complex(0.0, 0.0);
    const rocComplex phase = make_complex(std::cos(theta), std::sin(theta));
    return launch_single_qubit_matrix(handle, state, numQubits, targetQubit,
                                      one, zero, zero, phase);
}

rocqStatus_t rocsvApplyRxBatch(rocsvHandle_t handle,
                               rocComplex* d_state,
                               unsigned numQubits,
                               unsigned targetQubit,
                               const double* h_thetas,
                               size_t thetaCount) {
    return rocsvApplyRotationBatch(handle, d_state, numQubits, targetQubit, h_thetas, thetaCount, RotationAxis::X);
}

rocqStatus_t rocsvApplyRyBatch(rocsvHandle_t handle,
                               rocComplex* d_state,
                               unsigned numQubits,
                               unsigned targetQubit,
                               const double* h_thetas,
                               size_t thetaCount) {
    return rocsvApplyRotationBatch(handle, d_state, numQubits, targetQubit, h_thetas, thetaCount, RotationAxis::Y);
}

rocqStatus_t rocsvApplyRzBatch(rocsvHandle_t handle,
                               rocComplex* d_state,
                               unsigned numQubits,
                               unsigned targetQubit,
                               const double* h_thetas,
                               size_t thetaCount) {
    return rocsvApplyRotationBatch(handle, d_state, numQubits, targetQubit, h_thetas, thetaCount, RotationAxis::Z);
}

rocqStatus_t rocsvApplyPBatch(rocsvHandle_t handle,
                              rocComplex* d_state,
                              unsigned numQubits,
                              unsigned targetQubit,
                              const double* h_thetas,
                              size_t thetaCount) {
    if (!handle || !h_thetas || thetaCount == 0) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    const size_t batch_size = effective_batch_size(handle, state);
    if (thetaCount != batch_size) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    std::vector<rocComplex> matrices = make_phase_matrices(h_thetas, thetaCount);
    return launch_single_qubit_matrix_batch(handle, state, numQubits, targetQubit, matrices);
}

// --- Two-qubit named gates ---------------------------------------------------

rocqStatus_t rocsvApplyCNOT(rocsvHandle_t handle,
                            rocComplex* d_state,
                            unsigned numQubits,
                            unsigned controlQubit,
                            unsigned targetQubit) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) return ROCQ_STATUS_INVALID_VALUE;
    if (!validate_qubit_index(controlQubit, numQubits) ||
        !validate_qubit_index(targetQubit, numQubits) ||
        controlQubit == targetQubit) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    if (uses_distributed_state(handle, d_state)) {
        if (numQubits != handle->globalNumQubits) {
            return ROCQ_STATUS_INVALID_VALUE;
        }
        if (!distributed_qubit_local(handle, controlQubit) ||
            !distributed_qubit_local(handle, targetQubit)) {
            const rocComplex zero = make_complex(0.0, 0.0);
            const rocComplex one = make_complex(1.0, 0.0);
            return launch_controlled_single_qubit_matrix(handle,
                                                         state,
                                                         numQubits,
                                                         controlQubit,
                                                         targetQubit,
                                                         zero,
                                                         one,
                                                         one,
                                                         zero);
        }
        if (handle->localSliceElements < 4) {
            return ROCQ_STATUS_SUCCESS;
        }

        const size_t total_states = handle->localSliceElements;
        constexpr int threads_per_block = 256;
        const int blocks =
            static_cast<int>((total_states + threads_per_block - 1) / threads_per_block);
        for (int rank = 0; rank < handle->distributedGpuCount; ++rank) {
            if (hipSetDevice(handle->distributedDeviceIds[static_cast<size_t>(rank)]) != hipSuccess) {
                return ROCQ_STATUS_HIP_ERROR;
            }
            hipLaunchKernelGGL(apply_CNOT_kernel,
                               dim3(blocks > 0 ? blocks : 1),
                               dim3(threads_per_block),
                               0,
                               handle->distributedStreams[static_cast<size_t>(rank)],
                               handle->distributedSlices[static_cast<size_t>(rank)],
                               handle->numLocalQubitsPerGpu,
                               controlQubit,
                               targetQubit,
                               1);
            rocqStatus_t status = check_last_hip_error();
            if (status != ROCQ_STATUS_SUCCESS) {
                return status;
            }
        }
        return ROCQ_STATUS_SUCCESS;
    }

    const size_t elements_per_state = 1ULL << numQubits;
    if (elements_per_state < 4) {
        return ROCQ_STATUS_SUCCESS;
    }

    const size_t total_states = handle->batchSize * elements_per_state;
    constexpr int threads_per_block = 256;
    const int blocks = static_cast<int>((total_states + threads_per_block - 1) / threads_per_block);

    hipLaunchKernelGGL(apply_CNOT_kernel,
                       dim3(blocks > 0 ? blocks : 1),
                       dim3(threads_per_block),
                       0,
                       handle->streams[0],
                       state,
                       numQubits,
                       controlQubit,
                       targetQubit,
                       handle->batchSize);
    return check_last_hip_error();
}

rocqStatus_t rocsvApplyCZ(rocsvHandle_t handle,
                          rocComplex* d_state,
                          unsigned numQubits,
                          unsigned controlQubit,
                          unsigned targetQubit) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) return ROCQ_STATUS_INVALID_VALUE;
    if (!validate_qubit_index(controlQubit, numQubits) ||
        !validate_qubit_index(targetQubit, numQubits) ||
        controlQubit == targetQubit) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    if (uses_distributed_state(handle, d_state)) {
        if (numQubits != handle->globalNumQubits) {
            return ROCQ_STATUS_INVALID_VALUE;
        }
        if (!distributed_qubit_local(handle, controlQubit) ||
            !distributed_qubit_local(handle, targetQubit)) {
            const rocComplex one = make_complex(1.0, 0.0);
            const rocComplex zero = make_complex(0.0, 0.0);
            const rocComplex minus_one = make_complex(-1.0, 0.0);
            return launch_controlled_single_qubit_matrix(handle,
                                                         state,
                                                         numQubits,
                                                         controlQubit,
                                                         targetQubit,
                                                         one,
                                                         zero,
                                                         zero,
                                                         minus_one);
        }
        if (handle->localSliceElements < 4) {
            return ROCQ_STATUS_SUCCESS;
        }

        const size_t total_states = handle->localSliceElements;
        constexpr int threads_per_block = 256;
        const int blocks =
            static_cast<int>((total_states + threads_per_block - 1) / threads_per_block);
        for (int rank = 0; rank < handle->distributedGpuCount; ++rank) {
            if (hipSetDevice(handle->distributedDeviceIds[static_cast<size_t>(rank)]) != hipSuccess) {
                return ROCQ_STATUS_HIP_ERROR;
            }
            hipLaunchKernelGGL(apply_CZ_kernel,
                               dim3(blocks > 0 ? blocks : 1),
                               dim3(threads_per_block),
                               0,
                               handle->distributedStreams[static_cast<size_t>(rank)],
                               handle->distributedSlices[static_cast<size_t>(rank)],
                               handle->numLocalQubitsPerGpu,
                               controlQubit,
                               targetQubit,
                               1);
            rocqStatus_t status = check_last_hip_error();
            if (status != ROCQ_STATUS_SUCCESS) {
                return status;
            }
        }
        return ROCQ_STATUS_SUCCESS;
    }

    const size_t elements_per_state = 1ULL << numQubits;
    if (elements_per_state < 4) {
        return ROCQ_STATUS_SUCCESS;
    }

    const size_t total_states = handle->batchSize * elements_per_state;
    constexpr int threads_per_block = 256;
    const int blocks = static_cast<int>((total_states + threads_per_block - 1) / threads_per_block);

    hipLaunchKernelGGL(apply_CZ_kernel,
                       dim3(blocks > 0 ? blocks : 1),
                       dim3(threads_per_block),
                       0,
                       handle->streams[0],
                       state,
                       numQubits,
                       controlQubit,
                       targetQubit,
                       handle->batchSize);
    return check_last_hip_error();
}

rocqStatus_t rocsvApplySWAP(rocsvHandle_t handle,
                            rocComplex* d_state,
                            unsigned numQubits,
                            unsigned qubitA,
                            unsigned qubitB) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) return ROCQ_STATUS_INVALID_VALUE;
    if (!validate_qubit_index(qubitA, numQubits) ||
        !validate_qubit_index(qubitB, numQubits) ||
        qubitA == qubitB) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    if (uses_distributed_state(handle, d_state)) {
        if (numQubits != handle->globalNumQubits) {
            return ROCQ_STATUS_INVALID_VALUE;
        }
        return rocsvSwapIndexBits(handle, qubitA, qubitB);
    }

    const size_t elements_per_state = 1ULL << numQubits;
    if (elements_per_state < 4) {
        return ROCQ_STATUS_SUCCESS;
    }

    const size_t groups_per_state = elements_per_state >> 2;
    const size_t total_groups = handle->batchSize * groups_per_state;
    if (total_groups == 0) {
        return ROCQ_STATUS_SUCCESS;
    }

    constexpr int threads_per_block = 256;
    const int blocks = static_cast<int>((total_groups + threads_per_block - 1) / threads_per_block);

    hipLaunchKernelGGL(apply_SWAP_kernel,
                       dim3(blocks > 0 ? blocks : 1),
                       dim3(threads_per_block),
                       0,
                       handle->streams[0],
                       state,
                       numQubits,
                       qubitA,
                       qubitB,
                       handle->batchSize);
    return check_last_hip_error();
}

rocqStatus_t rocsvApplyCRX(rocsvHandle_t handle,
                           rocComplex* d_state,
                           unsigned numQubits,
                           unsigned controlQubit,
                           unsigned targetQubit,
                           double theta) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) return ROCQ_STATUS_INVALID_VALUE;

    const double half = theta / 2.0;
    const rocComplex c = make_complex(std::cos(half), 0.0);
    const rocComplex minus_i_s = make_complex(0.0, -std::sin(half));
    return launch_controlled_single_qubit_matrix(handle, state, numQubits, controlQubit, targetQubit,
                                                 c, minus_i_s, minus_i_s, c);
}

rocqStatus_t rocsvApplyCRY(rocsvHandle_t handle,
                           rocComplex* d_state,
                           unsigned numQubits,
                           unsigned controlQubit,
                           unsigned targetQubit,
                           double theta) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) return ROCQ_STATUS_INVALID_VALUE;

    const double half = theta / 2.0;
    const rocComplex c = make_complex(std::cos(half), 0.0);
    const rocComplex minus_s = make_complex(-std::sin(half), 0.0);
    const rocComplex plus_s = make_complex(std::sin(half), 0.0);
    return launch_controlled_single_qubit_matrix(handle, state, numQubits, controlQubit, targetQubit,
                                                 c, minus_s, plus_s, c);
}

rocqStatus_t rocsvApplyCRZ(rocsvHandle_t handle,
                           rocComplex* d_state,
                           unsigned numQubits,
                           unsigned controlQubit,
                           unsigned targetQubit,
                           double theta) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) return ROCQ_STATUS_INVALID_VALUE;

    const double half = theta / 2.0;
    const rocComplex phase_pos = make_complex(std::cos(half), std::sin(half));
    const rocComplex phase_neg = make_complex(std::cos(half), -std::sin(half));
    const rocComplex zero = make_complex(0.0, 0.0);
    return launch_controlled_single_qubit_matrix(handle, state, numQubits, controlQubit, targetQubit,
                                                 phase_neg, zero, zero, phase_pos);
}

rocqStatus_t rocsvApplyCP(rocsvHandle_t handle,
                          rocComplex* d_state,
                          unsigned numQubits,
                          unsigned controlQubit,
                          unsigned targetQubit,
                          double theta) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) return ROCQ_STATUS_INVALID_VALUE;

    const rocComplex one = make_complex(1.0, 0.0);
    const rocComplex zero = make_complex(0.0, 0.0);
    const rocComplex phase = make_complex(std::cos(theta), std::sin(theta));
    return launch_controlled_single_qubit_matrix(handle, state, numQubits, controlQubit, targetQubit,
                                                 one, zero, zero, phase);
}

inline rocqStatus_t rocsvApplyControlledRotationBatch(rocsvHandle_t handle,
                                                      rocComplex* d_state,
                                                      unsigned numQubits,
                                                      unsigned controlQubit,
                                                      unsigned targetQubit,
                                                      const double* h_thetas,
                                                      size_t thetaCount,
                                                      RotationAxis axis) {
    if (!handle || !h_thetas || thetaCount == 0) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    const size_t batch_size = effective_batch_size(handle, state);
    if (thetaCount != batch_size) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    std::vector<rocComplex> matrices = make_rotation_matrices(h_thetas, thetaCount, axis);
    return launch_controlled_single_qubit_matrix_batch(
        handle,
        state,
        numQubits,
        controlQubit,
        targetQubit,
        matrices);
}

rocqStatus_t rocsvApplyCRXBatch(rocsvHandle_t handle,
                                rocComplex* d_state,
                                unsigned numQubits,
                                unsigned controlQubit,
                                unsigned targetQubit,
                                const double* h_thetas,
                                size_t thetaCount) {
    return rocsvApplyControlledRotationBatch(
        handle, d_state, numQubits, controlQubit, targetQubit, h_thetas, thetaCount, RotationAxis::X);
}

rocqStatus_t rocsvApplyCRYBatch(rocsvHandle_t handle,
                                rocComplex* d_state,
                                unsigned numQubits,
                                unsigned controlQubit,
                                unsigned targetQubit,
                                const double* h_thetas,
                                size_t thetaCount) {
    return rocsvApplyControlledRotationBatch(
        handle, d_state, numQubits, controlQubit, targetQubit, h_thetas, thetaCount, RotationAxis::Y);
}

rocqStatus_t rocsvApplyCRZBatch(rocsvHandle_t handle,
                                rocComplex* d_state,
                                unsigned numQubits,
                                unsigned controlQubit,
                                unsigned targetQubit,
                                const double* h_thetas,
                                size_t thetaCount) {
    return rocsvApplyControlledRotationBatch(
        handle, d_state, numQubits, controlQubit, targetQubit, h_thetas, thetaCount, RotationAxis::Z);
}

rocqStatus_t rocsvApplyCPBatch(rocsvHandle_t handle,
                               rocComplex* d_state,
                               unsigned numQubits,
                               unsigned controlQubit,
                               unsigned targetQubit,
                               const double* h_thetas,
                               size_t thetaCount) {
    if (!handle || !h_thetas || thetaCount == 0) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    const size_t batch_size = effective_batch_size(handle, state);
    if (thetaCount != batch_size) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    std::vector<rocComplex> matrices = make_phase_matrices(h_thetas, thetaCount);
    return launch_controlled_single_qubit_matrix_batch(
        handle,
        state,
        numQubits,
        controlQubit,
        targetQubit,
        matrices);
}

// --- Multi-controlled gates --------------------------------------------------

rocqStatus_t rocsvApplyMultiControlledX(rocsvHandle_t handle,
                                        rocComplex* d_state,
                                        unsigned numQubits,
                                        const unsigned* controlQubits,
                                        unsigned numControlQubits,
                                        unsigned targetQubit) {
    if (!handle || !controlQubits || numControlQubits == 0) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) return ROCQ_STATUS_INVALID_VALUE;
    if (!validate_qubit_index(targetQubit, numQubits)) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (uses_distributed_state(handle, d_state)) {
        return ROCQ_STATUS_NOT_IMPLEMENTED;
    }
    if (numControlQubits > 63) {
        return ROCQ_STATUS_NOT_IMPLEMENTED; // Needs 64-bit mask extension.
    }

    unsigned long long mask = 0ULL;
    for (unsigned i = 0; i < numControlQubits; ++i) {
        unsigned ctrl = controlQubits[i];
        if (!validate_qubit_index(ctrl, numQubits) || ctrl == targetQubit) {
            return ROCQ_STATUS_INVALID_VALUE;
        }
        mask |= (1ULL << ctrl);
    }

    const size_t elements_per_state = 1ULL << numQubits;
    if (elements_per_state < 2) {
        return ROCQ_STATUS_SUCCESS;
    }

    const size_t total_states = handle->batchSize * elements_per_state;
    constexpr int threads_per_block = 256;
    const int blocks = static_cast<int>((total_states + threads_per_block - 1) / threads_per_block);

    hipLaunchKernelGGL(apply_multi_controlled_x_kernel,
                       dim3(blocks > 0 ? blocks : 1),
                       dim3(threads_per_block),
                       0,
                       handle->streams[0],
                       state,
                       numQubits,
                       mask,
                       targetQubit,
                       handle->batchSize);
    return check_last_hip_error();
}

rocqStatus_t rocsvApplyCSWAP(rocsvHandle_t handle,
                             rocComplex* d_state,
                             unsigned numQubits,
                             unsigned controlQubit,
                             unsigned targetQubit1,
                             unsigned targetQubit2) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) return ROCQ_STATUS_INVALID_VALUE;
    if (!validate_qubit_index(controlQubit, numQubits) ||
        !validate_qubit_index(targetQubit1, numQubits) ||
        !validate_qubit_index(targetQubit2, numQubits) ||
        controlQubit == targetQubit1 ||
        controlQubit == targetQubit2 ||
        targetQubit1 == targetQubit2) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (uses_distributed_state(handle, d_state)) {
        return ROCQ_STATUS_NOT_IMPLEMENTED;
    }

    const size_t elements_per_state = 1ULL << numQubits;
    if (elements_per_state < 8) {
        return ROCQ_STATUS_SUCCESS;
    }

    const size_t total_states = handle->batchSize * elements_per_state;
    constexpr int threads_per_block = 256;
    const int blocks = static_cast<int>((total_states + threads_per_block - 1) / threads_per_block);

    hipLaunchKernelGGL(apply_CSWAP_kernel,
                       dim3(blocks > 0 ? blocks : 1),
                       dim3(threads_per_block),
                       0,
                       handle->streams[0],
                       state,
                       numQubits,
                       controlQubit,
                       targetQubit1,
                       targetQubit2,
                       handle->batchSize);
    return check_last_hip_error();
}

// --- State-vector readback helpers ------------------------------------------

rocqStatus_t rocsvGetStateVectorFull(rocsvHandle_t handle,
                                     rocComplex* d_state,
                                     rocComplex* h_state) {
    if (!handle || !h_state) return ROCQ_STATUS_INVALID_VALUE;

    if (uses_distributed_state(handle, d_state)) {
        std::vector<rocComplex> host_full;
        rocqStatus_t status = gather_distributed_state_to_host(handle, &host_full);
        if (status != ROCQ_STATUS_SUCCESS) {
            return status;
        }
        if (!host_full.empty()) {
            std::memcpy(h_state, host_full.data(), host_full.size() * sizeof(rocComplex));
        }
        return ROCQ_STATUS_SUCCESS;
    }

    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) return ROCQ_STATUS_INVALID_VALUE;
    const size_t elements_per_state = 1ULL << handle->numQubits;
    const size_t total_elements = handle->batchSize * elements_per_state;
    if (hipMemcpy(h_state,
                  state,
                  total_elements * sizeof(rocComplex),
                  hipMemcpyDeviceToHost) != hipSuccess) {
        return ROCQ_STATUS_HIP_ERROR;
    }
    return ROCQ_STATUS_SUCCESS;
}

rocqStatus_t rocsvGetStateVectorSlice(rocsvHandle_t handle,
                                      rocComplex* d_state,
                                      rocComplex* h_state,
                                      unsigned batch_index) {
    if (!handle || !h_state) return ROCQ_STATUS_INVALID_VALUE;

    if (uses_distributed_state(handle, d_state)) {
        if (batch_index != 0) {
            return ROCQ_STATUS_INVALID_VALUE;
        }
        std::vector<rocComplex> host_full;
        rocqStatus_t status = gather_distributed_state_to_host(handle, &host_full);
        if (status != ROCQ_STATUS_SUCCESS) {
            return status;
        }
        if (!host_full.empty()) {
            std::memcpy(h_state, host_full.data(), host_full.size() * sizeof(rocComplex));
        }
        return ROCQ_STATUS_SUCCESS;
    }

    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) return ROCQ_STATUS_INVALID_VALUE;

    const size_t elements_per_state = 1ULL << handle->numQubits;
    if (batch_index >= handle->batchSize) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    const size_t offset = batch_index * elements_per_state;
    const rocComplex* slice_ptr = state + offset;
    if (hipMemcpy(h_state,
                  slice_ptr,
                  elements_per_state * sizeof(rocComplex),
                  hipMemcpyDeviceToHost) != hipSuccess) {
        return ROCQ_STATUS_HIP_ERROR;
    }
    return ROCQ_STATUS_SUCCESS;
}

rocqStatus_t rocsvApplyFusedSingleQubitMatrix(rocsvHandle_t handle,
                                              unsigned targetQubit,
                                              const rocComplex* d_fusedMatrix) {
    if (!handle || !d_fusedMatrix) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (!handle->d_state || handle->numQubits == 0) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    std::vector<rocComplex> matrix_host;
    rocqStatus_t status = copy_matrix_from_device(d_fusedMatrix, 4, handle->streams[0], &matrix_host);
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }

    return launch_single_qubit_matrix(handle,
                                      handle->d_state,
                                      handle->numQubits,
                                      targetQubit,
                                      matrix_host[0],
                                      matrix_host[1],
                                      matrix_host[2],
                                      matrix_host[3]);
}

rocqStatus_t rocsvApplyMatrixGetWorkspaceSize(rocsvHandle_t handle,
                                              unsigned numQubits,
                                              unsigned numTargetQubits,
                                              size_t* workspaceSizeBytes) {
    if (!handle || !workspaceSizeBytes || numQubits == 0 || numTargetQubits == 0 || numTargetQubits > numQubits) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    *workspaceSizeBytes = 0;
    return ROCQ_STATUS_SUCCESS;
}

rocqStatus_t rocsvApplyMatrix(rocsvHandle_t handle,
                              rocComplex* d_state,
                              unsigned numQubits,
                              const unsigned* qubitIndices,
                              unsigned numTargetQubits,
                              const rocComplex* matrixDevice,
                              unsigned matrixDim) {
    if (!handle || !matrixDevice || !qubitIndices || numTargetQubits == 0 || numTargetQubits > numQubits) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    std::vector<unsigned> targets;
    rocqStatus_t status = validate_unique_qubits(qubitIndices, numTargetQubits, numQubits, &targets);
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }

    size_t expected_matrix_dim = 0;
    if (!compute_power_of_two(numTargetQubits, &expected_matrix_dim) || matrixDim != expected_matrix_dim) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (expected_matrix_dim > 0 &&
        expected_matrix_dim > (std::numeric_limits<size_t>::max() / expected_matrix_dim)) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    const size_t matrix_elements = expected_matrix_dim * expected_matrix_dim;

    if (uses_distributed_state(handle, d_state)) {
        if (numQubits != handle->globalNumQubits) {
            return ROCQ_STATUS_INVALID_VALUE;
        }
        if (!distributed_all_qubits_local(handle, targets)) {
            if (!distributed_host_fallback_enabled()) {
                return ROCQ_STATUS_NOT_IMPLEMENTED;
            }
            if (hipSetDevice(handle->distributedDeviceIds[0]) != hipSuccess) {
                return ROCQ_STATUS_HIP_ERROR;
            }
            std::vector<rocComplex> matrix_host;
            status = copy_matrix_from_device(matrixDevice,
                                             matrix_elements,
                                             handle->streams[0],
                                             &matrix_host);
            if (status != ROCQ_STATUS_SUCCESS) {
                return status;
            }
            return apply_matrix_distributed_host_fallback(handle,
                                                          numQubits,
                                                          targets,
                                                          {},
                                                          matrix_host);
        }
        if (hipSetDevice(handle->distributedDeviceIds[0]) != hipSuccess) {
            return ROCQ_STATUS_HIP_ERROR;
        }

        if (numTargetQubits == 1) {
            std::vector<rocComplex> matrix_host;
            status = copy_matrix_from_device(matrixDevice, 4, handle->streams[0], &matrix_host);
            if (status != ROCQ_STATUS_SUCCESS) {
                return status;
            }
            return launch_single_qubit_matrix(handle,
                                              handle->d_state,
                                              numQubits,
                                              targets[0],
                                              matrix_host[0],
                                              matrix_host[1],
                                              matrix_host[2],
                                              matrix_host[3]);
        }

        if (numTargetQubits >= 2 && numTargetQubits <= 4) {
            std::vector<rocComplex> matrix_host;
            status = copy_matrix_from_device(matrixDevice,
                                             matrix_elements,
                                             handle->streams[0],
                                             &matrix_host);
            if (status != ROCQ_STATUS_SUCCESS) {
                return status;
            }

            constexpr int threads_per_block = 256;
            const size_t groups_per_slice = handle->localSliceElements >> numTargetQubits;
            const int blocks = compute_reduction_blocks(groups_per_slice, threads_per_block);
            for (int rank = 0; rank < handle->distributedGpuCount; ++rank) {
                const size_t rank_idx = static_cast<size_t>(rank);
                if (hipSetDevice(handle->distributedDeviceIds[rank_idx]) != hipSuccess) {
                    return ROCQ_STATUS_HIP_ERROR;
                }

                unsigned* d_targets_local = handle->distributedTargetScratch[rank_idx];
                rocComplex* d_matrix_local = handle->distributedMatrixScratch[rank_idx];
                if (!d_targets_local || !d_matrix_local) {
                    return ROCQ_STATUS_ALLOCATION_FAILED;
                }

                if (hipMemcpyAsync(d_targets_local,
                                   targets.data(),
                                   numTargetQubits * sizeof(unsigned),
                                   hipMemcpyHostToDevice,
                                   handle->distributedStreams[rank_idx]) != hipSuccess) {
                    return ROCQ_STATUS_HIP_ERROR;
                }

                if (hipMemcpyAsync(d_matrix_local,
                                   matrix_host.data(),
                                   matrix_elements * sizeof(rocComplex),
                                   hipMemcpyHostToDevice,
                                   handle->distributedStreams[rank_idx]) != hipSuccess) {
                    return ROCQ_STATUS_HIP_ERROR;
                }

                if (numTargetQubits == 2) {
                    hipLaunchKernelGGL(apply_two_qubit_generic_matrix_kernel,
                                       dim3(blocks),
                                       dim3(threads_per_block),
                                       0,
                                       handle->distributedStreams[rank_idx],
                                       handle->distributedSlices[rank_idx],
                                       handle->numLocalQubitsPerGpu,
                                       d_targets_local,
                                       d_matrix_local);
                } else if (numTargetQubits == 3) {
                    hipLaunchKernelGGL(apply_three_qubit_generic_matrix_kernel,
                                       dim3(blocks),
                                       dim3(threads_per_block),
                                       0,
                                       handle->distributedStreams[rank_idx],
                                       handle->distributedSlices[rank_idx],
                                       handle->numLocalQubitsPerGpu,
                                       d_targets_local,
                                       d_matrix_local);
                } else {
                    hipLaunchKernelGGL(apply_four_qubit_generic_matrix_kernel,
                                       dim3(blocks),
                                       dim3(threads_per_block),
                                       0,
                                       handle->distributedStreams[rank_idx],
                                       handle->distributedSlices[rank_idx],
                                       handle->numLocalQubitsPerGpu,
                                       d_targets_local,
                                       d_matrix_local);
                }

                rocqStatus_t launch_status = check_last_hip_error();
                if (launch_status != ROCQ_STATUS_SUCCESS) {
                    return launch_status;
                }
            }
            return ROCQ_STATUS_SUCCESS;
        }

        return ROCQ_STATUS_NOT_IMPLEMENTED;
    }

    const size_t batch_size = effective_batch_size(handle, state);
    size_t state_elements = 0;
    if (!compute_power_of_two(numQubits, &state_elements)) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    const bool using_internal_state = (state == handle->d_state);
    const bool kernel_fast_path_ok = using_internal_state || handle->batchSize == 1;

    if (numTargetQubits == 1 && kernel_fast_path_ok) {
        std::vector<rocComplex> matrix_host;
        status = copy_matrix_from_device(matrixDevice, 4, handle->streams[0], &matrix_host);
        if (status != ROCQ_STATUS_SUCCESS) {
            return status;
        }
        if (using_internal_state) {
            handle->numQubits = numQubits;
        }
        return launch_single_qubit_matrix(handle,
                                          state,
                                          numQubits,
                                          targets[0],
                                          matrix_host[0],
                                          matrix_host[1],
                                          matrix_host[2],
                                          matrix_host[3]);
    }

    if ((numTargetQubits == 2 || numTargetQubits == 3 || numTargetQubits == 4) && kernel_fast_path_ok) {
        void* d_targets = nullptr;
        status = device_malloc(handle, &d_targets, numTargetQubits * sizeof(unsigned));
        if (status != ROCQ_STATUS_SUCCESS) {
            return status;
        }
        status = copy_host_to_device(d_targets,
                                     targets.data(),
                                     numTargetQubits * sizeof(unsigned),
                                     handle->streams[0]);
        if (status != ROCQ_STATUS_SUCCESS) {
            device_free(handle, d_targets);
            return status;
        }

        const int threads_per_block = 256;
        const size_t groups_per_state = state_elements >> numTargetQubits;
        const int blocks = compute_reduction_blocks(groups_per_state, threads_per_block);

        for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            rocComplex* batch_state = state + batch_idx * state_elements;
            if (numTargetQubits == 2) {
                hipLaunchKernelGGL(apply_two_qubit_generic_matrix_kernel,
                                   dim3(blocks),
                                   dim3(threads_per_block),
                                   0,
                                   handle->streams[0],
                                   batch_state,
                                   numQubits,
                                   static_cast<const unsigned*>(d_targets),
                                   matrixDevice);
            } else if (numTargetQubits == 3) {
                hipLaunchKernelGGL(apply_three_qubit_generic_matrix_kernel,
                                   dim3(blocks),
                                   dim3(threads_per_block),
                                   0,
                                   handle->streams[0],
                                   batch_state,
                                   numQubits,
                                   static_cast<const unsigned*>(d_targets),
                                   matrixDevice);
            } else {
                hipLaunchKernelGGL(apply_four_qubit_generic_matrix_kernel,
                                   dim3(blocks),
                                   dim3(threads_per_block),
                                   0,
                                   handle->streams[0],
                                   batch_state,
                                   numQubits,
                                   static_cast<const unsigned*>(d_targets),
                                   matrixDevice);
            }
            status = check_last_hip_error();
            if (status != ROCQ_STATUS_SUCCESS) {
                device_free(handle, d_targets);
                return status;
            }
        }

        device_free(handle, d_targets);
        return status;
    }

    if (!allow_host_matrix_fallback()) {
        return ROCQ_STATUS_NOT_IMPLEMENTED;
    }

    std::vector<rocComplex> matrix_host;
    status = copy_matrix_from_device(matrixDevice, matrix_elements, handle->streams[0], &matrix_host);
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }

    return apply_matrix_host_impl(state,
                                  numQubits,
                                  targets,
                                  {},
                                  matrix_host,
                                  batch_size,
                                  state_elements,
                                  handle->streams[0]);
}

rocqStatus_t rocsvApplySparseMatrix(rocsvHandle_t handle,
                                    rocComplex* d_state,
                                    unsigned numQubits,
                                    const unsigned* targetQubits,
                                    unsigned numTargetQubits,
                                    const rocComplex* d_data,
                                    const size_t* d_indices,
                                    const size_t* d_indptr,
                                    size_t rows,
                                    size_t cols,
                                    size_t nnz) {
    if (!handle || !targetQubits || !d_indptr || numQubits == 0 ||
        numTargetQubits == 0 || numTargetQubits > numQubits ||
        rows == 0 || cols == 0 || rows != cols) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (nnz > 0 && (!d_data || !d_indices)) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    std::vector<unsigned> targets;
    rocqStatus_t status = validate_unique_qubits(targetQubits, numTargetQubits, numQubits, &targets);
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }

    size_t state_elements = 0;
    if (!compute_power_of_two(numQubits, &state_elements)) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    size_t local_dimension = 0;
    if (!compute_power_of_two(numTargetQubits, &local_dimension) ||
        rows != local_dimension || cols != local_dimension) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    if (uses_distributed_state(handle, d_state)) {
        return apply_sparse_matrix_distributed_local(handle,
                                                     numQubits,
                                                     targets,
                                                     d_data,
                                                     d_indices,
                                                     d_indptr,
                                                     rows,
                                                     cols,
                                                     nnz);
    }

    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    const size_t batch_size = effective_batch_size(handle, state);
    if (batch_size > 65535) {
        return ROCQ_STATUS_NOT_IMPLEMENTED;
    }
    if (batch_size > std::numeric_limits<size_t>::max() / state_elements) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    const size_t total_elements = batch_size * state_elements;

    void* d_targets_void = nullptr;
    status = device_malloc(handle, &d_targets_void, targets.size() * sizeof(unsigned));
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }
    unsigned* d_targets = static_cast<unsigned*>(d_targets_void);
    status = copy_host_to_device(d_targets,
                                 targets.data(),
                                 targets.size() * sizeof(unsigned),
                                 handle->streams[0]);
    if (status != ROCQ_STATUS_SUCCESS) {
        device_free(handle, d_targets);
        return status;
    }

    void* d_output_void = nullptr;
    status = device_malloc(handle, &d_output_void, total_elements * sizeof(rocComplex));
    if (status != ROCQ_STATUS_SUCCESS) {
        device_free(handle, d_targets);
        return status;
    }
    rocComplex* d_output = static_cast<rocComplex*>(d_output_void);

    constexpr int threads_per_block = 256;
    const int blocks = compute_reduction_blocks(state_elements, threads_per_block);
    hipLaunchKernelGGL(apply_sparse_matrix_kernel,
                       dim3(blocks, static_cast<unsigned>(batch_size)),
                       dim3(threads_per_block),
                       0,
                       handle->streams[0],
                       state,
                       d_output,
                       state_elements,
                       d_targets,
                       static_cast<unsigned>(targets.size()),
                       d_data,
                       d_indices,
                       d_indptr);
    status = check_last_hip_error();
    if (status != ROCQ_STATUS_SUCCESS) {
        device_free(handle, d_output);
        device_free(handle, d_targets);
        return status;
    }

    if (hipMemcpyAsync(state,
                       d_output,
                       total_elements * sizeof(rocComplex),
                       hipMemcpyDeviceToDevice,
                       handle->streams[0]) != hipSuccess) {
        device_free(handle, d_output);
        device_free(handle, d_targets);
        return ROCQ_STATUS_HIP_ERROR;
    }
    if (hipStreamSynchronize(handle->streams[0]) != hipSuccess) { // ROCQ_ASYNC_ALLOWED_SYNC
        device_free(handle, d_output);
        device_free(handle, d_targets);
        return ROCQ_STATUS_HIP_ERROR;
    }

    rocqStatus_t free_status = device_free(handle, d_output);
    rocqStatus_t targets_free_status = device_free(handle, d_targets);
    if (free_status != ROCQ_STATUS_SUCCESS) {
        return free_status;
    }
    return targets_free_status;
}

rocqStatus_t rocsvMeasure(rocsvHandle_t handle,
                          rocComplex* d_state,
                          unsigned numQubits,
                          unsigned qubitToMeasure,
                          int* outcome,
                          double* probability) {
    if (!handle || !outcome || !probability || !validate_qubit_index(qubitToMeasure, numQubits)) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    if (uses_distributed_state(handle, d_state)) {
        if (numQubits != handle->globalNumQubits || handle->distributedGpuCount <= 0) {
            return ROCQ_STATUS_INVALID_VALUE;
        }
        if (handle->localSliceElements == 0) {
            return ROCQ_STATUS_INVALID_VALUE;
        }

        const bool local_measure = distributed_qubit_local(handle, qubitToMeasure);
        double prob0 = 0.0;
        double prob1 = 0.0;
        rocqStatus_t status = ROCQ_STATUS_SUCCESS;

        if (local_measure) {
            for (int rank = 0; rank < handle->distributedGpuCount; ++rank) {
                if (hipSetDevice(handle->distributedDeviceIds[static_cast<size_t>(rank)]) != hipSuccess) {
                    return ROCQ_STATUS_HIP_ERROR;
                }
                double slice_prob0 = 0.0;
                status = reduce_prob0_on_slice(handle,
                                               handle->distributedSlices[static_cast<size_t>(rank)],
                                               handle->localSliceElements,
                                               qubitToMeasure,
                                               handle->distributedStreams[static_cast<size_t>(rank)],
                                               &slice_prob0);
                if (status != ROCQ_STATUS_SUCCESS) {
                    return status;
                }
                prob0 += slice_prob0;
            }
            prob0 = clamp_probability(prob0);
            prob1 = clamp_probability(1.0 - prob0);
        } else {
            const unsigned rank_bit = qubitToMeasure - handle->numLocalQubitsPerGpu;
            for (int rank = 0; rank < handle->distributedGpuCount; ++rank) {
                if (hipSetDevice(handle->distributedDeviceIds[static_cast<size_t>(rank)]) != hipSuccess) {
                    return ROCQ_STATUS_HIP_ERROR;
                }
                double slice_norm = 0.0;
                status = reduce_prob0_on_slice(handle,
                                               handle->distributedSlices[static_cast<size_t>(rank)],
                                               handle->localSliceElements,
                                               handle->numLocalQubitsPerGpu,
                                               handle->distributedStreams[static_cast<size_t>(rank)],
                                               &slice_norm);
                if (status != ROCQ_STATUS_SUCCESS) {
                    return status;
                }
                const int bit = (rank >> rank_bit) & 1;
                if (bit == 0) {
                    prob0 += slice_norm;
                } else {
                    prob1 += slice_norm;
                }
            }
            const double total = prob0 + prob1;
            if (total > 0.0) {
                prob0 /= total;
                prob1 /= total;
            }
            prob0 = clamp_probability(prob0);
            prob1 = clamp_probability(prob1);
        }

        std::uniform_real_distribution<double> dist(0.0, 1.0);
        const double random_value = dist(handle->rng);

        int measured = 0;
        if (is_effectively_zero(prob0)) {
            measured = 1;
        } else if (is_effectively_zero(prob1)) {
            measured = 0;
        } else {
            measured = (random_value < prob0) ? 0 : 1;
        }

        const double measured_prob = (measured == 0) ? prob0 : prob1;
        *outcome = measured;
        *probability = measured_prob;
        if (is_effectively_zero(measured_prob)) {
            return ROCQ_STATUS_SUCCESS;
        }

        const double inv_norm = 1.0 / std::sqrt(measured_prob);
        constexpr int threads_per_block = 256;
        const int blocks =
            compute_reduction_blocks(handle->localSliceElements, threads_per_block);
        const size_t bytes_per_slice = handle->localSliceElements * sizeof(rocComplex);

        if (local_measure) {
            for (int rank = 0; rank < handle->distributedGpuCount; ++rank) {
                if (hipSetDevice(handle->distributedDeviceIds[static_cast<size_t>(rank)]) != hipSuccess) {
                    return ROCQ_STATUS_HIP_ERROR;
                }
                hipLaunchKernelGGL(collapse_and_renorm_measure_kernel,
                                   dim3(blocks),
                                   dim3(threads_per_block),
                                   0,
                                   handle->distributedStreams[static_cast<size_t>(rank)],
                                   handle->distributedSlices[static_cast<size_t>(rank)],
                                   handle->localSliceElements,
                                   qubitToMeasure,
                                   measured,
                                   inv_norm);
                status = check_last_hip_error();
                if (status != ROCQ_STATUS_SUCCESS) {
                    return status;
                }
            }
        } else {
            const unsigned rank_bit = qubitToMeasure - handle->numLocalQubitsPerGpu;
            for (int rank = 0; rank < handle->distributedGpuCount; ++rank) {
                if (hipSetDevice(handle->distributedDeviceIds[static_cast<size_t>(rank)]) != hipSuccess) {
                    return ROCQ_STATUS_HIP_ERROR;
                }
                const int bit = (rank >> rank_bit) & 1;
                if (bit != measured) {
                    if (hipMemsetAsync(handle->distributedSlices[static_cast<size_t>(rank)],
                                       0,
                                       bytes_per_slice,
                                       handle->distributedStreams[static_cast<size_t>(rank)]) != hipSuccess) {
                        return ROCQ_STATUS_HIP_ERROR;
                    }
                } else {
                    hipLaunchKernelGGL(renormalize_state_kernel,
                                       dim3(blocks),
                                       dim3(threads_per_block),
                                       0,
                                       handle->distributedStreams[static_cast<size_t>(rank)],
                                       handle->distributedSlices[static_cast<size_t>(rank)],
                                       handle->numLocalQubitsPerGpu,
                                       static_cast<real_t>(inv_norm));
                    status = check_last_hip_error();
                    if (status != ROCQ_STATUS_SUCCESS) {
                        return status;
                    }
                }
            }
        }
        return sync_distributed_streams(handle); // ROCQ_ASYNC_ALLOWED_SYNC
    }

    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (effective_batch_size(handle, state) != 1) {
        return ROCQ_STATUS_NOT_IMPLEMENTED;
    }

    size_t state_elements = 0;
    if (!compute_power_of_two(numQubits, &state_elements)) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    constexpr int threads_per_block = 256;
    const int blocks = compute_reduction_blocks(state_elements, threads_per_block);

    void* d_block_sums_void = nullptr;
    rocqStatus_t status = device_malloc(handle, &d_block_sums_void, static_cast<size_t>(blocks) * sizeof(double));
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }
    double* d_block_sums = static_cast<double*>(d_block_sums_void);

    hipLaunchKernelGGL(reduce_measure_prob0_kernel,
                       dim3(blocks),
                       dim3(threads_per_block),
                       threads_per_block * sizeof(double),
                       handle->streams[0],
                       state,
                       state_elements,
                       qubitToMeasure,
                       d_block_sums);
    status = check_last_hip_error();
    if (status != ROCQ_STATUS_SUCCESS) {
        device_free(handle, d_block_sums);
        return status;
    }

    double prob0 = 0.0;
    status = reduce_blocks_to_scalar(handle, d_block_sums, blocks, &prob0);
    device_free(handle, d_block_sums);
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }

    prob0 = clamp_probability(prob0);
    const double prob1 = clamp_probability(1.0 - prob0);

    std::uniform_real_distribution<double> dist(0.0, 1.0);
    const double random_value = dist(handle->rng);

    int measured = 0;
    if (is_effectively_zero(prob0)) {
        measured = 1;
    } else if (is_effectively_zero(prob1)) {
        measured = 0;
    } else {
        measured = (random_value < prob0) ? 0 : 1;
    }

    const double measured_prob = (measured == 0) ? prob0 : prob1;
    *outcome = measured;
    *probability = measured_prob;

    if (is_effectively_zero(measured_prob)) {
        return ROCQ_STATUS_SUCCESS;
    }

    const double inv_norm = 1.0 / std::sqrt(measured_prob);
    hipLaunchKernelGGL(collapse_and_renorm_measure_kernel,
                       dim3(blocks),
                       dim3(threads_per_block),
                       0,
                       handle->streams[0],
                       state,
                       state_elements,
                       qubitToMeasure,
                       measured,
                       inv_norm);
    return check_last_hip_error();
}

rocqStatus_t rocsvGetExpectationWorkspaceSize(rocsvHandle_t handle,
                                              unsigned numQubits,
                                              size_t* workspaceSizeBytes) {
    if (!handle || !workspaceSizeBytes || numQubits == 0) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    *workspaceSizeBytes = 0;
    return ROCQ_STATUS_SUCCESS;
}

rocqStatus_t rocsvEnsurePinnedBuffer(rocsvHandle_t handle, size_t minSizeBytes) {
    if (!handle) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (minSizeBytes == 0) {
        return ROCQ_STATUS_SUCCESS;
    }
    if (handle->pinnedBuffer && handle->pinnedBufferBytes >= minSizeBytes) {
        return ROCQ_STATUS_SUCCESS;
    }

    if (handle->pinnedBuffer) {
        if (hipHostFree(handle->pinnedBuffer) != hipSuccess) {
            return ROCQ_STATUS_HIP_ERROR;
        }
        handle->pinnedBuffer = nullptr;
        handle->pinnedBufferBytes = 0;
    }

    void* pinned = nullptr;
    if (hipHostMalloc(&pinned, minSizeBytes) != hipSuccess) {
        return ROCQ_STATUS_ALLOCATION_FAILED;
    }
    handle->pinnedBuffer = pinned;
    handle->pinnedBufferBytes = minSizeBytes;
    return ROCQ_STATUS_SUCCESS;
}

void* rocsvGetPinnedBufferPointer(rocsvHandle_t handle) {
    if (!handle) {
        return nullptr;
    }
    return handle->pinnedBuffer;
}

rocqStatus_t rocsvFreePinnedBuffer(rocsvHandle_t handle) {
    if (!handle) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (!handle->pinnedBuffer) {
        return ROCQ_STATUS_SUCCESS;
    }
    if (hipHostFree(handle->pinnedBuffer) != hipSuccess) {
        return ROCQ_STATUS_HIP_ERROR;
    }
    handle->pinnedBuffer = nullptr;
    handle->pinnedBufferBytes = 0;
    return ROCQ_STATUS_SUCCESS;
}

rocqStatus_t rocsvGetExpectationValueSinglePauliZ(rocsvHandle_t handle,
                                                  rocComplex* d_state,
                                                  unsigned numQubits,
                                                  unsigned targetQubit,
                                                  double* result) {
    if (!handle || !result || !validate_qubit_index(targetQubit, numQubits)) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (uses_distributed_state(handle, d_state)) {
        return distributed_expectation_with_fallback(handle,
                                                     numQubits,
                                                     std::vector<unsigned>{targetQubit},
                                                     DistributedExpectationKind::SingleZ,
                                                     "",
                                                     result);
    }
    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (effective_batch_size(handle, state) != 1) {
        return ROCQ_STATUS_NOT_IMPLEMENTED;
    }

    size_t state_elements = 0;
    if (!compute_power_of_two(numQubits, &state_elements)) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    constexpr int threads_per_block = 256;
    const int blocks = compute_reduction_blocks(state_elements, threads_per_block);
    void* d_block_sums_void = nullptr;
    rocqStatus_t status = device_malloc(handle, &d_block_sums_void, static_cast<size_t>(blocks) * sizeof(double));
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }
    double* d_block_sums = static_cast<double*>(d_block_sums_void);

    hipLaunchKernelGGL(reduce_expectation_z_kernel,
                       dim3(blocks),
                       dim3(threads_per_block),
                       threads_per_block * sizeof(double),
                       handle->streams[0],
                       state,
                       state_elements,
                       targetQubit,
                       d_block_sums);
    status = check_last_hip_error();
    if (status != ROCQ_STATUS_SUCCESS) {
        device_free(handle, d_block_sums);
        return status;
    }

    status = reduce_blocks_to_scalar(handle, d_block_sums, blocks, result);
    device_free(handle, d_block_sums);
    return status;
}

rocqStatus_t rocsvGetExpectationValueSinglePauliX(rocsvHandle_t handle,
                                                  rocComplex* d_state,
                                                  unsigned numQubits,
                                                  unsigned targetQubit,
                                                  double* result) {
    if (!handle || !result || !validate_qubit_index(targetQubit, numQubits)) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (uses_distributed_state(handle, d_state)) {
        return distributed_expectation_with_fallback(handle,
                                                     numQubits,
                                                     std::vector<unsigned>{targetQubit},
                                                     DistributedExpectationKind::SingleX,
                                                     "",
                                                     result);
    }
    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (effective_batch_size(handle, state) != 1) {
        return ROCQ_STATUS_NOT_IMPLEMENTED;
    }

    size_t state_elements = 0;
    if (!compute_power_of_two(numQubits, &state_elements)) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    constexpr int threads_per_block = 256;
    const int blocks = compute_reduction_blocks(state_elements, threads_per_block);
    void* d_block_sums_void = nullptr;
    rocqStatus_t status = device_malloc(handle, &d_block_sums_void, static_cast<size_t>(blocks) * sizeof(double));
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }
    double* d_block_sums = static_cast<double*>(d_block_sums_void);

    hipLaunchKernelGGL(reduce_expectation_x_kernel,
                       dim3(blocks),
                       dim3(threads_per_block),
                       threads_per_block * sizeof(double),
                       handle->streams[0],
                       state,
                       state_elements,
                       targetQubit,
                       d_block_sums);
    status = check_last_hip_error();
    if (status != ROCQ_STATUS_SUCCESS) {
        device_free(handle, d_block_sums);
        return status;
    }

    status = reduce_blocks_to_scalar(handle, d_block_sums, blocks, result);
    device_free(handle, d_block_sums);
    return status;
}

rocqStatus_t rocsvGetExpectationValueSinglePauliY(rocsvHandle_t handle,
                                                  rocComplex* d_state,
                                                  unsigned numQubits,
                                                  unsigned targetQubit,
                                                  double* result) {
    if (!handle || !result || !validate_qubit_index(targetQubit, numQubits)) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (uses_distributed_state(handle, d_state)) {
        return distributed_expectation_with_fallback(handle,
                                                     numQubits,
                                                     std::vector<unsigned>{targetQubit},
                                                     DistributedExpectationKind::SingleY,
                                                     "",
                                                     result);
    }
    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (effective_batch_size(handle, state) != 1) {
        return ROCQ_STATUS_NOT_IMPLEMENTED;
    }

    size_t state_elements = 0;
    if (!compute_power_of_two(numQubits, &state_elements)) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    constexpr int threads_per_block = 256;
    const int blocks = compute_reduction_blocks(state_elements, threads_per_block);
    void* d_block_sums_void = nullptr;
    rocqStatus_t status = device_malloc(handle, &d_block_sums_void, static_cast<size_t>(blocks) * sizeof(double));
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }
    double* d_block_sums = static_cast<double*>(d_block_sums_void);

    hipLaunchKernelGGL(reduce_expectation_y_kernel,
                       dim3(blocks),
                       dim3(threads_per_block),
                       threads_per_block * sizeof(double),
                       handle->streams[0],
                       state,
                       state_elements,
                       targetQubit,
                       d_block_sums);
    status = check_last_hip_error();
    if (status != ROCQ_STATUS_SUCCESS) {
        device_free(handle, d_block_sums);
        return status;
    }

    status = reduce_blocks_to_scalar(handle, d_block_sums, blocks, result);
    device_free(handle, d_block_sums);
    return status;
}

rocqStatus_t rocsvGetExpectationValuePauliProductZ(rocsvHandle_t handle,
                                                   rocComplex* d_state,
                                                   unsigned numQubits,
                                                   const unsigned* targetQubits,
                                                   unsigned numTargetPaulis,
                                                   double* result) {
    if (!handle || !result) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (numTargetPaulis == 0) {
        *result = 1.0;
        return ROCQ_STATUS_SUCCESS;
    }

    std::vector<unsigned> targets;
    rocqStatus_t status = validate_unique_qubits(targetQubits, numTargetPaulis, numQubits, &targets);
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }
    if (uses_distributed_state(handle, d_state)) {
        return distributed_expectation_with_fallback(handle,
                                                     numQubits,
                                                     targets,
                                                     DistributedExpectationKind::ZProduct,
                                                     "",
                                                     result);
    }

    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (effective_batch_size(handle, state) != 1) {
        return ROCQ_STATUS_NOT_IMPLEMENTED;
    }

    size_t state_elements = 0;
    if (!compute_power_of_two(numQubits, &state_elements)) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    void* d_targets_void = nullptr;
    status = device_malloc(handle, &d_targets_void, targets.size() * sizeof(unsigned));
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }
    unsigned* d_targets = static_cast<unsigned*>(d_targets_void);
    status = copy_host_to_device(d_targets,
                                 targets.data(),
                                 targets.size() * sizeof(unsigned),
                                 handle->streams[0]);
    if (status != ROCQ_STATUS_SUCCESS) {
        device_free(handle, d_targets);
        return status;
    }

    constexpr int threads_per_block = 256;
    const int blocks = compute_reduction_blocks(state_elements, threads_per_block);
    void* d_block_sums_void = nullptr;
    status = device_malloc(handle, &d_block_sums_void, static_cast<size_t>(blocks) * sizeof(double));
    if (status != ROCQ_STATUS_SUCCESS) {
        device_free(handle, d_targets);
        return status;
    }
    double* d_block_sums = static_cast<double*>(d_block_sums_void);

    hipLaunchKernelGGL(reduce_expectation_z_product_kernel,
                       dim3(blocks),
                       dim3(threads_per_block),
                       threads_per_block * sizeof(double),
                       handle->streams[0],
                       state,
                       state_elements,
                       d_targets,
                       static_cast<unsigned>(targets.size()),
                       d_block_sums);
    status = check_last_hip_error();
    if (status != ROCQ_STATUS_SUCCESS) {
        device_free(handle, d_block_sums);
        device_free(handle, d_targets);
        return status;
    }

    status = reduce_blocks_to_scalar(handle, d_block_sums, blocks, result);
    device_free(handle, d_block_sums);
    device_free(handle, d_targets);
    return status;
}

rocqStatus_t rocsvGetExpectationPauliString(rocsvHandle_t handle,
                                            rocComplex* d_state,
                                            unsigned numQubits,
                                            const char* pauliString,
                                            const unsigned* targetQubits,
                                            unsigned numTargetPaulis,
                                            double* result) {
    if (!handle || !pauliString || !result) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (numTargetPaulis == 0) {
        *result = 1.0;
        return ROCQ_STATUS_SUCCESS;
    }

    std::vector<unsigned> targets;
    rocqStatus_t status = validate_unique_qubits(targetQubits, numTargetPaulis, numQubits, &targets);
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }

    const std::string pauli(pauliString);
    if (pauli.size() != numTargetPaulis) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    for (char c : pauli) {
        const char u = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        if (u != 'I' && u != 'X' && u != 'Y' && u != 'Z') {
            return ROCQ_STATUS_INVALID_VALUE;
        }
    }
    if (uses_distributed_state(handle, d_state)) {
        return distributed_expectation_with_fallback(handle,
                                                     numQubits,
                                                     targets,
                                                     DistributedExpectationKind::PauliString,
                                                     pauli,
                                                     result);
    }

    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (effective_batch_size(handle, state) != 1) {
        return ROCQ_STATUS_NOT_IMPLEMENTED;
    }

    size_t state_elements = 0;
    if (!compute_power_of_two(numQubits, &state_elements)) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    void* d_targets_void = nullptr;
    status = device_malloc(handle, &d_targets_void, targets.size() * sizeof(unsigned));
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }
    unsigned* d_targets = static_cast<unsigned*>(d_targets_void);

    status = copy_host_to_device(d_targets,
                                 targets.data(),
                                 targets.size() * sizeof(unsigned),
                                 handle->streams[0]);
    if (status != ROCQ_STATUS_SUCCESS) {
        device_free(handle, d_targets);
        return status;
    }

    void* d_pauli_void = nullptr;
    status = device_malloc(handle, &d_pauli_void, pauli.size() * sizeof(char));
    if (status != ROCQ_STATUS_SUCCESS) {
        device_free(handle, d_targets);
        return status;
    }
    char* d_pauli = static_cast<char*>(d_pauli_void);

    status = copy_host_to_device(d_pauli,
                                 pauli.data(),
                                 pauli.size() * sizeof(char),
                                 handle->streams[0]);
    if (status != ROCQ_STATUS_SUCCESS) {
        device_free(handle, d_pauli);
        device_free(handle, d_targets);
        return status;
    }

    constexpr int threads_per_block = 256;
    const int blocks = compute_reduction_blocks(state_elements, threads_per_block);
    void* d_block_sums_void = nullptr;
    status = device_malloc(handle, &d_block_sums_void, static_cast<size_t>(blocks) * sizeof(double));
    if (status != ROCQ_STATUS_SUCCESS) {
        device_free(handle, d_pauli);
        device_free(handle, d_targets);
        return status;
    }
    double* d_block_sums = static_cast<double*>(d_block_sums_void);

    hipLaunchKernelGGL(reduce_expectation_pauli_string_kernel,
                       dim3(blocks),
                       dim3(threads_per_block),
                       threads_per_block * sizeof(double),
                       handle->streams[0],
                       state,
                       state_elements,
                       d_pauli,
                       d_targets,
                       static_cast<unsigned>(targets.size()),
                       d_block_sums);
    status = check_last_hip_error();
    if (status != ROCQ_STATUS_SUCCESS) {
        device_free(handle, d_block_sums);
        device_free(handle, d_pauli);
        device_free(handle, d_targets);
        return status;
    }

    status = reduce_blocks_to_scalar(handle, d_block_sums, blocks, result);
    device_free(handle, d_block_sums);
    device_free(handle, d_pauli);
    device_free(handle, d_targets);
    return status;
}

rocqStatus_t rocsvGetExpectationPauliStringBatch(rocsvHandle_t handle,
                                                 rocComplex* d_state,
                                                 unsigned numQubits,
                                                 const char* pauliString,
                                                 const unsigned* targetQubits,
                                                 unsigned numTargetPaulis,
                                                 double* results) {
    if (!handle || !pauliString || !results) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (numTargetPaulis == 0) {
        const size_t batch_size = effective_batch_size(handle, state);
        std::fill_n(results, batch_size, 1.0);
        return ROCQ_STATUS_SUCCESS;
    }

    std::vector<unsigned> targets;
    rocqStatus_t status = validate_unique_qubits(targetQubits, numTargetPaulis, numQubits, &targets);
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }

    const std::string pauli(pauliString);
    if (pauli.size() != numTargetPaulis) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    for (char c : pauli) {
        const char u = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        if (u != 'I' && u != 'X' && u != 'Y' && u != 'Z') {
            return ROCQ_STATUS_INVALID_VALUE;
        }
    }

    if (uses_distributed_state(handle, d_state)) {
        return distributed_expectation_with_fallback(handle,
                                                     numQubits,
                                                     targets,
                                                     DistributedExpectationKind::PauliString,
                                                     pauli,
                                                     results);
    }

    if (!state) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    return compute_local_expectation_pauli_string_batch(handle, state, numQubits, targets, pauli, results);
}

rocqStatus_t rocsvGetExpectationMatrix(rocsvHandle_t handle,
                                       rocComplex* d_state,
                                       unsigned numQubits,
                                       const unsigned* targetQubits,
                                       unsigned numTargetQubits,
                                       const rocComplex* d_matrix,
                                       size_t matrixDim,
                                       rocComplex* result) {
    if (!handle || !targetQubits || numTargetQubits == 0 || !d_matrix || !result) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    size_t expected_matrix_dim = 0;
    if (!compute_power_of_two(numTargetQubits, &expected_matrix_dim) ||
        matrixDim != expected_matrix_dim) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    std::vector<unsigned> targets;
    rocqStatus_t status = validate_unique_qubits(targetQubits, numTargetQubits, numQubits, &targets);
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }

    size_t state_elements = 0;
    if (!compute_power_of_two(numQubits, &state_elements)) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    if (uses_distributed_state(handle, d_state)) {
        return expectation_matrix_distributed_host_fallback(handle,
                                                            numQubits,
                                                            targets,
                                                            d_matrix,
                                                            matrixDim,
                                                            result);
    }

    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (effective_batch_size(handle, state) != 1) {
        return ROCQ_STATUS_NOT_IMPLEMENTED;
    }
    if (numTargetQubits > 4) {
        return expectation_matrix_host_fallback(handle,
                                                state,
                                                numQubits,
                                                targets,
                                                d_matrix,
                                                matrixDim,
                                                result);
    }

    void* d_targets_void = nullptr;
    status = device_malloc(handle, &d_targets_void, targets.size() * sizeof(unsigned));
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }
    unsigned* d_targets = static_cast<unsigned*>(d_targets_void);
    status = copy_host_to_device(d_targets,
                                 targets.data(),
                                 targets.size() * sizeof(unsigned),
                                 handle->streams[0]);
    if (status != ROCQ_STATUS_SUCCESS) {
        device_free(handle, d_targets);
        return status;
    }

    constexpr int threads_per_block = 256;
    const int blocks = compute_reduction_blocks(state_elements, threads_per_block);
    void* d_block_sums_void = nullptr;
    status = device_malloc(handle,
                           &d_block_sums_void,
                           static_cast<size_t>(blocks) * sizeof(rocComplex));
    if (status != ROCQ_STATUS_SUCCESS) {
        device_free(handle, d_targets);
        return status;
    }
    rocComplex* d_block_sums = static_cast<rocComplex*>(d_block_sums_void);

    hipLaunchKernelGGL(reduce_expectation_matrix_kernel,
                       dim3(blocks),
                       dim3(threads_per_block),
                       2 * threads_per_block * sizeof(double),
                       handle->streams[0],
                       state,
                       state_elements,
                       d_targets,
                       static_cast<unsigned>(targets.size()),
                       d_matrix,
                       matrixDim,
                       d_block_sums);
    status = check_last_hip_error();
    if (status != ROCQ_STATUS_SUCCESS) {
        device_free(handle, d_block_sums);
        device_free(handle, d_targets);
        return status;
    }

    std::vector<rocComplex> block_sums(static_cast<size_t>(blocks));
    status = copy_device_to_host(block_sums.data(),
                                 d_block_sums,
                                 block_sums.size() * sizeof(rocComplex),
                                 handle->streams[0]);
    device_free(handle, d_block_sums);
    device_free(handle, d_targets);
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }

    double total_re = 0.0;
    double total_im = 0.0;
    for (const rocComplex& block_sum : block_sums) {
        total_re += static_cast<double>(block_sum.x);
        total_im += static_cast<double>(block_sum.y);
    }
    *result = make_complex(total_re, total_im);
    return ROCQ_STATUS_SUCCESS;
}

rocqStatus_t rocsvGetExpectationMatrixBatch(rocsvHandle_t handle,
                                            rocComplex* d_state,
                                            unsigned numQubits,
                                            const unsigned* targetQubits,
                                            unsigned numTargetQubits,
                                            const rocComplex* d_matrix,
                                            size_t matrixDim,
                                            rocComplex* results) {
    if (!handle || !targetQubits || numTargetQubits == 0 || !d_matrix || !results) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    size_t expected_matrix_dim = 0;
    if (!compute_power_of_two(numTargetQubits, &expected_matrix_dim) ||
        matrixDim != expected_matrix_dim) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    std::vector<unsigned> targets;
    rocqStatus_t status = validate_unique_qubits(targetQubits, numTargetQubits, numQubits, &targets);
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }
    if (uses_distributed_state(handle, d_state)) {
        return ROCQ_STATUS_NOT_IMPLEMENTED;
    }

    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    const size_t batch_size = effective_batch_size(handle, state);
    if (batch_size > 65535) {
        return ROCQ_STATUS_NOT_IMPLEMENTED;
    }
    if (numTargetQubits > 4) {
        return expectation_matrix_batch_host_fallback(handle,
                                                      state,
                                                      numQubits,
                                                      targets,
                                                      d_matrix,
                                                      matrixDim,
                                                      results);
    }

    size_t state_elements = 0;
    if (!compute_power_of_two(numQubits, &state_elements)) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    void* d_targets_void = nullptr;
    status = device_malloc(handle, &d_targets_void, targets.size() * sizeof(unsigned));
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }
    unsigned* d_targets = static_cast<unsigned*>(d_targets_void);
    status = copy_host_to_device(d_targets,
                                 targets.data(),
                                 targets.size() * sizeof(unsigned),
                                 handle->streams[0]);
    if (status != ROCQ_STATUS_SUCCESS) {
        device_free(handle, d_targets);
        return status;
    }

    constexpr int threads_per_block = 256;
    const int blocks = compute_reduction_blocks(state_elements, threads_per_block);
    if (batch_size > std::numeric_limits<size_t>::max() / static_cast<size_t>(blocks)) {
        device_free(handle, d_targets);
        return ROCQ_STATUS_INVALID_VALUE;
    }
    const size_t total_blocks = batch_size * static_cast<size_t>(blocks);

    void* d_block_sums_void = nullptr;
    status = device_malloc(handle, &d_block_sums_void, total_blocks * sizeof(rocComplex));
    if (status != ROCQ_STATUS_SUCCESS) {
        device_free(handle, d_targets);
        return status;
    }
    rocComplex* d_block_sums = static_cast<rocComplex*>(d_block_sums_void);

    hipLaunchKernelGGL(reduce_expectation_matrix_kernel,
                       dim3(blocks, static_cast<unsigned>(batch_size)),
                       dim3(threads_per_block),
                       2 * threads_per_block * sizeof(double),
                       handle->streams[0],
                       state,
                       state_elements,
                       d_targets,
                       static_cast<unsigned>(targets.size()),
                       d_matrix,
                       matrixDim,
                       d_block_sums);
    status = check_last_hip_error();
    if (status != ROCQ_STATUS_SUCCESS) {
        device_free(handle, d_block_sums);
        device_free(handle, d_targets);
        return status;
    }

    std::vector<rocComplex> block_sums(total_blocks);
    status = copy_device_to_host(block_sums.data(),
                                 d_block_sums,
                                 block_sums.size() * sizeof(rocComplex),
                                 handle->streams[0]);
    device_free(handle, d_block_sums);
    device_free(handle, d_targets);
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }

    for (size_t batch = 0; batch < batch_size; ++batch) {
        double total_re = 0.0;
        double total_im = 0.0;
        const size_t offset = batch * static_cast<size_t>(blocks);
        for (int block = 0; block < blocks; ++block) {
            const rocComplex& block_sum = block_sums[offset + static_cast<size_t>(block)];
            total_re += static_cast<double>(block_sum.x);
            total_im += static_cast<double>(block_sum.y);
        }
        results[batch] = make_complex(total_re, total_im);
    }
    return ROCQ_STATUS_SUCCESS;
}

rocqStatus_t rocsvGetSparseMatrixMoments(rocsvHandle_t handle,
                                         rocComplex* d_state,
                                         unsigned numQubits,
                                         const rocComplex* d_data,
                                         const size_t* d_indices,
                                         const size_t* d_indptr,
                                         size_t rows,
                                         size_t cols,
                                         size_t nnz,
                                         rocComplex* mean,
                                         rocComplex* secondMoment) {
    if (!handle || !d_indptr || !mean || !secondMoment || rows == 0 || cols == 0) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (rows != cols) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (nnz > 0 && (!d_data || !d_indices)) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    size_t state_elements = 0;
    if (!compute_power_of_two(numQubits, &state_elements)) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (rows != state_elements || cols != state_elements) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    if (uses_distributed_state(handle, d_state)) {
        return sparse_matrix_moments_distributed_host_fallback(handle,
                                                               numQubits,
                                                               d_data,
                                                               d_indices,
                                                               d_indptr,
                                                               rows,
                                                               cols,
                                                               nnz,
                                                               mean,
                                                               secondMoment);
    }

    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (effective_batch_size(handle, state) != 1) {
        return ROCQ_STATUS_NOT_IMPLEMENTED;
    }

    constexpr int threads_per_block = 256;
    const int blocks = compute_reduction_blocks(rows, threads_per_block);

    void* d_mean_block_sums_void = nullptr;
    rocqStatus_t status = device_malloc(handle,
                                        &d_mean_block_sums_void,
                                        static_cast<size_t>(blocks) * sizeof(rocComplex));
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }
    rocComplex* d_mean_block_sums = static_cast<rocComplex*>(d_mean_block_sums_void);

    void* d_second_block_sums_void = nullptr;
    status = device_malloc(handle,
                           &d_second_block_sums_void,
                           static_cast<size_t>(blocks) * sizeof(rocComplex));
    if (status != ROCQ_STATUS_SUCCESS) {
        device_free(handle, d_mean_block_sums);
        return status;
    }
    rocComplex* d_second_block_sums = static_cast<rocComplex*>(d_second_block_sums_void);

    hipLaunchKernelGGL(reduce_sparse_matrix_moments_kernel,
                       dim3(blocks),
                       dim3(threads_per_block),
                       4 * threads_per_block * sizeof(double),
                       handle->streams[0],
                       state,
                       d_data,
                       d_indices,
                       d_indptr,
                       rows,
                       d_mean_block_sums,
                       d_second_block_sums);
    status = check_last_hip_error();
    if (status != ROCQ_STATUS_SUCCESS) {
        device_free(handle, d_second_block_sums);
        device_free(handle, d_mean_block_sums);
        return status;
    }

    std::vector<rocComplex> mean_block_sums(static_cast<size_t>(blocks));
    status = copy_device_to_host(mean_block_sums.data(),
                                 d_mean_block_sums,
                                 mean_block_sums.size() * sizeof(rocComplex),
                                 handle->streams[0]);
    if (status != ROCQ_STATUS_SUCCESS) {
        device_free(handle, d_second_block_sums);
        device_free(handle, d_mean_block_sums);
        return status;
    }

    std::vector<rocComplex> second_block_sums(static_cast<size_t>(blocks));
    status = copy_device_to_host(second_block_sums.data(),
                                 d_second_block_sums,
                                 second_block_sums.size() * sizeof(rocComplex),
                                 handle->streams[0]);
    device_free(handle, d_second_block_sums);
    device_free(handle, d_mean_block_sums);
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }

    double mean_re = 0.0;
    double mean_im = 0.0;
    double second_re = 0.0;
    double second_im = 0.0;
    for (size_t block = 0; block < mean_block_sums.size(); ++block) {
        mean_re += static_cast<double>(mean_block_sums[block].x);
        mean_im += static_cast<double>(mean_block_sums[block].y);
        second_re += static_cast<double>(second_block_sums[block].x);
        second_im += static_cast<double>(second_block_sums[block].y);
    }

    *mean = make_complex(mean_re, mean_im);
    *secondMoment = make_complex(second_re, second_im);
    return ROCQ_STATUS_SUCCESS;
}

rocqStatus_t rocsvGetSparseMatrixMomentsBatch(rocsvHandle_t handle,
                                              rocComplex* d_state,
                                              unsigned numQubits,
                                              const rocComplex* d_data,
                                              const size_t* d_indices,
                                              const size_t* d_indptr,
                                              size_t rows,
                                              size_t cols,
                                              size_t nnz,
                                              rocComplex* means,
                                              rocComplex* secondMoments) {
    if (!handle || !d_indptr || !means || !secondMoments || rows == 0 || cols == 0) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (rows != cols) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (nnz > 0 && (!d_data || !d_indices)) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (uses_distributed_state(handle, d_state)) {
        return ROCQ_STATUS_NOT_IMPLEMENTED;
    }

    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    const size_t batch_size = effective_batch_size(handle, state);
    if (batch_size > 65535) {
        return ROCQ_STATUS_NOT_IMPLEMENTED;
    }

    size_t state_elements = 0;
    if (!compute_power_of_two(numQubits, &state_elements)) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (rows != state_elements || cols != state_elements) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    constexpr int threads_per_block = 256;
    const int blocks = compute_reduction_blocks(rows, threads_per_block);
    if (batch_size > std::numeric_limits<size_t>::max() / static_cast<size_t>(blocks)) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    const size_t total_blocks = batch_size * static_cast<size_t>(blocks);

    void* d_mean_block_sums_void = nullptr;
    rocqStatus_t status = device_malloc(handle,
                                        &d_mean_block_sums_void,
                                        total_blocks * sizeof(rocComplex));
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }
    rocComplex* d_mean_block_sums = static_cast<rocComplex*>(d_mean_block_sums_void);

    void* d_second_block_sums_void = nullptr;
    status = device_malloc(handle,
                           &d_second_block_sums_void,
                           total_blocks * sizeof(rocComplex));
    if (status != ROCQ_STATUS_SUCCESS) {
        device_free(handle, d_mean_block_sums);
        return status;
    }
    rocComplex* d_second_block_sums = static_cast<rocComplex*>(d_second_block_sums_void);

    hipLaunchKernelGGL(reduce_sparse_matrix_moments_batch_kernel,
                       dim3(blocks, static_cast<unsigned>(batch_size)),
                       dim3(threads_per_block),
                       4 * threads_per_block * sizeof(double),
                       handle->streams[0],
                       state,
                       d_data,
                       d_indices,
                       d_indptr,
                       rows,
                       d_mean_block_sums,
                       d_second_block_sums);
    status = check_last_hip_error();
    if (status != ROCQ_STATUS_SUCCESS) {
        device_free(handle, d_second_block_sums);
        device_free(handle, d_mean_block_sums);
        return status;
    }

    std::vector<rocComplex> mean_block_sums(total_blocks);
    status = copy_device_to_host(mean_block_sums.data(),
                                 d_mean_block_sums,
                                 mean_block_sums.size() * sizeof(rocComplex),
                                 handle->streams[0]);
    if (status != ROCQ_STATUS_SUCCESS) {
        device_free(handle, d_second_block_sums);
        device_free(handle, d_mean_block_sums);
        return status;
    }

    std::vector<rocComplex> second_block_sums(total_blocks);
    status = copy_device_to_host(second_block_sums.data(),
                                 d_second_block_sums,
                                 second_block_sums.size() * sizeof(rocComplex),
                                 handle->streams[0]);
    device_free(handle, d_second_block_sums);
    device_free(handle, d_mean_block_sums);
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }

    for (size_t batch = 0; batch < batch_size; ++batch) {
        double mean_re = 0.0;
        double mean_im = 0.0;
        double second_re = 0.0;
        double second_im = 0.0;
        const size_t offset = batch * static_cast<size_t>(blocks);
        for (int block = 0; block < blocks; ++block) {
            const rocComplex& mean_block = mean_block_sums[offset + static_cast<size_t>(block)];
            const rocComplex& second_block = second_block_sums[offset + static_cast<size_t>(block)];
            mean_re += static_cast<double>(mean_block.x);
            mean_im += static_cast<double>(mean_block.y);
            second_re += static_cast<double>(second_block.x);
            second_im += static_cast<double>(second_block.y);
        }
        means[batch] = make_complex(mean_re, mean_im);
        secondMoments[batch] = make_complex(second_re, second_im);
    }
    return ROCQ_STATUS_SUCCESS;
}

rocqStatus_t rocsvProbabilities(rocsvHandle_t handle,
                                rocComplex* d_state,
                                unsigned numQubits,
                                const unsigned* measuredQubits,
                                unsigned numMeasuredQubits,
                                double* h_probabilities) {
    if (!handle || !h_probabilities || !measuredQubits || numMeasuredQubits == 0) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (numMeasuredQubits > 20) {
        return ROCQ_STATUS_NOT_IMPLEMENTED;
    }

    std::vector<unsigned> measured;
    rocqStatus_t status = validate_unique_qubits(measuredQubits, numMeasuredQubits, numQubits, &measured);
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }
    if (uses_distributed_state(handle, d_state)) {
        return compute_distributed_sample_probabilities(handle, numQubits, measured, h_probabilities);
    }

    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    return compute_local_sample_probabilities(handle, state, numQubits, measured, h_probabilities);
}

rocqStatus_t rocsvProbabilitiesBatch(rocsvHandle_t handle,
                                     rocComplex* d_state,
                                     unsigned numQubits,
                                     const unsigned* measuredQubits,
                                     unsigned numMeasuredQubits,
                                     double* h_probabilities) {
    if (!handle || !h_probabilities || !measuredQubits || numMeasuredQubits == 0) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (numMeasuredQubits > 20) {
        return ROCQ_STATUS_NOT_IMPLEMENTED;
    }

    std::vector<unsigned> measured;
    rocqStatus_t status = validate_unique_qubits(measuredQubits, numMeasuredQubits, numQubits, &measured);
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }
    if (uses_distributed_state(handle, d_state)) {
        return compute_distributed_sample_probabilities(handle, numQubits, measured, h_probabilities);
    }

    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    return compute_local_sample_probabilities_batch(handle, state, numQubits, measured, h_probabilities);
}

rocqStatus_t rocsvSample(rocsvHandle_t handle,
                         rocComplex* d_state,
                         unsigned numQubits,
                         const unsigned* measuredQubits,
                         unsigned numMeasuredQubits,
                         unsigned numShots,
                         uint64_t* h_results) {
    if (!handle || !h_results || !measuredQubits || numMeasuredQubits == 0) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (numMeasuredQubits > 20) {
        return ROCQ_STATUS_NOT_IMPLEMENTED;
    }
    if (numShots == 0) {
        return ROCQ_STATUS_SUCCESS;
    }

    std::vector<unsigned> measured;
    rocqStatus_t status = validate_unique_qubits(measuredQubits, numMeasuredQubits, numQubits, &measured);
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }
    if (uses_distributed_state(handle, d_state)) {
        return distributed_sample_with_fallback(handle, numQubits, measured, numShots, h_results);
    }

    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    const size_t num_outcomes = size_t{1} << numMeasuredQubits;
    double* d_outcome_probs = nullptr;
    unsigned* d_measured = nullptr;
    status = accumulate_local_sample_probabilities(
        handle, state, numQubits, measured, &d_outcome_probs, &d_measured);
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }

    hipLaunchKernelGGL(build_sampling_cdf_kernel,
                       dim3(1),
                       dim3(1),
                       0,
                       handle->streams[0],
                       d_outcome_probs,
                       num_outcomes);
    status = check_last_hip_error();
    if (status != ROCQ_STATUS_SUCCESS) {
        device_free(handle, d_outcome_probs);
        device_free(handle, d_measured);
        return status;
    }

    double cdf_last = 0.0;
    if (hipMemcpyAsync(&cdf_last,
                       d_outcome_probs + (num_outcomes - 1),
                       sizeof(double),
                       hipMemcpyDeviceToHost,
                       handle->streams[0]) != hipSuccess) {
        device_free(handle, d_outcome_probs);
        device_free(handle, d_measured);
        return ROCQ_STATUS_HIP_ERROR;
    }
    if (hipStreamSynchronize(handle->streams[0]) != hipSuccess) { // ROCQ_ASYNC_ALLOWED_SYNC
        device_free(handle, d_outcome_probs);
        device_free(handle, d_measured);
        return ROCQ_STATUS_HIP_ERROR;
    }
    if (cdf_last <= 0.0) {
        device_free(handle, d_outcome_probs);
        device_free(handle, d_measured);
        return ROCQ_STATUS_FAILURE;
    }

    void* d_results_void = nullptr;
    status = device_malloc(handle, &d_results_void, static_cast<size_t>(numShots) * sizeof(uint64_t));
    if (status != ROCQ_STATUS_SUCCESS) {
        device_free(handle, d_outcome_probs);
        device_free(handle, d_measured);
        return status;
    }
    uint64_t* d_results = static_cast<uint64_t*>(d_results_void);

    constexpr int threads_per_block = 256;
    const unsigned long long seed = handle->rng();
    const int sample_blocks = compute_reduction_blocks(static_cast<size_t>(numShots), threads_per_block);
    hipLaunchKernelGGL(sample_from_cdf_kernel,
                       dim3(sample_blocks),
                       dim3(threads_per_block),
                       0,
                       handle->streams[0],
                       d_outcome_probs,
                       num_outcomes,
                       d_results,
                       numShots,
                       seed);
    status = check_last_hip_error();
    if (status != ROCQ_STATUS_SUCCESS) {
        device_free(handle, d_results);
        device_free(handle, d_outcome_probs);
        device_free(handle, d_measured);
        return status;
    }

    status = copy_device_to_host(h_results,
                                 d_results,
                                 static_cast<size_t>(numShots) * sizeof(uint64_t),
                                 handle->streams[0]);
    device_free(handle, d_results);
    device_free(handle, d_outcome_probs);
    device_free(handle, d_measured);
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }
    return ROCQ_STATUS_SUCCESS;
}

rocqStatus_t rocsvApplyControlledMatrix(rocsvHandle_t handle,
                                        rocComplex* d_state,
                                        unsigned numQubits,
                                        const unsigned* controlQubits,
                                        unsigned numControls,
                                        const unsigned* targetQubits,
                                        unsigned numTargets,
                                        const rocComplex* d_matrix) {
    if (!handle || !d_matrix || !targetQubits || numTargets == 0 || numTargets > numQubits) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    std::vector<unsigned> targets;
    rocqStatus_t status = validate_unique_qubits(targetQubits, numTargets, numQubits, &targets);
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }

    std::vector<unsigned> controls;
    if (numControls > 0) {
        status = validate_unique_qubits(controlQubits, numControls, numQubits, &controls);
        if (status != ROCQ_STATUS_SUCCESS) {
            return status;
        }
        std::unordered_set<unsigned> target_set(targets.begin(), targets.end());
        for (unsigned ctrl : controls) {
            if (target_set.count(ctrl) > 0) {
                return ROCQ_STATUS_INVALID_VALUE;
            }
        }
    }

    if (uses_distributed_state(handle, d_state)) {
        if (numQubits != handle->globalNumQubits) {
            return ROCQ_STATUS_INVALID_VALUE;
        }
        if (!distributed_all_qubits_local(handle, targets) ||
            !distributed_all_qubits_local(handle, controls)) {
            if (!distributed_host_fallback_enabled()) {
                return ROCQ_STATUS_NOT_IMPLEMENTED;
            }
            if (hipSetDevice(handle->distributedDeviceIds[0]) != hipSuccess) {
                return ROCQ_STATUS_HIP_ERROR;
            }
            size_t matrix_dim = 0;
            if (!compute_power_of_two(numTargets, &matrix_dim) ||
                matrix_dim > (std::numeric_limits<size_t>::max() / matrix_dim)) {
                return ROCQ_STATUS_INVALID_VALUE;
            }
            std::vector<rocComplex> matrix_host;
            status = copy_matrix_from_device(d_matrix,
                                             matrix_dim * matrix_dim,
                                             handle->streams[0],
                                             &matrix_host);
            if (status != ROCQ_STATUS_SUCCESS) {
                return status;
            }
            return apply_matrix_distributed_host_fallback(handle,
                                                          numQubits,
                                                          targets,
                                                          controls,
                                                          matrix_host);
        }
        if (hipSetDevice(handle->distributedDeviceIds[0]) != hipSuccess) {
            return ROCQ_STATUS_HIP_ERROR;
        }

        if (numControls == 0) {
            size_t matrix_dim_no_ctrl = 0;
            if (!compute_power_of_two(numTargets, &matrix_dim_no_ctrl) ||
                matrix_dim_no_ctrl > static_cast<size_t>(std::numeric_limits<unsigned>::max())) {
                return ROCQ_STATUS_INVALID_VALUE;
            }
            return rocsvApplyMatrix(handle,
                                    d_state,
                                    numQubits,
                                    targets.data(),
                                    numTargets,
                                    d_matrix,
                                    static_cast<unsigned>(matrix_dim_no_ctrl));
        }

        if (numControls == 1 && numTargets == 1) {
            std::vector<rocComplex> matrix_host;
            status = copy_matrix_from_device(d_matrix, 4, handle->streams[0], &matrix_host);
            if (status != ROCQ_STATUS_SUCCESS) {
                return status;
            }
            return launch_controlled_single_qubit_matrix(handle,
                                                         handle->d_state,
                                                         numQubits,
                                                         controls[0],
                                                         targets[0],
                                                         matrix_host[0],
                                                         matrix_host[1],
                                                         matrix_host[2],
                                                         matrix_host[3]);
        }
        if (!distributed_host_fallback_enabled()) {
            return ROCQ_STATUS_NOT_IMPLEMENTED;
        }
        size_t matrix_dim = 0;
        if (!compute_power_of_two(numTargets, &matrix_dim) ||
            matrix_dim > (std::numeric_limits<size_t>::max() / matrix_dim)) {
            return ROCQ_STATUS_INVALID_VALUE;
        }
        std::vector<rocComplex> matrix_host;
        status = copy_matrix_from_device(d_matrix,
                                         matrix_dim * matrix_dim,
                                         handle->streams[0],
                                         &matrix_host);
        if (status != ROCQ_STATUS_SUCCESS) {
            return status;
        }
        return apply_matrix_distributed_host_fallback(handle,
                                                      numQubits,
                                                      targets,
                                                      controls,
                                                      matrix_host);
    }

    if (numControls == 0) {
        size_t matrix_dim_no_ctrl = 0;
        if (!compute_power_of_two(numTargets, &matrix_dim_no_ctrl) ||
            matrix_dim_no_ctrl > static_cast<size_t>(std::numeric_limits<unsigned>::max())) {
            return ROCQ_STATUS_INVALID_VALUE;
        }
        return rocsvApplyMatrix(handle,
                                state,
                                numQubits,
                                targets.data(),
                                numTargets,
                                d_matrix,
                                static_cast<unsigned>(matrix_dim_no_ctrl));
    }

    size_t matrix_dim = 0;
    if (!compute_power_of_two(numTargets, &matrix_dim)) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (matrix_dim > 0 && matrix_dim > (std::numeric_limits<size_t>::max() / matrix_dim)) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    const size_t matrix_elements = matrix_dim * matrix_dim;

    const bool using_internal_state = (state == handle->d_state);
    const bool kernel_fast_path_ok = using_internal_state || handle->batchSize == 1;
    if (numControls == 1 && numTargets == 1 && kernel_fast_path_ok) {
        std::vector<rocComplex> matrix_host;
        status = copy_matrix_from_device(d_matrix, 4, handle->streams[0], &matrix_host);
        if (status != ROCQ_STATUS_SUCCESS) {
            return status;
        }
        return launch_controlled_single_qubit_matrix(handle,
                                                     state,
                                                     numQubits,
                                                     controls[0],
                                                     targets[0],
                                                     matrix_host[0],
                                                     matrix_host[1],
                                                     matrix_host[2],
                                                     matrix_host[3]);
    }

    if (!allow_host_matrix_fallback()) {
        return ROCQ_STATUS_NOT_IMPLEMENTED;
    }

    std::vector<rocComplex> matrix_host;
    status = copy_matrix_from_device(d_matrix, matrix_elements, handle->streams[0], &matrix_host);
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }

    const size_t batch_size = effective_batch_size(handle, state);
    size_t state_elements = 0;
    if (!compute_power_of_two(numQubits, &state_elements)) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    return apply_matrix_host_impl(state,
                                  numQubits,
                                  targets,
                                  controls,
                                  matrix_host,
                                  batch_size,
                                  state_elements,
                                  handle->streams[0]);
}

rocqStatus_t rocsvApplyMatrixAndMeasure(rocsvHandle_t handle,
                                        rocComplex* d_state,
                                        unsigned numQubits,
                                        const unsigned* targetQubits,
                                        unsigned numTargetQubits,
                                        const rocComplex* d_matrix,
                                        unsigned qubitToMeasure,
                                        int* outcome) {
    if (!outcome) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    size_t matrix_dim = 0;
    if (!compute_power_of_two(numTargetQubits, &matrix_dim)) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    rocqStatus_t status = rocsvApplyMatrix(handle,
                                           d_state,
                                           numQubits,
                                           targetQubits,
                                           numTargetQubits,
                                           d_matrix,
                                           static_cast<unsigned>(matrix_dim));
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }

    double prob = 0.0;
    return rocsvMeasure(handle, d_state, numQubits, qubitToMeasure, outcome, &prob);
}
