#include "rocquantum/hipStateVec.h"

#include <hip/hip_runtime.h>
#include <hiprand/hiprand.h>
#include <hiprand/hiprand_kernel.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Kernels implemented in the corresponding .hip translation units.
__global__ void apply_single_qubit_matrix_kernel(rocComplex* state,
                                                 unsigned numQubits,
                                                 unsigned targetQubit,
                                                 rocComplex m00,
                                                 rocComplex m01,
                                                 rocComplex m10,
                                                 rocComplex m11,
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
};

namespace {

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

inline rocqStatus_t clear_distributed_state_storage(rocsvInternalHandle* handle) {
    if (!handle) {
        return ROCQ_STATUS_INVALID_VALUE;
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
            return ROCQ_STATUS_NOT_IMPLEMENTED;
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
        return sync_distributed_streams(handle);
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
            return ROCQ_STATUS_NOT_IMPLEMENTED;
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
        return sync_distributed_streams(handle);
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

rocqStatus_t rocsvSynchronize(rocsvHandle_t handle) {
    if (!handle) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (handle->distributedMode) {
        return sync_distributed_streams(handle);
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
    }

    if (hipSetDevice(original_device) != hipSuccess) {
        clear_distributed_state_storage(handle);
        return ROCQ_STATUS_HIP_ERROR;
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
    return sync_distributed_streams(handle);
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
            return sync_distributed_streams(handle);
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

// --- Parametrised single-qubit rotations ------------------------------------

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
            return ROCQ_STATUS_NOT_IMPLEMENTED;
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
        return sync_distributed_streams(handle);
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
            return ROCQ_STATUS_NOT_IMPLEMENTED;
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
        return sync_distributed_streams(handle);
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
            return ROCQ_STATUS_NOT_IMPLEMENTED;
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
            std::vector<unsigned*> d_targets_local_buffers(
                static_cast<size_t>(handle->distributedGpuCount), nullptr);
            std::vector<rocComplex*> d_matrix_local_buffers(
                static_cast<size_t>(handle->distributedGpuCount), nullptr);

            int restore_device = 0;
            const bool has_restore_device = (hipGetDevice(&restore_device) == hipSuccess);
            auto free_tmp_buffers = [&]() {
                for (int r = 0; r < handle->distributedGpuCount; ++r) {
                    const size_t rank_idx = static_cast<size_t>(r);
                    if (hipSetDevice(handle->distributedDeviceIds[rank_idx]) != hipSuccess) {
                        continue;
                    }
                    if (d_matrix_local_buffers[rank_idx]) {
                        (void)hipFree(d_matrix_local_buffers[rank_idx]);
                        d_matrix_local_buffers[rank_idx] = nullptr;
                    }
                    if (d_targets_local_buffers[rank_idx]) {
                        (void)hipFree(d_targets_local_buffers[rank_idx]);
                        d_targets_local_buffers[rank_idx] = nullptr;
                    }
                }
                if (has_restore_device) {
                    (void)hipSetDevice(restore_device);
                }
            };

            for (int rank = 0; rank < handle->distributedGpuCount; ++rank) {
                const size_t rank_idx = static_cast<size_t>(rank);
                if (hipSetDevice(handle->distributedDeviceIds[rank_idx]) != hipSuccess) {
                    free_tmp_buffers();
                    return ROCQ_STATUS_HIP_ERROR;
                }

                unsigned* d_targets_local = nullptr;
                if (hipMalloc(&d_targets_local, numTargetQubits * sizeof(unsigned)) != hipSuccess) {
                    free_tmp_buffers();
                    return ROCQ_STATUS_ALLOCATION_FAILED;
                }
                d_targets_local_buffers[rank_idx] = d_targets_local;
                if (hipMemcpyAsync(d_targets_local,
                                   targets.data(),
                                   numTargetQubits * sizeof(unsigned),
                                   hipMemcpyHostToDevice,
                                   handle->distributedStreams[rank_idx]) != hipSuccess) {
                    free_tmp_buffers();
                    return ROCQ_STATUS_HIP_ERROR;
                }

                rocComplex* d_matrix_local = nullptr;
                if (hipMalloc(&d_matrix_local, matrix_elements * sizeof(rocComplex)) != hipSuccess) {
                    free_tmp_buffers();
                    return ROCQ_STATUS_ALLOCATION_FAILED;
                }
                d_matrix_local_buffers[rank_idx] = d_matrix_local;
                if (hipMemcpyAsync(d_matrix_local,
                                   matrix_host.data(),
                                   matrix_elements * sizeof(rocComplex),
                                   hipMemcpyHostToDevice,
                                   handle->distributedStreams[rank_idx]) != hipSuccess) {
                    free_tmp_buffers();
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
                    free_tmp_buffers();
                    return launch_status;
                }
            }
            const rocqStatus_t sync_status = sync_distributed_streams(handle);
            free_tmp_buffers();
            return sync_status;
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
        return sync_distributed_streams(handle);
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
        return ROCQ_STATUS_NOT_IMPLEMENTED;
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
        return ROCQ_STATUS_NOT_IMPLEMENTED;
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
        return ROCQ_STATUS_NOT_IMPLEMENTED;
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
    if (uses_distributed_state(handle, d_state)) {
        return ROCQ_STATUS_NOT_IMPLEMENTED;
    }

    std::vector<unsigned> targets;
    rocqStatus_t status = validate_unique_qubits(targetQubits, numTargetPaulis, numQubits, &targets);
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
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
    if (uses_distributed_state(handle, d_state)) {
        return ROCQ_STATUS_NOT_IMPLEMENTED;
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
    if (uses_distributed_state(handle, d_state)) {
        return ROCQ_STATUS_NOT_IMPLEMENTED;
    }

    std::vector<unsigned> measured;
    rocqStatus_t status = validate_unique_qubits(measuredQubits, numMeasuredQubits, numQubits, &measured);
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
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

    void* d_measured_void = nullptr;
    status = device_malloc(handle, &d_measured_void, measured.size() * sizeof(unsigned));
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

    const size_t num_outcomes = size_t{1} << numMeasuredQubits;
    void* d_outcome_probs_void = nullptr;
    status = device_malloc(handle, &d_outcome_probs_void, num_outcomes * sizeof(double));
    if (status != ROCQ_STATUS_SUCCESS) {
        device_free(handle, d_measured);
        return status;
    }
    double* d_outcome_probs = static_cast<double*>(d_outcome_probs_void);
    if (hipMemsetAsync(d_outcome_probs, 0, num_outcomes * sizeof(double), handle->streams[0]) != hipSuccess) {
        device_free(handle, d_outcome_probs);
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
                       d_outcome_probs);
    status = check_last_hip_error();
    if (status != ROCQ_STATUS_SUCCESS) {
        device_free(handle, d_outcome_probs);
        device_free(handle, d_measured);
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
            return ROCQ_STATUS_NOT_IMPLEMENTED;
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
        return ROCQ_STATUS_NOT_IMPLEMENTED;
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
