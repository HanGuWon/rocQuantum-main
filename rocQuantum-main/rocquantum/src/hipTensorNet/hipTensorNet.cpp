#include "rocquantum/hipTensorNet.h"
#include "rocquantum/hipTensorNet_api.h"

#include <algorithm>
#include <limits>
#include <map>
#include <new>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>

#ifdef HAS_METIS
#include <metis.h>
#endif

struct rocTnStruct {
    rocquantum::TensorNetworkBase* tn_instance = nullptr;
    rocDataType_t dtype = ROC_TENSORNET_COMPILED_COMPLEX_DTYPE;
};

namespace {

inline bool tensor_dtype_supported_by_build(rocDataType_t dtype) {
    return dtype == ROC_TENSORNET_COMPILED_COMPLEX_DTYPE;
}

inline size_t tensor_element_size_bytes() {
    return sizeof(rocComplex);
}

inline bool pathfinder_algorithm_available(rocPathfinderAlgorithm_t algorithm) {
    switch (algorithm) {
        case ROCTN_PATHFINDER_ALGO_GREEDY:
            return true;
        case ROCTN_PATHFINDER_ALGO_KAHYPAR:
#ifdef HAS_KAHYPAR
            return true;
#else
            return false;
#endif
        case ROCTN_PATHFINDER_ALGO_METIS:
#ifdef HAS_METIS
            return true;
#else
            return false;
#endif
        default:
            return false;
    }
}

inline rocPathfinderAlgorithm_t effective_pathfinder_algorithm(rocPathfinderAlgorithm_t algorithm) {
    return pathfinder_algorithm_available(algorithm) ? algorithm : ROCTN_PATHFINDER_ALGO_GREEDY;
}

inline rocqStatus_t validate_optimizer_config(const hipTensorNetContractionOptimizerConfig_t& config) {
    switch (config.pathfinder_algorithm) {
        case ROCTN_PATHFINDER_ALGO_GREEDY:
        case ROCTN_PATHFINDER_ALGO_KAHYPAR:
        case ROCTN_PATHFINDER_ALGO_METIS:
            break;
        default:
            return ROCQ_STATUS_INVALID_VALUE;
    }
    if (config.num_slices < 0) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    return ROCQ_STATUS_SUCCESS;
}

inline bool contains_mode_index(const std::vector<std::pair<int, int>>& pairs,
                                int mode,
                                bool in_a) {
    for (const auto& p : pairs) {
        if (in_a && p.first == mode) {
            return true;
        }
        if (!in_a && p.second == mode) {
            return true;
        }
    }
    return false;
}

inline long long product_dims(const std::vector<long long>& dims) {
    if (dims.empty()) {
        return 1;
    }
    long long v = 1;
    for (long long d : dims) {
        v *= d;
    }
    return v;
}

inline size_t estimate_pair_memory_bytes(const rocquantum::util::rocTensor& a,
                                         const rocquantum::util::rocTensor& b,
                                         const std::vector<long long>& result_dims) {
    const long long a_elems = a.get_element_count();
    const long long b_elems = b.get_element_count();
    const long long r_elems = product_dims(result_dims);
    const long long total_elems = a_elems + b_elems + r_elems;
    return static_cast<size_t>(total_elems) * tensor_element_size_bytes();
}

inline size_t selection_cost_for_algorithm(const hipTensorNetContractionOptimizerConfig_t& config,
                                           size_t required_bytes,
                                           size_t contracted_modes) {
    const rocPathfinderAlgorithm_t algorithm =
        effective_pathfinder_algorithm(config.pathfinder_algorithm);
    size_t cost = required_bytes;

    if (algorithm == ROCTN_PATHFINDER_ALGO_GREEDY) {
        cost += contracted_modes;
    }

    if (config.memory_limit_bytes > 0 && required_bytes > config.memory_limit_bytes) {
        const size_t automatic_slices =
            (required_bytes + config.memory_limit_bytes - 1) / config.memory_limit_bytes;
        const size_t requested_slices =
            config.num_slices > 0 ? static_cast<size_t>(config.num_slices) : automatic_slices;
        const size_t effective_slices = std::max<size_t>(1, requested_slices);
        const size_t sliced_required_bytes =
            (required_bytes + effective_slices - 1) / effective_slices;
        cost = sliced_required_bytes + (effective_slices * 1024);
    }

    return cost;
}

#ifdef HAS_METIS
inline idx_t metis_shared_mode_weight(const rocquantum::util::rocTensor& a,
                                      const rocquantum::util::rocTensor& b) {
    idx_t weight = 0;
    for (size_t i = 0; i < a.labels_.size(); ++i) {
        for (size_t j = 0; j < b.labels_.size(); ++j) {
            if (a.labels_[i] == b.labels_[j] && a.dimensions_[i] == b.dimensions_[j]) {
                const long long dim = std::max<long long>(1, a.dimensions_[i]);
                const long long capped = std::min<long long>(
                    dim, static_cast<long long>(std::numeric_limits<idx_t>::max()));
                weight += static_cast<idx_t>(capped);
            }
        }
    }
    return weight;
}

inline bool metis_partition_active_tensors(
    const std::map<int, rocquantum::util::rocTensor>& active,
    const hipTensorNetContractionOptimizerConfig_t& config,
    std::map<int, int>* partition_by_id) {
    if (!partition_by_id || active.size() < 3) {
        return false;
    }

    std::vector<int> tensor_ids;
    tensor_ids.reserve(active.size());
    std::vector<const rocquantum::util::rocTensor*> tensors;
    tensors.reserve(active.size());
    for (const auto& entry : active) {
        tensor_ids.push_back(entry.first);
        tensors.push_back(&entry.second);
    }

    std::vector<std::vector<std::pair<idx_t, idx_t>>> adjacency(tensors.size());
    for (size_t i = 0; i < tensors.size(); ++i) {
        for (size_t j = i + 1; j < tensors.size(); ++j) {
            const idx_t weight = metis_shared_mode_weight(*tensors[i], *tensors[j]);
            if (weight <= 0) {
                continue;
            }
            adjacency[i].push_back({static_cast<idx_t>(j), weight});
            adjacency[j].push_back({static_cast<idx_t>(i), weight});
        }
    }

    size_t edge_entries = 0;
    for (const auto& neighbors : adjacency) {
        edge_entries += neighbors.size();
    }
    if (edge_entries == 0) {
        return false;
    }

    std::vector<idx_t> xadj(tensors.size() + 1, 0);
    std::vector<idx_t> adjncy;
    std::vector<idx_t> adjwgt;
    adjncy.reserve(edge_entries);
    adjwgt.reserve(edge_entries);
    for (size_t i = 0; i < adjacency.size(); ++i) {
        xadj[i + 1] = xadj[i] + static_cast<idx_t>(adjacency[i].size());
        for (const auto& neighbor : adjacency[i]) {
            adjncy.push_back(neighbor.first);
            adjwgt.push_back(neighbor.second);
        }
    }

    idx_t nvtxs = static_cast<idx_t>(tensors.size());
    idx_t ncon = 1;
    idx_t nparts = 2;
    idx_t edgecut = 0;
    std::vector<idx_t> part(tensors.size(), 0);
    std::vector<idx_t> options(METIS_NOPTIONS);
    METIS_SetDefaultOptions(options.data());
    if (config.algo_config.metis_config.num_iterations > 0) {
        options[METIS_OPTION_NITER] =
            static_cast<idx_t>(config.algo_config.metis_config.num_iterations);
    }

    const int metis_status = METIS_PartGraphKway(&nvtxs,
                                                 &ncon,
                                                 xadj.data(),
                                                 adjncy.data(),
                                                 nullptr,
                                                 nullptr,
                                                 adjwgt.data(),
                                                 &nparts,
                                                 nullptr,
                                                 nullptr,
                                                 options.data(),
                                                 &edgecut,
                                                 part.data());
    if (metis_status != METIS_OK) {
        return false;
    }

    partition_by_id->clear();
    for (size_t i = 0; i < tensor_ids.size(); ++i) {
        (*partition_by_id)[tensor_ids[i]] = static_cast<int>(part[i]);
    }
    return true;
}

inline size_t add_metis_partition_penalty(size_t selection_cost) {
    constexpr size_t penalty = std::numeric_limits<size_t>::max() / 4;
    if (selection_cost > std::numeric_limits<size_t>::max() - penalty) {
        return std::numeric_limits<size_t>::max();
    }
    return selection_cost + penalty;
}
#endif

} // namespace

namespace rocquantum {

template <typename T>
TensorNetwork<T>::TensorNetwork(util::WorkspaceManager* external_workspace, hipStream_t stream) {
    if (external_workspace) {
        workspace_ = external_workspace;
        owns_workspace_ = false;
    } else {
        workspace_ = new util::WorkspaceManager(DEFAULT_WORKSPACE_SIZE_BYTES, stream);
        owns_workspace_ = true;
    }
}

template <typename T>
TensorNetwork<T>::~TensorNetwork() {
    if (owns_workspace_ && workspace_) {
        delete workspace_;
        workspace_ = nullptr;
    }
}

template <typename T>
int TensorNetwork<T>::add_tensor(const util::rocTensor& tensor) {
    initial_tensors_.push_back(tensor);
    return static_cast<int>(initial_tensors_.size() - 1);
}

template <typename T>
std::vector<std::pair<int, int>> TensorNetwork<T>::find_shared_mode_indices(
    const util::rocTensor& t1,
    const util::rocTensor& t2) const {
    std::vector<std::pair<int, int>> out;
    for (size_t i = 0; i < t1.labels_.size(); ++i) {
        for (size_t j = 0; j < t2.labels_.size(); ++j) {
            if (t1.labels_[i] == t2.labels_[j]) {
                if (t1.dimensions_[i] != t2.dimensions_[j]) {
                    continue;
                }
                out.emplace_back(static_cast<int>(i), static_cast<int>(j));
            }
        }
    }
    return out;
}

template <typename T>
void TensorNetwork<T>::get_resulting_tensor_metadata(
    const util::rocTensor& t1,
    const util::rocTensor& t2,
    const std::vector<std::pair<int, int>>& contracted_mode_pairs,
    std::vector<long long>& out_new_dims,
    std::vector<std::string>& out_new_labels) const {
    out_new_dims.clear();
    out_new_labels.clear();

    for (size_t i = 0; i < t1.labels_.size(); ++i) {
        if (!contains_mode_index(contracted_mode_pairs, static_cast<int>(i), true)) {
            out_new_labels.push_back(t1.labels_[i]);
            out_new_dims.push_back(t1.dimensions_[i]);
        }
    }
    for (size_t j = 0; j < t2.labels_.size(); ++j) {
        if (!contains_mode_index(contracted_mode_pairs, static_cast<int>(j), false)) {
            out_new_labels.push_back(t2.labels_[j]);
            out_new_dims.push_back(t2.dimensions_[j]);
        }
    }

    if (out_new_dims.empty()) {
        out_new_dims = {1};
        out_new_labels = {"scalar"};
    }
}

template <typename T>
rocqStatus_t TensorNetwork<T>::contract(const hipTensorNetContractionOptimizerConfig_t* config,
                                        util::rocTensor* result_tensor,
                                        rocblas_handle blas_handle,
                                        hipStream_t stream) {
    if (!config || !result_tensor) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (!blas_handle) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    rocqStatus_t config_status = validate_optimizer_config(*config);
    if (config_status != ROCQ_STATUS_SUCCESS) {
        return config_status;
    }

    if (rocblas_set_stream(blas_handle, stream) != rocblas_status_success) {
        return ROCQ_STATUS_FAILURE;
    }

    if (initial_tensors_.empty()) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    if (initial_tensors_.size() == 1) {
        if (result_tensor->owned_ && result_tensor->data_) {
            util::rocTensorFree(result_tensor);
        }
        *result_tensor = initial_tensors_[0];
        return ROCQ_STATUS_SUCCESS;
    }

    std::map<int, util::rocTensor> active;
    std::set<int> owned_intermediate_ids;
    for (size_t i = 0; i < initial_tensors_.size(); ++i) {
        active[static_cast<int>(i)] = initial_tensors_[i];
    }
    int next_intermediate_id = static_cast<int>(initial_tensors_.size());

    while (active.size() > 1) {
        bool found = false;
        int best_a_id = -1;
        int best_b_id = -1;
        size_t best_cost = std::numeric_limits<size_t>::max();
        std::vector<std::pair<int, int>> best_mode_pairs;
        std::vector<long long> best_result_dims;
        std::vector<std::string> best_result_labels;

        std::map<int, int> metis_partition;
#ifdef HAS_METIS
        const bool using_metis_partition =
            effective_pathfinder_algorithm(config->pathfinder_algorithm) == ROCTN_PATHFINDER_ALGO_METIS &&
            metis_partition_active_tensors(active, *config, &metis_partition);
#else
        constexpr bool using_metis_partition = false;
#endif

        for (auto it_a = active.begin(); it_a != active.end(); ++it_a) {
            auto it_b = it_a;
            ++it_b;
            for (; it_b != active.end(); ++it_b) {
                const util::rocTensor& a = it_a->second;
                const util::rocTensor& b = it_b->second;

                std::vector<std::pair<int, int>> mode_pairs = find_shared_mode_indices(a, b);
                std::vector<long long> result_dims;
                std::vector<std::string> result_labels;
                get_resulting_tensor_metadata(a, b, mode_pairs, result_dims, result_labels);

                const size_t required_bytes = estimate_pair_memory_bytes(a, b, result_dims);

                size_t selection_cost =
                    selection_cost_for_algorithm(*config, required_bytes, mode_pairs.size());
#ifdef HAS_METIS
                if (using_metis_partition &&
                    metis_partition[it_a->first] != metis_partition[it_b->first]) {
                    selection_cost = add_metis_partition_penalty(selection_cost);
                }
#endif

                if (!found || selection_cost < best_cost) {
                    found = true;
                    best_cost = selection_cost;
                    best_a_id = it_a->first;
                    best_b_id = it_b->first;
                    best_mode_pairs = std::move(mode_pairs);
                    best_result_dims = std::move(result_dims);
                    best_result_labels = std::move(result_labels);
                }
            }
        }

        if (!found) {
            return ROCQ_STATUS_FAILURE;
        }

        util::rocTensor& tensor_a = active[best_a_id];
        util::rocTensor& tensor_b = active[best_b_id];

        std::vector<int> result_a_modes_order;
        result_a_modes_order.reserve(tensor_a.rank());
        for (size_t i = 0; i < tensor_a.rank(); ++i) {
            if (!contains_mode_index(best_mode_pairs, static_cast<int>(i), true)) {
                result_a_modes_order.push_back(static_cast<int>(i));
            }
        }

        std::vector<int> result_b_modes_order;
        result_b_modes_order.reserve(tensor_b.rank());
        for (size_t j = 0; j < tensor_b.rank(); ++j) {
            if (!contains_mode_index(best_mode_pairs, static_cast<int>(j), false)) {
                result_b_modes_order.push_back(static_cast<int>(j));
            }
        }

        util::rocTensor contracted;
        contracted.dimensions_ = best_result_dims;
        contracted.labels_ = best_result_labels;
        contracted.calculate_strides();
        rocqStatus_t alloc_status = util::rocTensorAllocate(&contracted);
        if (alloc_status != ROCQ_STATUS_SUCCESS) {
            return alloc_status;
        }

        rocqStatus_t status = util::rocTensorContractPair_internal(&contracted,
                                                                   &tensor_a,
                                                                   &tensor_b,
                                                                   best_mode_pairs,
                                                                   result_a_modes_order,
                                                                   result_b_modes_order,
                                                                   blas_handle,
                                                                   stream);
        if (status != ROCQ_STATUS_SUCCESS) {
            util::rocTensorFree(&contracted);
            return status;
        }

        if (owned_intermediate_ids.count(best_a_id) > 0) {
            util::rocTensorFree(&tensor_a);
            owned_intermediate_ids.erase(best_a_id);
        }
        if (owned_intermediate_ids.count(best_b_id) > 0) {
            util::rocTensorFree(&tensor_b);
            owned_intermediate_ids.erase(best_b_id);
        }

        active.erase(best_a_id);
        active.erase(best_b_id);

        const int new_id = next_intermediate_id++;
        active[new_id] = std::move(contracted);
        owned_intermediate_ids.insert(new_id);
    }

    auto final_it = active.begin();
    const int final_id = final_it->first;

    if (result_tensor->owned_ && result_tensor->data_) {
        util::rocTensorFree(result_tensor);
    }
    *result_tensor = std::move(final_it->second);

    owned_intermediate_ids.erase(final_id);
    for (int id : owned_intermediate_ids) {
        auto it = active.find(id);
        if (it != active.end()) {
            util::rocTensorFree(&it->second);
        }
    }

    return ROCQ_STATUS_SUCCESS;
}

template class TensorNetwork<rocComplex>;

} // namespace rocquantum

extern "C" {

rocqStatus_t rocTensorNetworkGetCapabilities(hipTensorNetCapabilities_t* capabilities) {
    if (!capabilities) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    capabilities->supports_c64 = (ROC_TENSORNET_COMPILED_COMPLEX_DTYPE == ROC_DATATYPE_C64) ? 1 : 0;
    capabilities->supports_c128 = (ROC_TENSORNET_COMPILED_COMPLEX_DTYPE == ROC_DATATYPE_C128) ? 1 : 0;
    capabilities->supports_pathfinder_greedy = 1;
#ifdef HAS_KAHYPAR
    capabilities->supports_pathfinder_kahypar = 1;
#else
    capabilities->supports_pathfinder_kahypar = 0;
#endif
#ifdef HAS_METIS
    capabilities->supports_pathfinder_metis = 1;
#else
    capabilities->supports_pathfinder_metis = 0;
#endif
    capabilities->supports_memory_limit_planning = 1;
    capabilities->supports_runtime_slicing = 0;
    return ROCQ_STATUS_SUCCESS;
}

rocqStatus_t rocTensorNetworkCreate(rocTensorNetworkHandle_t* handle, rocDataType_t dtype) {
    if (!handle) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    rocTnStruct* h = new (std::nothrow) rocTnStruct;
    if (!h) {
        return ROCQ_STATUS_ALLOCATION_FAILED;
    }

    try {
        if (!tensor_dtype_supported_by_build(dtype)) {
            delete h;
            return ROCQ_STATUS_NOT_IMPLEMENTED;
        }
        h->tn_instance = new rocquantum::TensorNetwork<rocComplex>();
        h->dtype = dtype;
        *handle = h;
        return ROCQ_STATUS_SUCCESS;
    } catch (...) {
        delete h;
        return ROCQ_STATUS_ALLOCATION_FAILED;
    }
}

rocqStatus_t rocTensorNetworkDestroy(rocTensorNetworkHandle_t handle) {
    if (handle) {
        delete handle->tn_instance;
        delete handle;
    }
    return ROCQ_STATUS_SUCCESS;
}

rocqStatus_t rocTensorNetworkAddTensor(rocTensorNetworkHandle_t handle, const rocquantum::util::rocTensor* tensor) {
    if (!handle || !handle->tn_instance || !tensor) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    handle->tn_instance->add_tensor(*tensor);
    return ROCQ_STATUS_SUCCESS;
}

rocqStatus_t rocTensorNetworkContract(rocTensorNetworkHandle_t handle,
                                      const hipTensorNetContractionOptimizerConfig_t* config,
                                      rocquantum::util::rocTensor* result_tensor,
                                      rocblas_handle blas_handle,
                                      hipStream_t stream) {
    if (!handle || !handle->tn_instance || !config || !result_tensor) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    bool owns_blas = false;
    if (!blas_handle) {
        if (rocblas_create_handle(&blas_handle) != rocblas_status_success) {
            return ROCQ_STATUS_FAILURE;
        }
        owns_blas = true;
    }

    rocqStatus_t status = handle->tn_instance->contract(config, result_tensor, blas_handle, stream);

    if (owns_blas) {
        rocblas_destroy_handle(blas_handle);
    }
    return status;
}

rocqStatus_t rocTensorSVD(rocTensorNetworkHandle_t handle,
                          rocquantum::util::rocTensor* U,
                          rocquantum::util::rocTensor* S,
                          rocquantum::util::rocTensor* V,
                          const rocquantum::util::rocTensor* A,
                          void* workspace) {
    (void)workspace;
    if (!handle || !U || !S || !V || !A || !A->data_) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (handle->dtype != ROC_TENSORNET_COMPILED_COMPLEX_DTYPE) {
        return ROCQ_STATUS_NOT_IMPLEMENTED;
    }
    if (A->rank() != 2 || A->dimensions_.size() != 2) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (A->dimensions_[0] <= 0 || A->dimensions_[1] <= 0) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    const rocblas_int m = static_cast<rocblas_int>(A->dimensions_[0]);
    const rocblas_int n = static_cast<rocblas_int>(A->dimensions_[1]);
    const rocblas_int min_mn = std::min(m, n);

    auto ensure_output_tensor = [](rocquantum::util::rocTensor* tensor,
                                   const std::vector<long long>& expected_dims) -> rocqStatus_t {
        if (!tensor) {
            return ROCQ_STATUS_INVALID_VALUE;
        }
        if (tensor->data_ && !tensor->owned_) {
            if (tensor->dimensions_ != expected_dims) {
                return ROCQ_STATUS_INVALID_VALUE;
            }
            if (tensor->strides_.empty()) {
                tensor->calculate_strides();
            }
            return ROCQ_STATUS_SUCCESS;
        }
        tensor->dimensions_ = expected_dims;
        tensor->calculate_strides();
        return rocquantum::util::rocTensorAllocate(tensor);
    };

    rocqStatus_t status = ensure_output_tensor(U, {static_cast<long long>(m), static_cast<long long>(m)});
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }
    status = ensure_output_tensor(V, {static_cast<long long>(n), static_cast<long long>(n)});
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }
    status = ensure_output_tensor(S, {static_cast<long long>(min_mn)});
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }

    rocquantum::util::rocTensor A_work;
    A_work.dimensions_ = A->dimensions_;
    A_work.calculate_strides();
    status = rocquantum::util::rocTensorAllocate(&A_work);
    if (status != ROCQ_STATUS_SUCCESS) {
        return status;
    }

    if (hipMemcpy(A_work.data_,
                  A->data_,
                  static_cast<size_t>(m) * static_cast<size_t>(n) * sizeof(rocComplex),
                  hipMemcpyDeviceToDevice) != hipSuccess) {
        rocquantum::util::rocTensorFree(&A_work);
        return ROCQ_STATUS_HIP_ERROR;
    }

#ifdef ROCQ_PRECISION_DOUBLE
    using RocSolverReal = double;
    using RocSolverComplex = rocblas_double_complex;
#else
    using RocSolverReal = float;
    using RocSolverComplex = rocblas_float_complex;
#endif

    RocSolverReal* d_singular_values = nullptr;
    if (hipMalloc(&d_singular_values, static_cast<size_t>(min_mn) * sizeof(RocSolverReal)) != hipSuccess) {
        rocquantum::util::rocTensorFree(&A_work);
        return ROCQ_STATUS_ALLOCATION_FAILED;
    }

    RocSolverReal* d_superdiag = nullptr;
    const size_t superdiag_size = static_cast<size_t>(std::max<rocblas_int>(1, min_mn - 1));
    if (hipMalloc(&d_superdiag, superdiag_size * sizeof(RocSolverReal)) != hipSuccess) {
        hipFree(d_singular_values);
        rocquantum::util::rocTensorFree(&A_work);
        return ROCQ_STATUS_ALLOCATION_FAILED;
    }

    rocblas_int* d_info = nullptr;
    if (hipMalloc(&d_info, sizeof(rocblas_int)) != hipSuccess) {
        hipFree(d_superdiag);
        hipFree(d_singular_values);
        rocquantum::util::rocTensorFree(&A_work);
        return ROCQ_STATUS_ALLOCATION_FAILED;
    }
    if (hipMemset(d_info, 0, sizeof(rocblas_int)) != hipSuccess) {
        hipFree(d_info);
        hipFree(d_superdiag);
        hipFree(d_singular_values);
        rocquantum::util::rocTensorFree(&A_work);
        return ROCQ_STATUS_HIP_ERROR;
    }

    rocblas_handle blas_handle = nullptr;
    if (rocblas_create_handle(&blas_handle) != rocblas_status_success) {
        hipFree(d_info);
        hipFree(d_superdiag);
        hipFree(d_singular_values);
        rocquantum::util::rocTensorFree(&A_work);
        return ROCQ_STATUS_FAILURE;
    }

#ifdef ROCQ_PRECISION_DOUBLE
    const rocblas_status svd_status = rocsolver_zgesvd(blas_handle,
                                                       rocblas_svect_all,
                                                       rocblas_svect_all,
                                                       m,
                                                       n,
                                                       reinterpret_cast<RocSolverComplex*>(A_work.data_),
                                                       m,
                                                       d_singular_values,
                                                       reinterpret_cast<RocSolverComplex*>(U->data_),
                                                       m,
                                                       reinterpret_cast<RocSolverComplex*>(V->data_),
                                                       n,
                                                       d_superdiag,
                                                       rocblas_outofplace,
                                                       d_info);
#else
    const rocblas_status svd_status = rocsolver_cgesvd(blas_handle,
                                                       rocblas_svect_all,
                                                       rocblas_svect_all,
                                                       m,
                                                       n,
                                                       reinterpret_cast<RocSolverComplex*>(A_work.data_),
                                                       m,
                                                       d_singular_values,
                                                       reinterpret_cast<RocSolverComplex*>(U->data_),
                                                       m,
                                                       reinterpret_cast<RocSolverComplex*>(V->data_),
                                                       n,
                                                       d_superdiag,
                                                       rocblas_outofplace,
                                                       d_info);
#endif
    rocblas_destroy_handle(blas_handle);

    if (svd_status != rocblas_status_success) {
        hipFree(d_info);
        hipFree(d_superdiag);
        hipFree(d_singular_values);
        rocquantum::util::rocTensorFree(&A_work);
        return ROCQ_STATUS_FAILURE;
    }

    rocblas_int info_h = 0;
    if (hipMemcpy(&info_h, d_info, sizeof(rocblas_int), hipMemcpyDeviceToHost) != hipSuccess) {
        hipFree(d_info);
        hipFree(d_superdiag);
        hipFree(d_singular_values);
        rocquantum::util::rocTensorFree(&A_work);
        return ROCQ_STATUS_HIP_ERROR;
    }
    if (info_h != 0) {
        hipFree(d_info);
        hipFree(d_superdiag);
        hipFree(d_singular_values);
        rocquantum::util::rocTensorFree(&A_work);
        return ROCQ_STATUS_FAILURE;
    }

    std::vector<RocSolverReal> singular_values_host(static_cast<size_t>(min_mn), 0.0);
    if (hipMemcpy(singular_values_host.data(),
                  d_singular_values,
                  singular_values_host.size() * sizeof(RocSolverReal),
                  hipMemcpyDeviceToHost) != hipSuccess) {
        hipFree(d_info);
        hipFree(d_superdiag);
        hipFree(d_singular_values);
        rocquantum::util::rocTensorFree(&A_work);
        return ROCQ_STATUS_HIP_ERROR;
    }

    std::vector<rocComplex> singular_values_complex(static_cast<size_t>(min_mn));
    for (rocblas_int i = 0; i < min_mn; ++i) {
        singular_values_complex[static_cast<size_t>(i)] = {
            singular_values_host[static_cast<size_t>(i)],
            0.0};
    }

    if (hipMemcpy(S->data_,
                  singular_values_complex.data(),
                  singular_values_complex.size() * sizeof(rocComplex),
                  hipMemcpyHostToDevice) != hipSuccess) {
        hipFree(d_info);
        hipFree(d_superdiag);
        hipFree(d_singular_values);
        rocquantum::util::rocTensorFree(&A_work);
        return ROCQ_STATUS_HIP_ERROR;
    }

    hipFree(d_info);
    hipFree(d_superdiag);
    hipFree(d_singular_values);
    rocquantum::util::rocTensorFree(&A_work);
    return ROCQ_STATUS_SUCCESS;
}

} // extern "C"
