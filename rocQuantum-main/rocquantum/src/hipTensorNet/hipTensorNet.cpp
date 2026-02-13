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

struct rocTnStruct {
    rocquantum::TensorNetworkBase* tn_instance = nullptr;
    rocDataType_t dtype = ROC_DATATYPE_C64;
};

namespace {

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
    return static_cast<size_t>(total_elems) * sizeof(rocComplex);
}

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

                size_t selection_cost = required_bytes;
                if (config->memory_limit_bytes > 0 && required_bytes > config->memory_limit_bytes) {
                    // Bias toward smaller temporary footprints when over limit.
                    selection_cost = required_bytes + static_cast<size_t>(1) << 30;
                }

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

rocqStatus_t rocTensorNetworkCreate(rocTensorNetworkHandle_t* handle, rocDataType_t dtype) {
    if (!handle) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    rocTnStruct* h = new (std::nothrow) rocTnStruct;
    if (!h) {
        return ROCQ_STATUS_ALLOCATION_FAILED;
    }

    try {
        if (dtype != ROC_DATATYPE_C64) {
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
    (void)handle;
    (void)U;
    (void)S;
    (void)V;
    (void)A;
    (void)workspace;
    return ROCQ_STATUS_NOT_IMPLEMENTED;
}

} // extern "C"
