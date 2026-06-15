#include "rocquantum/hipTensorNet.h"
#include "rocquantum/hipTensorNet_api.h"
#include "rocquantum/rocTensorUtil.h"

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool check_status(rocqStatus_t status, const char* what) {
    if (status != ROCQ_STATUS_SUCCESS) {
        std::cerr << what << " failed with status " << static_cast<int>(status) << "\n";
        return false;
    }
    return true;
}

bool check_hip(hipError_t err, const char* what) {
    if (err != hipSuccess) {
        std::cerr << what << " failed: " << hipGetErrorString(err) << "\n";
        return false;
    }
    return true;
}

bool check_rocblas(rocblas_status status, const char* what) {
    if (status != rocblas_status_success) {
        std::cerr << what << " failed: " << static_cast<int>(status) << "\n";
        return false;
    }
    return true;
}

struct CaseResult {
    std::string name;
    int status = 0;
    double ms_per_trial = 0.0;
};

bool prepare_tensor(rocquantum::util::rocTensor& tensor,
                    const std::vector<long long>& dims,
                    const std::vector<std::string>& labels,
                    const std::vector<rocComplex>& host,
                    hipStream_t stream) {
    tensor.dimensions_ = dims;
    tensor.labels_ = labels;
    tensor.calculate_strides();
    return check_status(rocquantum::util::rocTensorAllocate(&tensor), "rocTensorAllocate") &&
           check_hip(hipMemcpyAsync(tensor.data_,
                                    host.data(),
                                    host.size() * sizeof(rocComplex),
                                    hipMemcpyHostToDevice,
                                    stream),
                     "hipMemcpyAsync(tensor)");
}

bool build_network(rocTensorNetworkHandle_t* tn,
                   rocquantum::util::rocTensor& a,
                   rocquantum::util::rocTensor& b,
                   hipStream_t stream) {
    if (!check_status(rocTensorNetworkCreate(tn, ROC_DATATYPE_C64), "rocTensorNetworkCreate")) {
        return false;
    }

    const std::vector<rocComplex> hA = {
        {1.0f, 0.0f}, {4.0f, 0.0f},
        {2.0f, 0.0f}, {5.0f, 0.0f},
        {3.0f, 0.0f}, {6.0f, 0.0f},
    };
    const std::vector<rocComplex> hB = {
        {7.0f, 0.0f}, {9.0f, 0.0f}, {11.0f, 0.0f},
        {8.0f, 0.0f}, {10.0f, 0.0f}, {12.0f, 0.0f},
    };

    if (!prepare_tensor(a, {2, 3}, {"i", "k"}, hA, stream) ||
        !prepare_tensor(b, {3, 2}, {"k", "j"}, hB, stream) ||
        !check_hip(hipStreamSynchronize(stream), "hipStreamSynchronize(upload)") ||
        !check_status(rocTensorNetworkAddTensor(*tn, &a), "rocTensorNetworkAddTensor(A)") ||
        !check_status(rocTensorNetworkAddTensor(*tn, &b), "rocTensorNetworkAddTensor(B)")) {
        return false;
    }

    return true;
}

CaseResult run_contraction_case(const std::string& name,
                                unsigned trials,
                                size_t memory_limit_bytes,
                                int num_slices,
                                rocblas_handle blas,
                                hipStream_t stream) {
    CaseResult result{name};
    rocTensorNetworkHandle_t tn = nullptr;
    rocquantum::util::rocTensor a;
    rocquantum::util::rocTensor b;

    if (!build_network(&tn, a, b, stream)) {
        result.status = 1;
        if (tn) {
            rocTensorNetworkDestroy(tn);
        }
        rocquantum::util::rocTensorFree(&a);
        rocquantum::util::rocTensorFree(&b);
        return result;
    }

    hipTensorNetContractionOptimizerConfig_t config{};
    config.pathfinder_algorithm = ROCTN_PATHFINDER_ALGO_GREEDY;
    config.memory_limit_bytes = memory_limit_bytes;
    config.num_slices = num_slices;

    auto start = std::chrono::steady_clock::now();
    for (unsigned i = 0; i < trials; ++i) {
        rocquantum::util::rocTensor result_tensor;
        if (!check_status(rocTensorNetworkContract(tn, &config, &result_tensor, blas, stream),
                          "rocTensorNetworkContract") ||
            !check_hip(hipStreamSynchronize(stream), "hipStreamSynchronize(contract)")) {
            result.status = 1;
            rocquantum::util::rocTensorFree(&result_tensor);
            break;
        }
        rocquantum::util::rocTensorFree(&result_tensor);
    }
    auto end = std::chrono::steady_clock::now();
    const unsigned completed = result.status == 0 ? trials : 1;
    result.ms_per_trial =
        std::chrono::duration<double, std::milli>(end - start).count() / static_cast<double>(completed);

    rocquantum::util::rocTensorFree(&a);
    rocquantum::util::rocTensorFree(&b);
    rocTensorNetworkDestroy(tn);
    return result;
}

void write_json(std::ostream& out, unsigned trials, const std::vector<CaseResult>& results) {
    out << "{\n"
        << "  \"benchmark\": \"tensornet_contraction\",\n"
        << "  \"trials\": " << trials << ",\n"
        << "  \"cases\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const CaseResult& r = results[i];
        out << "    {\"name\": \"" << r.name << "\", \"status\": " << r.status
            << ", \"ms_per_trial\": " << r.ms_per_trial << "}";
        out << (i + 1 == results.size() ? "\n" : ",\n");
    }
    out << "  ]\n}\n";
}

}  // namespace

int main(int argc, char** argv) {
    unsigned trials = 20;
    std::string output;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--trials" && i + 1 < argc) {
            trials = static_cast<unsigned>(std::stoul(argv[++i]));
        } else if (arg == "--output" && i + 1 < argc) {
            output = argv[++i];
        }
    }

    if (trials == 0) {
        std::cerr << "trials must be positive\n";
        return 1;
    }

    hipStream_t stream = nullptr;
    rocblas_handle blas = nullptr;
    if (!check_hip(hipStreamCreate(&stream), "hipStreamCreate") ||
        !check_rocblas(rocblas_create_handle(&blas), "rocblas_create_handle") ||
        !check_rocblas(rocblas_set_stream(blas, stream), "rocblas_set_stream")) {
        if (blas) {
            rocblas_destroy_handle(blas);
        }
        if (stream) {
            hipStreamDestroy(stream);
        }
        return 1;
    }

    std::vector<CaseResult> results;
    results.push_back(run_contraction_case("greedy_no_limit", trials, 0, 0, blas, stream));
    results.push_back(run_contraction_case("greedy_memory_limit_planning", trials, 1024 * 1024, 2, blas, stream));

    std::ostream* out = &std::cout;
    std::ofstream file;
    if (!output.empty()) {
        file.open(output);
        if (!file) {
            std::cerr << "failed to open output file: " << output << "\n";
            rocblas_destroy_handle(blas);
            hipStreamDestroy(stream);
            return 1;
        }
        out = &file;
    }

    write_json(*out, trials, results);

    bool any_success = false;
    for (const CaseResult& result : results) {
        any_success = any_success || result.status == 0;
    }

    rocblas_destroy_handle(blas);
    hipStreamDestroy(stream);
    return any_success ? 0 : 1;
}
