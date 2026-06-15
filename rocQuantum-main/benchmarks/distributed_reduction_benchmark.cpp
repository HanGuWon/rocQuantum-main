#include "rocquantum/hipStateVec.h"

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

void set_env_var(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

void unset_env_var(const char* name) {
#ifdef _WIN32
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

rocComplex make_complex(double real, double imag = 0.0) {
#ifdef ROCQ_PRECISION_DOUBLE
    return rocComplex{real, imag};
#else
    return rocComplex{static_cast<float>(real), static_cast<float>(imag)};
#endif
}

bool check_hip(hipError_t err, const char* what) {
    if (err != hipSuccess) {
        std::cerr << what << " failed: " << hipGetErrorString(err) << "\n";
        return false;
    }
    return true;
}

struct DeviceMatrix {
    rocComplex* ptr = nullptr;
    unsigned dim = 0;

    ~DeviceMatrix() {
        if (ptr) {
            hipFree(ptr);
        }
    }
};

struct DeviceCsr {
    rocComplex* data = nullptr;
    size_t* indices = nullptr;
    size_t* indptr = nullptr;
    size_t rows = 0;
    size_t nnz = 0;

    ~DeviceCsr() {
        if (data) {
            hipFree(data);
        }
        if (indices) {
            hipFree(indices);
        }
        if (indptr) {
            hipFree(indptr);
        }
    }
};

bool upload_matrix(const std::vector<rocComplex>& host, unsigned dim, DeviceMatrix& out) {
    out.dim = dim;
    const size_t bytes = host.size() * sizeof(rocComplex);
    if (!check_hip(hipMalloc(&out.ptr, bytes), "hipMalloc(matrix)")) {
        return false;
    }
    return check_hip(hipMemcpy(out.ptr, host.data(), bytes, hipMemcpyHostToDevice), "hipMemcpy(matrix)");
}

bool upload_rank_local_z_csr(unsigned qubits, DeviceCsr& out) {
    if (qubits >= std::numeric_limits<size_t>::digits) {
        std::cerr << "qubit count is too large for CSR benchmark dimensions\n";
        return false;
    }

    const size_t rows = size_t{1} << qubits;
    std::vector<rocComplex> data(rows);
    std::vector<size_t> indices(rows);
    std::vector<size_t> indptr(rows + 1);
    for (size_t row = 0; row < rows; ++row) {
        data[row] = make_complex((row & size_t{1}) == 0 ? 1.0 : -1.0);
        indices[row] = row;
        indptr[row] = row;
    }
    indptr[rows] = rows;

    out.rows = rows;
    out.nnz = rows;
    if (!check_hip(hipMalloc(&out.data, data.size() * sizeof(rocComplex)), "hipMalloc(csr data)") ||
        !check_hip(hipMalloc(&out.indices, indices.size() * sizeof(size_t)), "hipMalloc(csr indices)") ||
        !check_hip(hipMalloc(&out.indptr, indptr.size() * sizeof(size_t)), "hipMalloc(csr indptr)")) {
        return false;
    }
    return check_hip(hipMemcpy(out.data, data.data(), data.size() * sizeof(rocComplex), hipMemcpyHostToDevice),
                     "hipMemcpy(csr data)") &&
           check_hip(hipMemcpy(out.indices, indices.data(), indices.size() * sizeof(size_t), hipMemcpyHostToDevice),
                     "hipMemcpy(csr indices)") &&
           check_hip(hipMemcpy(out.indptr, indptr.data(), indptr.size() * sizeof(size_t), hipMemcpyHostToDevice),
                     "hipMemcpy(csr indptr)");
}

std::vector<rocComplex> pauli_z_matrix_col_major() {
    return {
        make_complex(1.0),
        make_complex(0.0),
        make_complex(0.0),
        make_complex(-1.0),
    };
}

std::vector<rocComplex> identity_matrix_col_major(unsigned dim) {
    std::vector<rocComplex> matrix(static_cast<size_t>(dim) * dim, make_complex(0.0));
    for (unsigned i = 0; i < dim; ++i) {
        matrix[static_cast<size_t>(i) + static_cast<size_t>(i) * dim] = make_complex(1.0);
    }
    return matrix;
}

struct CaseResult {
    std::string name;
    int status = 0;
    double expectation_ms = 0.0;
    double dense_expectation_ms = 0.0;
    double sparse_moments_ms = 0.0;
    double sampling_ms = 0.0;
    double generic_matrix_ms = 0.0;
};

bool ok(rocqStatus_t status, const char* what) {
    if (status != ROCQ_STATUS_SUCCESS) {
        std::cerr << what << " failed with status " << static_cast<int>(status) << "\n";
        return false;
    }
    return true;
}

CaseResult run_case(const std::string& name,
                    bool force_rccl,
                    unsigned qubits,
                    unsigned trials,
                    unsigned shots) {
    if (force_rccl) {
        set_env_var("ROCQ_DISTRIBUTED_COMM", "rccl");
        unset_env_var("ROCQ_DISABLE_RCCL");
        unset_env_var("ROCQ_DISTRIBUTED_FALLBACK_MODE");
    } else {
        set_env_var("ROCQ_DISABLE_RCCL", "1");
        set_env_var("ROCQ_DISTRIBUTED_FALLBACK_MODE", "host");
        unset_env_var("ROCQ_DISTRIBUTED_COMM");
    }

    CaseResult result{name};
    rocsvHandle_t handle = nullptr;
    if (!ok(rocsvCreate(&handle), "rocsvCreate")) {
        result.status = 1;
        return result;
    }
    if (!ok(rocsvAllocateDistributedState(handle, qubits), "rocsvAllocateDistributedState") ||
        !ok(rocsvInitializeDistributedState(handle), "rocsvInitializeDistributedState") ||
        !ok(rocsvApplyH(handle, nullptr, qubits, 0), "rocsvApplyH") ||
        !ok(rocsvSynchronize(handle), "rocsvSynchronize")) {
        result.status = 1;
        rocsvDestroy(handle);
        return result;
    }

    double expectation = 0.0;
    rocComplex dense_expectation = make_complex(0.0);
    rocComplex sparse_mean = make_complex(0.0);
    rocComplex sparse_second = make_complex(0.0);
    std::vector<uint64_t> samples(shots, 0);
    const unsigned measured[] = {0};
    const unsigned dense_target[] = {0};
    const unsigned generic_targets[] = {0, qubits > 1 ? qubits - 1 : 0};
    DeviceMatrix dense_matrix;
    DeviceMatrix generic_identity;
    DeviceCsr sparse_z;
    if (!upload_matrix(pauli_z_matrix_col_major(), 2, dense_matrix) ||
        !upload_matrix(identity_matrix_col_major(4), 4, generic_identity) ||
        !upload_rank_local_z_csr(qubits, sparse_z)) {
        result.status = 1;
        rocsvDestroy(handle);
        return result;
    }

    (void)rocsvGetExpectationValueSinglePauliZ(handle, nullptr, qubits, 0, &expectation);
    (void)rocsvGetExpectationMatrix(handle,
                                    nullptr,
                                    qubits,
                                    dense_target,
                                    1,
                                    dense_matrix.ptr,
                                    dense_matrix.dim,
                                    &dense_expectation);
    (void)rocsvGetSparseMatrixMoments(handle,
                                      nullptr,
                                      qubits,
                                      sparse_z.data,
                                      sparse_z.indices,
                                      sparse_z.indptr,
                                      sparse_z.rows,
                                      sparse_z.rows,
                                      sparse_z.nnz,
                                      &sparse_mean,
                                      &sparse_second);
    (void)rocsvSample(handle, nullptr, qubits, measured, 1, shots, samples.data());
    (void)rocsvApplyMatrix(handle,
                           nullptr,
                           qubits,
                           generic_targets,
                           2,
                           generic_identity.ptr,
                           generic_identity.dim);
    (void)rocsvSynchronize(handle);

    auto t0 = std::chrono::steady_clock::now();
    for (unsigned i = 0; i < trials; ++i) {
        if (!ok(rocsvGetExpectationValueSinglePauliZ(handle, nullptr, qubits, 0, &expectation),
                "rocsvGetExpectationValueSinglePauliZ")) {
            result.status = 1;
            break;
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    if (result.status == 0) {
        for (unsigned i = 0; i < trials; ++i) {
            if (!ok(rocsvGetExpectationMatrix(handle,
                                              nullptr,
                                              qubits,
                                              dense_target,
                                              1,
                                              dense_matrix.ptr,
                                              dense_matrix.dim,
                                              &dense_expectation),
                    "rocsvGetExpectationMatrix")) {
                result.status = 1;
                break;
            }
        }
    }
    auto t2 = std::chrono::steady_clock::now();
    if (result.status == 0) {
        for (unsigned i = 0; i < trials; ++i) {
            if (!ok(rocsvGetSparseMatrixMoments(handle,
                                                nullptr,
                                                qubits,
                                                sparse_z.data,
                                                sparse_z.indices,
                                                sparse_z.indptr,
                                                sparse_z.rows,
                                                sparse_z.rows,
                                                sparse_z.nnz,
                                                &sparse_mean,
                                                &sparse_second),
                    "rocsvGetSparseMatrixMoments")) {
                result.status = 1;
                break;
            }
        }
    }
    auto t3 = std::chrono::steady_clock::now();
    if (result.status == 0) {
        for (unsigned i = 0; i < trials; ++i) {
            if (!ok(rocsvSample(handle, nullptr, qubits, measured, 1, shots, samples.data()),
                    "rocsvSample")) {
                result.status = 1;
                break;
            }
        }
    }
    auto t4 = std::chrono::steady_clock::now();
    if (result.status == 0) {
        for (unsigned i = 0; i < trials; ++i) {
            if (!ok(rocsvApplyMatrix(handle,
                                     nullptr,
                                     qubits,
                                     generic_targets,
                                     2,
                                     generic_identity.ptr,
                                     generic_identity.dim),
                    "rocsvApplyMatrix(generic distributed identity)") ||
                !ok(rocsvSynchronize(handle), "rocsvSynchronize(generic matrix)")) {
                result.status = 1;
                break;
            }
        }
    }
    auto t5 = std::chrono::steady_clock::now();

    result.expectation_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count() / static_cast<double>(trials);
    result.dense_expectation_ms =
        std::chrono::duration<double, std::milli>(t2 - t1).count() / static_cast<double>(trials);
    result.sparse_moments_ms =
        std::chrono::duration<double, std::milli>(t3 - t2).count() / static_cast<double>(trials);
    result.sampling_ms =
        std::chrono::duration<double, std::milli>(t4 - t3).count() / static_cast<double>(trials);
    result.generic_matrix_ms =
        std::chrono::duration<double, std::milli>(t5 - t4).count() / static_cast<double>(trials);
    rocsvDestroy(handle);
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    unsigned qubits = 24;
    unsigned trials = 20;
    unsigned shots = 4096;
    std::string output;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--qubits" && i + 1 < argc) {
            qubits = static_cast<unsigned>(std::stoul(argv[++i]));
        } else if (arg == "--trials" && i + 1 < argc) {
            trials = static_cast<unsigned>(std::stoul(argv[++i]));
        } else if (arg == "--shots" && i + 1 < argc) {
            shots = static_cast<unsigned>(std::stoul(argv[++i]));
        } else if (arg == "--output" && i + 1 < argc) {
            output = argv[++i];
        }
    }

    if (trials == 0 || shots == 0) {
        std::cerr << "trials and shots must be positive\n";
        return 1;
    }

    std::vector<CaseResult> results;
    results.push_back(run_case("rccl", true, qubits, trials, shots));
    results.push_back(run_case("host_fallback", false, qubits, trials, shots));

    std::ostream* out = &std::cout;
    std::ofstream file;
    if (!output.empty()) {
        file.open(output);
        if (!file) {
            std::cerr << "failed to open output file: " << output << "\n";
            return 1;
        }
        out = &file;
    }

    *out << "{\n"
         << "  \"qubits\": " << qubits << ",\n"
         << "  \"trials\": " << trials << ",\n"
         << "  \"shots\": " << shots << ",\n"
         << "  \"cases\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const CaseResult& r = results[i];
        *out << "    {\"name\": \"" << r.name << "\", \"status\": " << r.status
             << ", \"expectation_ms\": " << r.expectation_ms
             << ", \"dense_expectation_ms\": " << r.dense_expectation_ms
             << ", \"sparse_moments_ms\": " << r.sparse_moments_ms
             << ", \"sampling_ms\": " << r.sampling_ms
             << ", \"generic_matrix_ms\": " << r.generic_matrix_ms << "}";
        *out << (i + 1 == results.size() ? "\n" : ",\n");
    }
    *out << "  ]\n}\n";

    return (results[0].status == 0 || results[1].status == 0) ? 0 : 1;
}
