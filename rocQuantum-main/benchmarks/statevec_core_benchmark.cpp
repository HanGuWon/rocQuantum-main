#include "rocquantum/GateFusion.h"
#include "rocquantum/hipStateVec.h"

#include <hip/hip_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
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

struct DeviceMatrix {
    rocComplex* ptr = nullptr;
    unsigned dim = 0;

    ~DeviceMatrix() {
        if (ptr) {
            hipFree(ptr);
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

std::vector<rocComplex> hadamard_matrix_col_major() {
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    return {
        make_complex(inv_sqrt2),
        make_complex(inv_sqrt2),
        make_complex(inv_sqrt2),
        make_complex(-inv_sqrt2),
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
    double ms_per_trial = 0.0;
};

template <typename Fn>
CaseResult run_timed_case(const std::string& name, unsigned trials, Fn&& fn) {
    CaseResult result{name};
    auto start = std::chrono::steady_clock::now();
    for (unsigned i = 0; i < trials; ++i) {
        if (!fn()) {
            result.status = 1;
            break;
        }
    }
    auto end = std::chrono::steady_clock::now();
    const unsigned completed = result.status == 0 ? trials : 1;
    result.ms_per_trial =
        std::chrono::duration<double, std::milli>(end - start).count() / static_cast<double>(completed);
    return result;
}

bool initialize_state(rocsvHandle_t handle, rocComplex* d_state, unsigned qubits) {
    return check_status(rocsvInitializeState(handle, d_state, qubits), "rocsvInitializeState") &&
           check_status(rocsvApplyH(handle, d_state, qubits, 0), "rocsvApplyH") &&
           check_status(rocsvApplyCNOT(handle, d_state, qubits, 0, 1), "rocsvApplyCNOT") &&
           check_status(rocsvSynchronize(handle), "rocsvSynchronize(init)");
}

void write_json(std::ostream& out,
                unsigned qubits,
                unsigned trials,
                unsigned shots,
                const std::vector<CaseResult>& results) {
    out << "{\n"
        << "  \"benchmark\": \"statevec_core_paths\",\n"
        << "  \"qubits\": " << qubits << ",\n"
        << "  \"trials\": " << trials << ",\n"
        << "  \"shots\": " << shots << ",\n"
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
    unsigned qubits = 20;
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

    if (qubits < 6 || trials == 0 || shots == 0) {
        std::cerr << "qubits must be >= 6 and trials/shots must be positive\n";
        return 1;
    }

    rocsvHandle_t handle = nullptr;
    rocComplex* d_state = nullptr;
    if (!check_status(rocsvCreate(&handle), "rocsvCreate") ||
        !check_status(rocsvAllocateState(handle, qubits, &d_state, 1), "rocsvAllocateState") ||
        !initialize_state(handle, d_state, qubits)) {
        if (handle) {
            rocsvDestroy(handle);
        }
        return 1;
    }

    DeviceMatrix h_matrix;
    DeviceMatrix identity_5q;
    if (!upload_matrix(hadamard_matrix_col_major(), 2, h_matrix) ||
        !upload_matrix(identity_matrix_col_major(32), 32, identity_5q)) {
        rocsvDestroy(handle);
        return 1;
    }

    std::vector<CaseResult> results;

    const unsigned one_target[] = {0};
    results.push_back(run_timed_case("matrix_fastpath_1q", trials, [&]() {
        return check_status(rocsvApplyMatrix(handle, d_state, qubits, one_target, 1, h_matrix.ptr, h_matrix.dim),
                            "rocsvApplyMatrix(1q)") &&
               check_status(rocsvSynchronize(handle), "rocsvSynchronize(matrix_fastpath)");
    }));

    const unsigned fallback_targets[] = {0, 1, 2, 3, 4};
    set_env_var("ROCQ_ALLOW_HOST_MATRIX_FALLBACK", "1");
    results.push_back(run_timed_case("matrix_host_fallback_5q_identity", trials, [&]() {
        return check_status(rocsvApplyMatrix(handle,
                                             d_state,
                                             qubits,
                                             fallback_targets,
                                             5,
                                             identity_5q.ptr,
                                             identity_5q.dim),
                            "rocsvApplyMatrix(5q host fallback)") &&
               check_status(rocsvSynchronize(handle), "rocsvSynchronize(matrix_fallback)");
    }));
    unset_env_var("ROCQ_ALLOW_HOST_MATRIX_FALLBACK");

    results.push_back(run_timed_case("gate_unfused_cnot_rz", trials, [&]() {
        return check_status(rocsvApplyCNOT(handle, d_state, qubits, 0, 1), "rocsvApplyCNOT(unfused)") &&
               check_status(rocsvApplyRz(handle, d_state, qubits, 1, 0.125), "rocsvApplyRz(unfused)") &&
               check_status(rocsvSynchronize(handle), "rocsvSynchronize(unfused)");
    }));

    rocquantum::GateFusion fusion(handle, d_state, qubits);
    rocquantum::GateOp cnot;
    cnot.name = "CNOT";
    cnot.controls = {0};
    cnot.targets = {1};
    rocquantum::GateOp rz;
    rz.name = "RZ";
    rz.targets = {1};
    rz.params = {0.125};
    const std::vector<rocquantum::GateOp> fusion_queue = {cnot, rz};
    results.push_back(run_timed_case("gate_fusion_cnot_rz", trials, [&]() {
        return check_status(fusion.processQueue(fusion_queue), "GateFusion.processQueue") &&
               check_status(rocsvSynchronize(handle), "rocsvSynchronize(fusion)");
    }));

    double expectation = 0.0;
    std::vector<uint64_t> samples(shots, 0);
    const unsigned measured[] = {0, 1, 2};
    results.push_back(run_timed_case("observe_z_and_sample", trials, [&]() {
        return check_status(rocsvGetExpectationValueSinglePauliZ(handle, d_state, qubits, 0, &expectation),
                            "rocsvGetExpectationValueSinglePauliZ") &&
               check_status(rocsvSample(handle, d_state, qubits, measured, 3, shots, samples.data()),
                            "rocsvSample");
    }));

    std::ostream* out = &std::cout;
    std::ofstream file;
    if (!output.empty()) {
        file.open(output);
        if (!file) {
            std::cerr << "failed to open output file: " << output << "\n";
            rocsvDestroy(handle);
            return 1;
        }
        out = &file;
    }

    write_json(*out, qubits, trials, shots, results);

    bool any_success = false;
    for (const CaseResult& result : results) {
        any_success = any_success || result.status == 0;
    }

    rocsvDestroy(handle);
    return any_success ? 0 : 1;
}
