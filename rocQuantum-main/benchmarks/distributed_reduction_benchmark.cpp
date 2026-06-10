#include "rocquantum/hipStateVec.h"

#include <chrono>
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

struct CaseResult {
    std::string name;
    int status = 0;
    double expectation_ms = 0.0;
    double sampling_ms = 0.0;
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
    std::vector<uint64_t> samples(shots, 0);
    const unsigned measured[] = {0};

    (void)rocsvGetExpectationValueSinglePauliZ(handle, nullptr, qubits, 0, &expectation);
    (void)rocsvSample(handle, nullptr, qubits, measured, 1, shots, samples.data());

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
            if (!ok(rocsvSample(handle, nullptr, qubits, measured, 1, shots, samples.data()),
                    "rocsvSample")) {
                result.status = 1;
                break;
            }
        }
    }
    auto t2 = std::chrono::steady_clock::now();

    result.expectation_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count() / static_cast<double>(trials);
    result.sampling_ms =
        std::chrono::duration<double, std::milli>(t2 - t1).count() / static_cast<double>(trials);
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
             << ", \"sampling_ms\": " << r.sampling_ms << "}";
        *out << (i + 1 == results.size() ? "\n" : ",\n");
    }
    *out << "  ]\n}\n";

    return (results[0].status == 0 || results[1].status == 0) ? 0 : 1;
}
