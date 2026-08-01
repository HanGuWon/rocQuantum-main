#include "rocqCompiler/MLIRCompiler.h"
#include "rocqCompiler/ReferenceStateVecBackend.h"

#include <atomic>
#include <cmath>
#include <complex>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

std::string formatTargets(const std::vector<unsigned>& targets) {
    std::string result;
    for (const auto target : targets) {
        if (!result.empty()) {
            result += ',';
        }
        result += std::to_string(target);
    }
    return result;
}

class RecordingBackend final : public rocq::QuantumBackend {
public:
    void initialize(unsigned num_qubits) override {
        initialized_qubits = num_qubits;
        events.clear();
    }

    void apply_gate(const std::string& name,
                    const std::vector<unsigned>& targets) override {
        events.push_back(name + ":" + formatTargets(targets));
    }

    void apply_parametrized_gate(const std::string& name,
                                 double parameter,
                                 const std::vector<unsigned>& targets) override {
        events.push_back(name + ":" + std::to_string(parameter) + ":" +
                         formatTargets(targets));
    }

    std::vector<std::complex<double>> get_state_vector() override {
        return {{1.0, 0.0}, {0.0, 0.0}};
    }

    void destroy() override {}

    unsigned initialized_qubits = 0;
    std::vector<std::string> events;
};

class PartiallyInitializingBackend final : public rocq::QuantumBackend {
public:
    void initialize(unsigned) override {
        ++initialize_calls;
        throw std::runtime_error("synthetic backend initialization failure");
    }
    void apply_gate(const std::string&, const std::vector<unsigned>&) override {}
    void apply_parametrized_gate(
        const std::string&, double, const std::vector<unsigned>&) override {}
    std::vector<std::complex<double>> get_state_vector() override {
        return {};
    }
    void destroy() override {
        ++destroy_calls;
    }

    unsigned initialize_calls = 0;
    unsigned destroy_calls = 0;
};

class ThrowingGateBackend final : public rocq::QuantumBackend {
public:
    void initialize(unsigned) override {}
    void apply_gate(
        const std::string&, const std::vector<unsigned>&) override {
        throw std::runtime_error("synthetic QIS callback failure");
    }
    void apply_parametrized_gate(
        const std::string&, double, const std::vector<unsigned>&) override {}
    std::vector<std::complex<double>> get_state_vector() override {
        return {};
    }
    void destroy() override {
        ++destroy_calls;
    }

    unsigned destroy_calls = 0;
};

const char* kBellModule = R"mlir(
module {
  func.func @bell() {
    %q0, %q1 = "quantum.qalloc"() {size = 2 : i64} : () -> (!quantum.qubit, !quantum.qubit)
    "quantum.h"(%q0) : (!quantum.qubit) -> ()
    "quantum.cnot"(%q0, %q1) : (!quantum.qubit, !quantum.qubit) -> ()
    return
  }
}
)mlir";

const char* kRotationModule = R"mlir(
module {
  func.func @rotation() {
    %q0 = "quantum.qalloc"() {size = 1 : i64} : () -> !quantum.qubit
    "quantum.rx"(%q0) {angle = 5.000000e-01 : f64} : (!quantum.qubit) -> ()
    return
  }
}
)mlir";

const char* kCzModule = R"mlir(
module {
  func.func @cz_decomposition() {
    %q0, %q1 = "quantum.qalloc"() {size = 2 : i64} : () -> (!quantum.qubit, !quantum.qubit)
    "quantum.cz"(%q0, %q1) : (!quantum.qubit, !quantum.qubit) -> ()
    return
  }
}
)mlir";

const char* kTwoOperandMcxModule = R"mlir(
module {
  func.func @mcx_one_control() {
    %q0, %q1 = "quantum.qalloc"() {size = 2 : i64} : () -> (!quantum.qubit, !quantum.qubit)
    "quantum.mcx"(%q0, %q1) : (!quantum.qubit, !quantum.qubit) -> ()
    return
  }
}
)mlir";

const char* kThreeOperandMcxModule = R"mlir(
module {
  func.func @mcx_two_controls() {
    %q0, %q1, %q2 = "quantum.qalloc"() {size = 3 : i64} : () -> (!quantum.qubit, !quantum.qubit, !quantum.qubit)
    "quantum.x"(%q0) : (!quantum.qubit) -> ()
    "quantum.x"(%q1) : (!quantum.qubit) -> ()
    "quantum.mcx"(%q0, %q1, %q2) : (!quantum.qubit, !quantum.qubit, !quantum.qubit) -> ()
    return
  }
}
)mlir";

const char* kUnsupportedWideMcxModule = R"mlir(
module {
  func.func @wide_mcx() {
    %q0, %q1, %q2, %q3 = "quantum.qalloc"() {size = 4 : i64} : () -> (!quantum.qubit, !quantum.qubit, !quantum.qubit, !quantum.qubit)
    "quantum.mcx"(%q0, %q1, %q2, %q3) : (!quantum.qubit, !quantum.qubit, !quantum.qubit, !quantum.qubit) -> ()
    return
  }
}
)mlir";

const char* kControlledRotationModule = R"mlir(
module {
  func.func @controlled_rotation() {
    %q0, %q1 = "quantum.qalloc"() {size = 2 : i64} : () -> (!quantum.qubit, !quantum.qubit)
    "quantum.crx"(%q0, %q1) {angle = 5.000000e-01 : f64} : (!quantum.qubit, !quantum.qubit) -> ()
    return
  }
}
)mlir";

const char* kDuplicateTargetModule = R"mlir(
module {
  func.func @duplicate_target() {
    %q0 = "quantum.qalloc"() {size = 1 : i64} : () -> !quantum.qubit
    "quantum.cnot"(%q0, %q0) : (!quantum.qubit, !quantum.qubit) -> ()
    return
  }
}
)mlir";

const char* kMismatchedAllocationModule = R"mlir(
module {
  func.func @bad_alloc() {
    %q0 = "quantum.qalloc"() {size = 2 : i64} : () -> !quantum.qubit
    return
  }
}
)mlir";

const char* kBaseProfileModule = R"mlir(
module {
  func.func @base_profile() {
    %q0, %q1 = "quantum.qalloc"() {size = 2 : i64} : () -> (!quantum.qubit, !quantum.qubit)
    "quantum.h"(%q0) : (!quantum.qubit) -> ()
    "quantum.cnot"(%q0, %q1) : (!quantum.qubit, !quantum.qubit) -> ()
    %r0 = "quantum.mz"(%q0) {registerName = "r0"} : (!quantum.qubit) -> !quantum.result
    %r1 = "quantum.mz"(%q1) {registerName = "r1"} : (!quantum.qubit) -> !quantum.result
    return
  }
}
)mlir";

const char* kBaseProfileWithoutMeasurement = R"mlir(
module {
  func.func @no_measurement() {
    %q0 = "quantum.qalloc"() {size = 1 : i64} : () -> !quantum.qubit
    "quantum.h"(%q0) : (!quantum.qubit) -> ()
    return
  }
}
)mlir";

const char* kDuplicateResultLabelModule = R"mlir(
module {
  func.func @duplicate_label() {
    %q0, %q1 = "quantum.qalloc"() {size = 2 : i64} : () -> (!quantum.qubit, !quantum.qubit)
    %r0 = "quantum.mz"(%q0) {registerName = "same"} : (!quantum.qubit) -> !quantum.result
    %r1 = "quantum.mz"(%q1) {registerName = "same"} : (!quantum.qubit) -> !quantum.result
    return
  }
}
)mlir";

const char* kEmptyResultLabelModule = R"mlir(
module {
  func.func @empty_label() {
    %q0 = "quantum.qalloc"() {size = 1 : i64} : () -> !quantum.qubit
    %r0 = "quantum.mz"(%q0) {registerName = ""} : (!quantum.qubit) -> !quantum.result
    return
  }
}
)mlir";

const char* kGateAfterMeasurementModule = R"mlir(
module {
  func.func @gate_after_measurement() {
    %q0, %q1 = "quantum.qalloc"() {size = 2 : i64} : () -> (!quantum.qubit, !quantum.qubit)
    %r0 = "quantum.mz"(%q0) {registerName = "r0"} : (!quantum.qubit) -> !quantum.result
    "quantum.x"(%q1) : (!quantum.qubit) -> ()
    return
  }
}
)mlir";

const char* kReusedMeasurementResultModule = R"mlir(
module {
  func.func @used_result() {
    %q0 = "quantum.qalloc"() {size = 1 : i64} : () -> !quantum.qubit
    %r0 = "quantum.mz"(%q0) {registerName = "r0"} : (!quantum.qubit) -> !quantum.result
    %b = builtin.unrealized_conversion_cast %r0 : !quantum.result to i1
    return
  }
}
)mlir";

void requireContains(const std::string& text, const std::string& needle) {
    if (text.find(needle) == std::string::npos) {
        throw std::runtime_error("expected output to contain: " + needle);
    }
}

void requireNotContains(const std::string& text, const std::string& needle) {
    if (text.find(needle) != std::string::npos) {
        throw std::runtime_error("expected output not to contain: " + needle);
    }
}

template <typename Callback>
void requireThrowsContaining(Callback&& callback, const std::string& needle) {
    try {
        callback();
    } catch (const std::exception& error) {
        if (std::string(error.what()).find(needle) != std::string::npos) {
            return;
        }
        throw std::runtime_error(
            "exception did not contain '" + needle + "': " + error.what());
    }
    throw std::runtime_error("expected an exception containing: " + needle);
}

} // namespace

int main() {
    try {
        rocq::MLIRCompiler offline(/*num_qubits=*/2);
        const std::string qir = offline.emit_qir(kBellModule);
        requireContains(qir, "__quantum__qis__h__body");
        requireContains(qir, "__quantum__qis__cnot__body");
        requireContains(qir, "qir_major_version");
        requireContains(qir, "required_num_qubits\"=\"2");
        requireContains(qir, "required_num_results\"=\"0");
        requireContains(qir, "entry_point");
        requireContains(qir, "qir_profiles\"=\"custom");
        requireContains(qir, "qir_minor_version");
        requireContains(qir, "dynamic_qubit_management");
        requireContains(qir, "dynamic_result_management");
        if (qir.find("Debug Info Version") != std::string::npos) {
            throw std::runtime_error("QIR unexpectedly retained a debug module flag");
        }

        rocq::MLIRCompiler rotation(/*num_qubits=*/1);
        const std::string rotation_qir = rotation.emit_qir(kRotationModule);
        requireContains(rotation_qir, "__quantum__qis__rx__body");
        requireContains(rotation_qir, "double 5.000000e-01");

        const std::string controlled_qir = offline.emit_qir(kControlledRotationModule);
        requireContains(controlled_qir, "__quantum__qis__h__body");
        requireContains(controlled_qir, "__quantum__qis__rz__body");
        requireContains(controlled_qir, "__quantum__qis__cnot__body");
        if (controlled_qir.find("__quantum__qis__crx") != std::string::npos) {
            throw std::runtime_error("CRX was emitted through a nonstandard direct QIS ABI");
        }

        const std::string base_qir =
            offline.emit_qir(kBaseProfileModule, "qir-v2-base");
        requireContains(base_qir, "define i64 @base_profile()");
        requireContains(base_qir, "__quantum__rt__initialize(ptr null)");
        requireContains(base_qir, "__quantum__qis__mz__body");
        requireContains(base_qir, "ptr writeonly");
        requireContains(base_qir, "\"irreversible\"");
        requireContains(base_qir, "__quantum__rt__result_record_output");
        requireContains(base_qir, "qir_profiles\"=\"base_profile");
        requireContains(base_qir, "output_labeling_schema\"=\"schema_id");
        requireContains(base_qir, "required_num_results\"=\"2");
        requireContains(base_qir, "measurements:");
        requireContains(base_qir, "output:");
        requireContains(base_qir, "ret i64 0");
        requireNotContains(base_qir, "qir_profiles\"=\"custom");

        requireThrowsContaining(
            [&] { offline.emit_qir(kBaseProfileModule); }, "qir-v2-base");
        requireThrowsContaining(
            [&] {
                offline.emit_qir(
                    kBaseProfileWithoutMeasurement, "qir-v2-base");
            },
            "at least one terminal quantum.mz");
        requireThrowsContaining(
            [&] {
                offline.emit_qir(kDuplicateResultLabelModule, "qir-v2-base");
            },
            "duplicate QIR output label");
        requireThrowsContaining(
            [&] {
                offline.emit_qir(kEmptyResultLabelModule, "qir-v2-base");
            },
            "non-empty, NUL-free registerName");
        requireThrowsContaining(
            [&] {
                offline.emit_qir(kGateAfterMeasurementModule, "qir-v2-base");
            },
            "precede all measurements");
        requireThrowsContaining(
            [&] {
                offline.emit_qir(kReusedMeasurementResultModule, "qir-v2-base");
            },
            "cannot drive classical or quantum operations");

        const std::string one_control_mcx_qir =
            offline.emit_qir(kTwoOperandMcxModule);
        requireContains(one_control_mcx_qir, "__quantum__qis__cnot__body");
        requireNotContains(one_control_mcx_qir, "__quantum__qis__mcx");

        rocq::MLIRCompiler three_qubit_offline(/*num_qubits=*/3);
        const std::string two_control_mcx_qir =
            three_qubit_offline.emit_qir(kThreeOperandMcxModule);
        requireContains(two_control_mcx_qir, "__quantum__qis__h__body");
        requireContains(two_control_mcx_qir, "__quantum__qis__t__body");
        requireContains(two_control_mcx_qir, "__quantum__qis__cnot__body");
        requireNotContains(two_control_mcx_qir, "__quantum__qis__mcx");

        rocq::MLIRCompiler four_qubit_offline(/*num_qubits=*/4);
        requireThrowsContaining(
            [&] { four_qubit_offline.emit_qir(kUnsupportedWideMcxModule); },
            "control-array lowering");
        requireThrowsContaining(
            [&] { offline.emit_qir(kDuplicateTargetModule); }, "distinct");
        requireThrowsContaining(
            [&] { offline.emit_qir(kMismatchedAllocationModule); }, "result count");
        requireThrowsContaining(
            [&] { offline.emit_qir(kBellModule, "nvqir-full-0.1"); },
            "qir-v2-static");

        auto backend = std::make_unique<RecordingBackend>();
        auto* recording = backend.get();
        rocq::MLIRCompiler executor(/*num_qubits=*/2, std::move(backend));
        const auto state = executor.compile_and_execute(kBellModule, {});
        if (recording->initialized_qubits != 2 || recording->events.size() != 2 ||
            recording->events[0] != "h:0" || recording->events[1] != "cnot:0,1" ||
            state.size() != 2) {
            throw std::runtime_error("QIR JIT RecordingBackend dispatch contract failed");
        }

        auto rotation_backend = std::make_unique<RecordingBackend>();
        auto* rotation_recording = rotation_backend.get();
        rocq::MLIRCompiler rotation_executor(
            /*num_qubits=*/1, std::move(rotation_backend));
        rotation_executor.compile_and_execute(kRotationModule, {{"strict", true}});
        if (rotation_recording->initialized_qubits != 1 ||
            rotation_recording->events.size() != 1 ||
            rotation_recording->events[0] != "rx:0.500000:0") {
            throw std::runtime_error(
                "QIR JIT RecordingBackend parameter dispatch contract failed");
        }

        auto decomposition_backend = std::make_unique<RecordingBackend>();
        auto* decomposition_recording = decomposition_backend.get();
        rocq::MLIRCompiler decomposition_executor(
            /*num_qubits=*/2, std::move(decomposition_backend));
        decomposition_executor.compile_and_execute(kCzModule, {});
        if (decomposition_recording->events !=
            std::vector<std::string>{"h:1", "cnot:0,1", "h:1"}) {
            throw std::runtime_error(
                "compile_and_execute bypassed the QIR CZ decomposition");
        }

        auto mcx_backend = std::make_unique<RecordingBackend>();
        auto* mcx_recording = mcx_backend.get();
        rocq::MLIRCompiler mcx_executor(
            /*num_qubits=*/2, std::move(mcx_backend));
        mcx_executor.compile_and_execute(kTwoOperandMcxModule, {});
        if (mcx_recording->events !=
            std::vector<std::string>{"cnot:0,1"}) {
            throw std::runtime_error(
                "one-control MCX did not lower through the QIR CNOT body");
        }

        rocq::MLIRCompiler numerical_executor(
            /*num_qubits=*/2, rocq::create_reference_backend());
        const auto bell_state =
            numerical_executor.compile_and_execute(kBellModule, {});
        const auto bell_amplitude = 1.0 / std::sqrt(2.0);
        if (bell_state.size() != 4 ||
            std::abs(bell_state[0] - std::complex<double>{bell_amplitude, 0.0}) >
                1.0e-12 ||
            std::abs(bell_state[1]) > 1.0e-12 ||
            std::abs(bell_state[2]) > 1.0e-12 ||
            std::abs(bell_state[3] - std::complex<double>{bell_amplitude, 0.0}) >
                1.0e-12) {
            throw std::runtime_error(
                "QIR JIT CPU reference-backend Bell state is incorrect");
        }

        rocq::MLIRCompiler mcx_numerical_executor(
            /*num_qubits=*/3, rocq::create_reference_backend());
        const auto mcx_state = mcx_numerical_executor.compile_and_execute(
            kThreeOperandMcxModule, {});
        if (mcx_state.size() != 8 ||
            std::abs(mcx_state[7] - std::complex<double>{1.0, 0.0}) >
                1.0e-12) {
            throw std::runtime_error(
                "two-control MCX QIR decomposition produced the wrong state");
        }

        requireThrowsContaining(
            [&] { executor.compile_and_execute(kBaseProfileModule, {}); },
            "qir-v2-base");

        auto failing_backend = std::make_unique<PartiallyInitializingBackend>();
        auto* failing_backend_observer = failing_backend.get();
        rocq::MLIRCompiler failing_executor(
            /*num_qubits=*/2, std::move(failing_backend));
        requireThrowsContaining(
            [&] { failing_executor.compile_and_execute(kBellModule, {}); },
            "synthetic backend initialization failure");
        if (failing_backend_observer->initialize_calls != 1 ||
            failing_backend_observer->destroy_calls != 2) {
            throw std::runtime_error(
                "partially initialized backend was not deterministically cleaned up");
        }

        auto throwing_backend = std::make_unique<ThrowingGateBackend>();
        auto* throwing_backend_observer = throwing_backend.get();
        rocq::MLIRCompiler throwing_executor(
            /*num_qubits=*/2, std::move(throwing_backend));
        requireThrowsContaining(
            [&] { throwing_executor.compile_and_execute(kBellModule, {}); },
            "synthetic QIS callback failure");
        if (throwing_backend_observer->destroy_calls != 2) {
            throw std::runtime_error(
                "backend was not cleaned up after a QIS callback failure");
        }

        std::atomic<unsigned> ready_threads{0};
        std::atomic<bool> start_threads{false};
        std::exception_ptr first_thread_failure;
        std::exception_ptr second_thread_failure;
        std::vector<std::complex<double>> first_thread_state;
        std::vector<std::complex<double>> second_thread_state;
        auto run_concurrent = [&](std::vector<std::complex<double>>& result,
                                  std::exception_ptr& failure) {
            ++ready_threads;
            while (!start_threads.load()) {
                std::this_thread::yield();
            }
            try {
                rocq::MLIRCompiler concurrent_executor(
                    /*num_qubits=*/2, rocq::create_reference_backend());
                result =
                    concurrent_executor.compile_and_execute(kBellModule, {});
            } catch (...) {
                failure = std::current_exception();
            }
        };
        std::thread first_thread(
            run_concurrent,
            std::ref(first_thread_state),
            std::ref(first_thread_failure));
        std::thread second_thread(
            run_concurrent,
            std::ref(second_thread_state),
            std::ref(second_thread_failure));
        while (ready_threads.load() != 2) {
            std::this_thread::yield();
        }
        start_threads = true;
        first_thread.join();
        second_thread.join();
        if (first_thread_failure) {
            std::rethrow_exception(first_thread_failure);
        }
        if (second_thread_failure) {
            std::rethrow_exception(second_thread_failure);
        }
        if (first_thread_state != bell_state ||
            second_thread_state != bell_state) {
            throw std::runtime_error(
                "concurrent QIR JIT executions crossed backend contexts");
        }

        requireThrowsContaining(
            [&] { executor.compile_and_execute(kBellModule, {{"kernel_argument", true}}); },
            "kernel argument binding");

        std::cout << "rocq compiler smoke tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
