#include "rocqCompiler/MLIRCompiler.h"

#include <complex>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
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

const char* kUnsupportedMcxModule = R"mlir(
module {
  func.func @bad() {
    %q0, %q1 = "quantum.qalloc"() {size = 2 : i64} : () -> (!quantum.qubit, !quantum.qubit)
    "quantum.mcx"(%q0, %q1) : (!quantum.qubit, !quantum.qubit) -> ()
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

        requireThrowsContaining(
            [&] { offline.emit_qir(kUnsupportedMcxModule); }, "mcx");
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
            throw std::runtime_error("RecordingBackend dispatch contract failed");
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
                "RecordingBackend parameter dispatch contract failed");
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
