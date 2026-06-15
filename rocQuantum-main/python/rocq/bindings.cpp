#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "rocquantum/hipStateVec.h"
#include "rocquantum/hipTensorNet.h" // Include new header
#include "rocquantum/hipTensorNet_api.h"
#include <algorithm>
#include <cctype>
#include <complex>                 // For std::complex
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

// Helper to map NumPy dtype to rocDataType_t enum
rocDataType_t get_rocq_dtype_from_numpy(py::dtype dt) {
    if (dt.is(py::dtype::of<float>())) return ROC_DATATYPE_F32;
    if (dt.is(py::dtype::of<double>())) return ROC_DATATYPE_F64;
    if (dt.is(py::dtype::of<std::complex<float>>())) return ROC_DATATYPE_C64;
    if (dt.is(py::dtype::of<std::complex<double>>())) return ROC_DATATYPE_C128;
    throw std::runtime_error("Unsupported NumPy data type for TensorNetwork");
}

std::string rocq_status_name(rocqStatus_t status) {
    switch (status) {
        case ROCQ_STATUS_SUCCESS:
            return "ROCQ_STATUS_SUCCESS";
        case ROCQ_STATUS_FAILURE:
            return "ROCQ_STATUS_FAILURE";
        case ROCQ_STATUS_INVALID_VALUE:
            return "ROCQ_STATUS_INVALID_VALUE";
        case ROCQ_STATUS_ALLOCATION_FAILED:
            return "ROCQ_STATUS_ALLOCATION_FAILED";
        case ROCQ_STATUS_HIP_ERROR:
            return "ROCQ_STATUS_HIP_ERROR";
        case ROCQ_STATUS_NOT_IMPLEMENTED:
            return "ROCQ_STATUS_NOT_IMPLEMENTED";
        case ROCQ_STATUS_RCCL_ERROR:
            return "ROCQ_STATUS_RCCL_ERROR";
        default:
            return std::string("ROCQ_STATUS_UNKNOWN(") + std::to_string(static_cast<int>(status)) + ")";
    }
}

std::string tensornet_pathfinder_name(rocPathfinderAlgorithm_t algorithm) {
    switch (algorithm) {
        case ROCTN_PATHFINDER_ALGO_GREEDY:
            return "greedy";
        case ROCTN_PATHFINDER_ALGO_KAHYPAR:
            return "kahypar";
        case ROCTN_PATHFINDER_ALGO_METIS:
            return "metis";
        default:
            return std::string("unknown(") + std::to_string(static_cast<int>(algorithm)) + ")";
    }
}

class ConceptualMLIRCompiler {
public:
    bool initialize_module(const std::string& module_name) {
        module_name_ = module_name;
        if (module_string_.empty()) {
            module_string_ = "module {\n}";
        }
        return true;
    }

    void dump_module() const {}

    std::string get_module_string() const {
        return module_string_;
    }

    bool load_module_from_string(const std::string& mlir_string) {
        module_string_ = mlir_string;
        return true;
    }

    bool run_adjoint_generation_pass() const {
        return false;
    }

    bool create_function(const std::string& func_name,
                         const std::vector<std::string>& arg_type_strs,
                         const std::vector<std::string>& result_type_strs) {
        (void)arg_type_strs;
        (void)result_type_strs;
        if (module_string_.empty()) {
            initialize_module(module_name_.empty() ? "rocq_module" : module_name_);
        }
        module_string_ += "\n// conceptual function declaration: " + func_name;
        return true;
    }

private:
    std::string module_name_;
    std::string module_string_;
};

bool tensornet_pathfinder_supported(const hipTensorNetCapabilities_t& caps,
                                    rocPathfinderAlgorithm_t algorithm) {
    switch (algorithm) {
        case ROCTN_PATHFINDER_ALGO_GREEDY:
            return caps.supports_pathfinder_greedy != 0;
        case ROCTN_PATHFINDER_ALGO_KAHYPAR:
            return caps.supports_pathfinder_kahypar != 0;
        case ROCTN_PATHFINDER_ALGO_METIS:
            return caps.supports_pathfinder_metis != 0;
        default:
            return true;
    }
}

std::string tensornet_status_message(const std::string& operation, rocqStatus_t status) {
    std::string message = operation + " failed with " + rocq_status_name(status);
    if (status == ROCQ_STATUS_NOT_IMPLEMENTED) {
        message +=
            ". This TensorNet path is not available in the current build; "
            "call get_tensornet_capabilities() to check compiled dtype, "
            "pathfinder, and runtime slicing support.";
    } else if (status == ROCQ_STATUS_INVALID_VALUE) {
        message += ". The TensorNet handle, tensor, dtype, or optimizer configuration is invalid.";
    } else if (status == ROCQ_STATUS_ALLOCATION_FAILED) {
        message += ". TensorNet device or host allocation failed.";
    }
    return message;
}

std::string tensornet_contract_status_message(
    rocqStatus_t status,
    const hipTensorNetContractionOptimizerConfig_t& config) {
    std::string message = tensornet_status_message("rocTensorNetworkContract", status);
    message += std::string(" pathfinder_algorithm=") +
               tensornet_pathfinder_name(config.pathfinder_algorithm) + ".";
    if (status == ROCQ_STATUS_NOT_IMPLEMENTED) {
        message += " The requested TensorNet functionality is unavailable in this build or layout; "
                   "memory_limit_bytes and num_slices support limited runtime K-sliced GEMM execution "
                   "but do not provide full cuTensorNet-style open-index slicing.";
    }
    if (status == ROCQ_STATUS_INVALID_VALUE && config.num_slices < 0) {
        message += " num_slices must be non-negative.";
    }
    return message;
}

void warn_tensornet_pathfinder_fallback(const hipTensorNetContractionOptimizerConfig_t& config) {
    if (config.pathfinder_algorithm == ROCTN_PATHFINDER_ALGO_GREEDY) {
        return;
    }

    hipTensorNetCapabilities_t caps{};
    rocqStatus_t status = rocTensorNetworkGetCapabilities(&caps);
    if (status != ROCQ_STATUS_SUCCESS) {
        throw std::runtime_error(tensornet_status_message("rocTensorNetworkGetCapabilities", status));
    }
    if (tensornet_pathfinder_supported(caps, config.pathfinder_algorithm)) {
        return;
    }

    const std::string message =
        "TensorNet pathfinder_algorithm=" +
        tensornet_pathfinder_name(config.pathfinder_algorithm) +
        " is not available in this build; falling back to greedy. "
        "Check get_tensornet_capabilities() before relying on METIS/KAHYPAR parity.";
    if (PyErr_WarnEx(PyExc_RuntimeWarning, message.c_str(), 1) != 0) {
        throw py::error_already_set();
    }
}

void warn_tensornet_limited_runtime_slicing(const hipTensorNetContractionOptimizerConfig_t& config) {
    if (config.memory_limit_bytes == 0 && config.num_slices <= 0) {
        return;
    }

    const std::string message =
        "TensorNet memory_limit_bytes and num_slices enable limited runtime "
        "K-sliced GEMM execution for pair contractions; this is not full "
        "cuTensorNet-style open-index slicing. Check "
        "get_tensornet_capabilities()['supports_runtime_slicing'] before relying "
        "on broader sliced-execution parity.";
    if (PyErr_WarnEx(PyExc_RuntimeWarning, message.c_str(), 1) != 0) {
        throw py::error_already_set();
    }
}


// Helper to convert py::array_t<rocComplex> (NumPy array from Python) to rocComplex* on device
// This is a simplified helper. Error handling and memory management need care.
// The caller is responsible for freeing d_matrix if it's allocated by this helper.
// For rocsvApplyMatrix, matrixDevice is already on device, so this helper is for
// a scenario where Python provides a host matrix that needs to be on device.
// However, the rocsvApplyMatrix C-API now expects matrixDevice to *already* be on device.
// So, the Python side will need to manage this (e.g. have a function to create device matrix from numpy).

// Let's define a simple DeviceMemory class for Python to manage GPU buffers for matrices
class DeviceBuffer {
public:
    void* ptr_ = nullptr;
    size_t size_bytes_ = 0;
    bool owned_ = true; // Does this wrapper own the memory (i.e., should it free it)?

    DeviceBuffer() = default;

    DeviceBuffer(size_t num_elements, size_t element_size) : size_bytes_(num_elements * element_size) {
        if (hipMalloc(&ptr_, size_bytes_) != hipSuccess) {
            throw std::runtime_error("Failed to allocate device memory in DeviceBuffer constructor");
        }
    }

    // Constructor to wrap an existing device pointer (e.g., d_state)
    // This wrapper does NOT own the memory.
    DeviceBuffer(void* existing_ptr, size_t size_bytes, bool take_ownership = false) 
        : ptr_(existing_ptr), size_bytes_(size_bytes), owned_(take_ownership) {}


    ~DeviceBuffer() {
        if (owned_ && ptr_) {
            hipFree(ptr_);
        }
    }

    // Disable copy constructor and assignment to prevent double free / shallow copies of owned memory
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    // Allow move construction and assignment
    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.ptr_), size_bytes_(other.size_bytes_), owned_(other.owned_) {
        other.ptr_ = nullptr;
        other.size_bytes_ = 0;
        other.owned_ = false; // Transferred ownership
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (owned_ && ptr_) {
                hipFree(ptr_);
            }
            ptr_ = other.ptr_;
            size_bytes_ = other.size_bytes_;
            owned_ = other.owned_;
            other.ptr_ = nullptr;
            other.size_bytes_ = 0;
            other.owned_ = false;
        }
        return *this;
    }

    void copy_from_numpy(py::array_t<rocComplex, py::array::c_style | py::array::forcecast> np_array) {
        if (!ptr_ || np_array.nbytes() > size_bytes_) {
            throw std::runtime_error("Device buffer not allocated, null, or NumPy array too large.");
        }
        if (hipMemcpy(ptr_, np_array.data(), np_array.nbytes(), hipMemcpyHostToDevice) != hipSuccess) {
            throw std::runtime_error("Failed to copy NumPy array to device");
        }
    }
    
    // Method to get the raw pointer (e.g., rocComplex*)
    template<typename T>
    T* get_ptr() const {
        return static_cast<T*>(ptr_);
    }
    
    size_t nbytes() const { return size_bytes_; }
};

size_t infer_batch_size_from_state_buffer(const DeviceBuffer& d_state_buffer,
                                          unsigned numQubits,
                                          const std::string& operation_name) {
    if (numQubits >= sizeof(size_t) * 8) {
        throw std::runtime_error("num_qubits is too large for " + operation_name + ".");
    }
    size_t elements_per_state = size_t{1} << numQubits;
    if (elements_per_state > std::numeric_limits<size_t>::max() / sizeof(rocComplex)) {
        throw std::runtime_error("state size is too large for " + operation_name + ".");
    }
    size_t bytes_per_state = elements_per_state * sizeof(rocComplex);
    if (d_state_buffer.nbytes() % bytes_per_state != 0) {
        throw std::runtime_error("d_state buffer size is incompatible with num_qubits.");
    }
    size_t batch_size = d_state_buffer.nbytes() / bytes_per_state;
    if (batch_size == 0) {
        throw std::runtime_error("d_state buffer does not contain any batch states.");
    }
    return batch_size;
}


// Wrapper for rocsvHandle_t to ensure proper creation and destruction
class RocsvHandleWrapper {
public:
    rocsvHandle_t handle_ = nullptr;

    RocsvHandleWrapper() {
        rocqStatus_t status = rocsvCreate(&handle_);
        if (status != ROCQ_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create rocsvHandle: " + std::to_string(status));
        }
    }

    ~RocsvHandleWrapper() {
        if (handle_) {
            rocsvDestroy(handle_); // Ignoring status on destroy for simplicity in destructor
        }
    }

    // Disable copy constructor and assignment
    RocsvHandleWrapper(const RocsvHandleWrapper&) = delete;
    RocsvHandleWrapper& operator=(const RocsvHandleWrapper&) = delete;

    // Allow move construction
    RocsvHandleWrapper(RocsvHandleWrapper&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }
    // Allow move assignment
    RocsvHandleWrapper& operator=(RocsvHandleWrapper&& other) noexcept {
        if (this != &other) {
            if (handle_) {
                rocsvDestroy(handle_);
            }
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    rocsvHandle_t get() const { return handle_; }
};


PYBIND11_MODULE(_rocq_hip_backend, m) {
    m.doc() = "Python bindings for rocQuantum hipStateVec library";

    py::enum_<rocqStatus_t>(m, "rocqStatus")
        .value("SUCCESS", ROCQ_STATUS_SUCCESS)
        .value("FAILURE", ROCQ_STATUS_FAILURE)
        .value("INVALID_VALUE", ROCQ_STATUS_INVALID_VALUE)
        .value("ALLOCATION_FAILED", ROCQ_STATUS_ALLOCATION_FAILED)
        .value("HIP_ERROR", ROCQ_STATUS_HIP_ERROR)
        .value("NOT_IMPLEMENTED", ROCQ_STATUS_NOT_IMPLEMENTED)
        .export_values();

    py::enum_<rocsvDistributedBackend_t>(m, "DistributedBackend")
        .value("NONE", ROCSV_DISTRIBUTED_BACKEND_NONE)
        .value("HOST_FALLBACK", ROCSV_DISTRIBUTED_BACKEND_HOST_FALLBACK)
        .value("RCCL", ROCSV_DISTRIBUTED_BACKEND_RCCL)
        .export_values();

    // DeviceBuffer class for managing device memory from Python
    py::class_<DeviceBuffer>(m, "DeviceBuffer")
        .def(py::init<>()) // Default constructor
        .def(py::init<size_t, size_t>(), py::arg("num_elements"), py::arg("element_size"))
        .def("copy_from_numpy", &DeviceBuffer::copy_from_numpy, "Copies data from a NumPy array to the device buffer.")
        .def("nbytes", &DeviceBuffer::nbytes, "Returns the size of the buffer in bytes.");
        // get_ptr is not directly exposed as it's unsafe for general Python.
        // Python code will pass DeviceBuffer objects to wrapped C functions that extract the pointer.


    // Wrapper for the handle
    py::class_<RocsvHandleWrapper>(m, "RocsvHandle")
        .def(py::init<>())
        .def("get_num_gpus", [](const RocsvHandleWrapper& self) {
            int count = 0;
            rocqStatus_t status = rocsvGetNumGpus(self.get(), &count);
            if (status != ROCQ_STATUS_SUCCESS) {
                throw std::runtime_error("rocsvGetNumGpus failed: " + std::to_string(status));
            }
            return count;
        })
        .def("get_distributed_backend", [](const RocsvHandleWrapper& self) {
            rocsvDistributedBackend_t backend = ROCSV_DISTRIBUTED_BACKEND_NONE;
            rocqStatus_t status = rocsvGetDistributedBackend(self.get(), &backend);
            if (status != ROCQ_STATUS_SUCCESS) {
                throw std::runtime_error("rocsvGetDistributedBackend failed: " + std::to_string(status));
            }
            return std::string(rocsvDistributedBackendName(backend));
        });
        // The handle itself is mostly opaque to Python users of this direct binding layer.
        // Higher-level Python classes (Simulator) will use it.

    // State management functions
    // rocsvAllocateState: The returned d_state (rocComplex**) is tricky.
    // We'll return a DeviceBuffer that wraps the allocated device pointer.
    m.def("allocate_state_internal", 
        [](const RocsvHandleWrapper& handle_wrapper, unsigned numQubits, size_t batch_size) {
            rocComplex* d_state_ptr = nullptr;
            rocqStatus_t status = rocsvAllocateState(handle_wrapper.get(), numQubits, &d_state_ptr, batch_size);
            if (status != ROCQ_STATUS_SUCCESS) {
                throw std::runtime_error("rocsvAllocateState failed: " + std::to_string(status));
            }
            size_t num_elements = batch_size * (1ULL << numQubits);
            // The DeviceBuffer now owns this d_state_ptr and will hipFree it.
            return DeviceBuffer(static_cast<void*>(d_state_ptr), num_elements * sizeof(rocComplex), true /*owned*/);
        }, py::arg("handle"), py::arg("num_qubits"), py::arg("batch_size") = 1, "Allocates state vector on device, returns an owning DeviceBuffer.");

    m.def("initialize_state", 
        [](const RocsvHandleWrapper& handle_wrapper, DeviceBuffer& d_state_buffer, unsigned numQubits) {
            infer_batch_size_from_state_buffer(d_state_buffer, numQubits, "initialize_state");
            return rocsvInitializeState(handle_wrapper.get(), d_state_buffer.get_ptr<rocComplex>(), numQubits);
        }, py::arg("handle"), py::arg("d_state_buffer"), py::arg("num_qubits"));

    m.def("allocate_distributed_state",
        [](RocsvHandleWrapper& handle_wrapper, unsigned totalNumQubits) {
            rocqStatus_t status = rocsvAllocateDistributedState(handle_wrapper.get(), totalNumQubits);
            if (status != ROCQ_STATUS_SUCCESS) {
                throw std::runtime_error("rocsvAllocateDistributedState failed: " + std::to_string(status));
            }
        }, py::arg("handle"), py::arg("total_num_qubits"), "Allocates a distributed state vector across multiple GPUs.");

    m.def("allocate_multi_node_distributed_state",
        [](RocsvHandleWrapper& handle_wrapper, unsigned totalNumQubits, unsigned nodeCount) {
            rocqStatus_t status =
                rocsvAllocateMultiNodeDistributedState(handle_wrapper.get(), totalNumQubits, nodeCount);
            if (status != ROCQ_STATUS_SUCCESS) {
                throw std::runtime_error(
                    "rocsvAllocateMultiNodeDistributedState failed: " + std::to_string(status));
            }
        },
        py::arg("handle"),
        py::arg("total_num_qubits"),
        py::arg("node_count"),
        "Explicit unsupported boundary for multi-node distributed state allocation.");

    m.def("initialize_distributed_state",
        [](RocsvHandleWrapper& handle_wrapper) {
            rocqStatus_t status = rocsvInitializeDistributedState(handle_wrapper.get());
            if (status != ROCQ_STATUS_SUCCESS) {
                throw std::runtime_error("rocsvInitializeDistributedState failed: " + std::to_string(status));
            }
        }, py::arg("handle"), "Initializes a distributed state vector to the |0...0> state.");

    // Specific single-qubit gates
    m.def("apply_x", [](const RocsvHandleWrapper& h, DeviceBuffer& d_state, unsigned nQ, unsigned tQ) {
        return rocsvApplyX(h.get(), d_state.get_ptr<rocComplex>(), nQ, tQ); }, "Applies X gate");
    m.def("apply_y", [](const RocsvHandleWrapper& h, DeviceBuffer& d_state, unsigned nQ, unsigned tQ) {
        return rocsvApplyY(h.get(), d_state.get_ptr<rocComplex>(), nQ, tQ); }, "Applies Y gate");
    m.def("apply_z", [](const RocsvHandleWrapper& h, DeviceBuffer& d_state, unsigned nQ, unsigned tQ) {
        return rocsvApplyZ(h.get(), d_state.get_ptr<rocComplex>(), nQ, tQ); }, "Applies Z gate");
    m.def("apply_h", [](const RocsvHandleWrapper& h, DeviceBuffer& d_state, unsigned nQ, unsigned tQ) {
        return rocsvApplyH(h.get(), d_state.get_ptr<rocComplex>(), nQ, tQ); }, "Applies H gate");
    m.def("apply_s", [](const RocsvHandleWrapper& h, DeviceBuffer& d_state, unsigned nQ, unsigned tQ) {
        return rocsvApplyS(h.get(), d_state.get_ptr<rocComplex>(), nQ, tQ); }, "Applies S gate");
    m.def("apply_t", [](const RocsvHandleWrapper& h, DeviceBuffer& d_state, unsigned nQ, unsigned tQ) {
        return rocsvApplyT(h.get(), d_state.get_ptr<rocComplex>(), nQ, tQ); }, "Applies T gate");
    m.def("apply_tdg", [](const RocsvHandleWrapper& h, DeviceBuffer& d_state, unsigned nQ, unsigned tQ) {
        return rocsvApplyTdg(h.get(), d_state.get_ptr<rocComplex>(), nQ, tQ); }, "Applies T dagger gate");
    m.def("apply_sdg", [](const RocsvHandleWrapper& h, DeviceBuffer& d_state, unsigned nQ, unsigned tQ) {
        return rocsvApplySdg(h.get(), d_state.get_ptr<rocComplex>(), nQ, tQ); }, "Applies S dagger gate");

    // Rotation gates
    m.def("apply_rx", [](const RocsvHandleWrapper& h, DeviceBuffer& d_state, unsigned nQ, unsigned tQ, double angle) {
        return rocsvApplyRx(h.get(), d_state.get_ptr<rocComplex>(), nQ, tQ, angle); }, "Applies Rx gate");
    m.def("apply_ry", [](const RocsvHandleWrapper& h, DeviceBuffer& d_state, unsigned nQ, unsigned tQ, double angle) {
        return rocsvApplyRy(h.get(), d_state.get_ptr<rocComplex>(), nQ, tQ, angle); }, "Applies Ry gate");
    m.def("apply_rz", [](const RocsvHandleWrapper& h, DeviceBuffer& d_state, unsigned nQ, unsigned tQ, double angle) {
        return rocsvApplyRz(h.get(), d_state.get_ptr<rocComplex>(), nQ, tQ, angle); }, "Applies Rz gate");
    m.def("apply_p", [](const RocsvHandleWrapper& h, DeviceBuffer& d_state, unsigned nQ, unsigned tQ, double angle) {
        return rocsvApplyP(h.get(), d_state.get_ptr<rocComplex>(), nQ, tQ, angle); }, "Applies phase gate");

    // Specific two-qubit gates
    m.def("apply_cnot", [](const RocsvHandleWrapper& h, DeviceBuffer& d_state, unsigned nQ, unsigned ctrlQ, unsigned tgtQ) {
        return rocsvApplyCNOT(h.get(), d_state.get_ptr<rocComplex>(), nQ, ctrlQ, tgtQ); }, "Applies CNOT gate");
    m.def("apply_cz", [](const RocsvHandleWrapper& h, DeviceBuffer& d_state, unsigned nQ, unsigned q1, unsigned q2) {
        return rocsvApplyCZ(h.get(), d_state.get_ptr<rocComplex>(), nQ, q1, q2); }, "Applies CZ gate");
    m.def("apply_swap", [](const RocsvHandleWrapper& h, DeviceBuffer& d_state, unsigned nQ, unsigned q1, unsigned q2) {
        return rocsvApplySWAP(h.get(), d_state.get_ptr<rocComplex>(), nQ, q1, q2); }, "Applies SWAP gate");

    // --- NEWLY ADDED BINDINGS ---
    m.def("apply_crx", [](const RocsvHandleWrapper& h, DeviceBuffer& d_state, unsigned nQ, unsigned cQ, unsigned tQ, double angle) {
        return rocsvApplyCRX(h.get(), d_state.get_ptr<rocComplex>(), nQ, cQ, tQ, angle);
    }, "Applies CRX gate");
    m.def("apply_cry", [](const RocsvHandleWrapper& h, DeviceBuffer& d_state, unsigned nQ, unsigned cQ, unsigned tQ, double angle) {
        return rocsvApplyCRY(h.get(), d_state.get_ptr<rocComplex>(), nQ, cQ, tQ, angle);
    }, "Applies CRY gate");
    m.def("apply_crz", [](const RocsvHandleWrapper& h, DeviceBuffer& d_state, unsigned nQ, unsigned cQ, unsigned tQ, double angle) {
        return rocsvApplyCRZ(h.get(), d_state.get_ptr<rocComplex>(), nQ, cQ, tQ, angle);
    }, "Applies CRZ gate");
    m.def("apply_cp", [](const RocsvHandleWrapper& h, DeviceBuffer& d_state, unsigned nQ, unsigned cQ, unsigned tQ, double angle) {
        return rocsvApplyCP(h.get(), d_state.get_ptr<rocComplex>(), nQ, cQ, tQ, angle);
    }, "Applies controlled phase gate");
    m.def("apply_mcx", [](const RocsvHandleWrapper& h, DeviceBuffer& d_state, unsigned nQ, const std::vector<unsigned>& cQs, unsigned tQ) {
        return rocsvApplyMultiControlledX(h.get(), d_state.get_ptr<rocComplex>(), nQ, cQs.data(), cQs.size(), tQ);
    }, "Applies multi-controlled X gate");
    m.def("apply_cswap", [](const RocsvHandleWrapper& h, DeviceBuffer& d_state, unsigned nQ, unsigned cQ, unsigned tQ1, unsigned tQ2) {
        return rocsvApplyCSWAP(h.get(), d_state.get_ptr<rocComplex>(), nQ, cQ, tQ1, tQ2);
    }, "Applies CSWAP gate");
    // --- END NEWLY ADDED BINDINGS ---
    
    // rocsvApplyMatrix
    m.def("apply_matrix", 
        [](const RocsvHandleWrapper& handle_wrapper, 
           DeviceBuffer& d_state_buffer, 
           unsigned numQubits, 
           std::vector<unsigned> qubitIndices_vec, // Use std::vector for easy conversion
           DeviceBuffer& matrix_device_buffer, // Matrix already on device
           unsigned matrixDim) {
            // Basic checks
            if (qubitIndices_vec.size() == 0) {
                throw std::runtime_error("qubitIndices must not be empty for apply_matrix");
            }
            unsigned numTargetQubits = qubitIndices_vec.size();
            // matrixDim should be 1U << numTargetQubits
            // matrix_device_buffer.nbytes() should be matrixDim * matrixDim * sizeof(rocComplex)
            
            // The C API expects const unsigned* for qubitIndices.
            // The d_targetIndices for m=3,4,>=5 in C++ code is created on device.
            // Here, qubitIndices is passed from Python as a list/vector, used by C++ to create d_targetIndices.
            // The current C API rocsvApplyMatrix takes const unsigned* qubitIndices (host pointer).
            // This is consistent.

            return rocsvApplyMatrix(handle_wrapper.get(), 
                                    d_state_buffer.get_ptr<rocComplex>(), 
                                    numQubits, 
                                    qubitIndices_vec.data(), // Pass pointer to vector's data
                                    numTargetQubits, 
                                    matrix_device_buffer.get_ptr<rocComplex>(), 
                                    matrixDim);
        }, py::arg("handle"), py::arg("d_state"), py::arg("num_qubits"), 
           py::arg("qubit_indices"), py::arg("matrix_device"), py::arg("matrix_dim"));

    // rocsvMeasure
    m.def("measure", 
        [](const RocsvHandleWrapper& handle_wrapper, DeviceBuffer& d_state_buffer, unsigned numQubits, unsigned qubitToMeasure) {
            int outcome = 0;
            double probability = 0.0;
            rocqStatus_t status = rocsvMeasure(handle_wrapper.get(), 
                                               d_state_buffer.get_ptr<rocComplex>(), 
                                               numQubits, 
                                               qubitToMeasure, 
                                               &outcome, &probability);
            if (status != ROCQ_STATUS_SUCCESS) {
                throw std::runtime_error("rocsvMeasure failed: " + std::to_string(status));
            }
            return py::make_tuple(outcome, probability);
        }, py::arg("handle"), py::arg("d_state"), py::arg("num_qubits"), py::arg("qubit_to_measure"),
           "Measures a single qubit. Returns (outcome, probability).");

    m.def("get_expectation_value_z",
        [](const RocsvHandleWrapper& handle_wrapper, DeviceBuffer& d_state_buffer, unsigned numQubits, unsigned targetQubit) {
            double result = 0.0;
            rocqStatus_t status = rocsvGetExpectationValueSinglePauliZ(
                                               handle_wrapper.get(),
                                               d_state_buffer.get_ptr<rocComplex>(),
                                               numQubits,
                                               targetQubit,
                                               &result);
            if (status != ROCQ_STATUS_SUCCESS) {
                throw std::runtime_error("rocsvGetExpectationValueSinglePauliZ failed: " + std::to_string(status));
            }
            return result;
        }, py::arg("handle"), py::arg("d_state"), py::arg("num_qubits"), py::arg("target_qubit"),
           "Calculates <Z_k> for the target qubit.");

    m.def("get_expectation_value_x",
        [](const RocsvHandleWrapper& handle_wrapper, DeviceBuffer& d_state_buffer, unsigned numQubits, unsigned targetQubit) {
            double result = 0.0;
            rocqStatus_t status = rocsvGetExpectationValueSinglePauliX(
                                               handle_wrapper.get(),
                                               d_state_buffer.get_ptr<rocComplex>(),
                                               numQubits,
                                               targetQubit,
                                               &result);
            if (status != ROCQ_STATUS_SUCCESS) {
                throw std::runtime_error("rocsvGetExpectationValueSinglePauliX failed: " + std::to_string(status));
            }
            return result;
        }, py::arg("handle"), py::arg("d_state"), py::arg("num_qubits"), py::arg("target_qubit"),
           "Calculates <X_k> for the target qubit. Modifies state vector.");

    m.def("get_expectation_value_y",
        [](const RocsvHandleWrapper& handle_wrapper, DeviceBuffer& d_state_buffer, unsigned numQubits, unsigned targetQubit) {
            double result = 0.0;
            rocqStatus_t status = rocsvGetExpectationValueSinglePauliY(
                                               handle_wrapper.get(),
                                               d_state_buffer.get_ptr<rocComplex>(),
                                               numQubits,
                                               targetQubit,
                                               &result);
            if (status != ROCQ_STATUS_SUCCESS) {
                throw std::runtime_error("rocsvGetExpectationValueSinglePauliY failed: " + std::to_string(status));
            }
            return result;
        }, py::arg("handle"), py::arg("d_state"), py::arg("num_qubits"), py::arg("target_qubit"),
           "Calculates <Y_k> for the target qubit. Modifies state vector.");

    m.def("get_expectation_value_pauli_product_z",
        [](const RocsvHandleWrapper& handle_wrapper, DeviceBuffer& d_state_buffer, unsigned numQubits,
           const std::vector<unsigned>& targetQubits_vec) {
            double result = 0.0;
            if (targetQubits_vec.empty()) { // Should be handled by Python PauliOperator logic too
                return 1.0; // Expectation of Identity
            }
            rocqStatus_t status = rocsvGetExpectationValuePauliProductZ(
                                               handle_wrapper.get(),
                                               d_state_buffer.get_ptr<rocComplex>(),
                                               numQubits,
                                               targetQubits_vec.data(), // Pass pointer to vector's data
                                               static_cast<unsigned>(targetQubits_vec.size()),
                                               &result);
            if (status != ROCQ_STATUS_SUCCESS) {
                throw std::runtime_error("rocsvGetExpectationValuePauliProductZ failed: " + std::to_string(status));
            }
            return result;
        }, py::arg("handle"), py::arg("d_state"), py::arg("num_qubits"), py::arg("target_qubits"),
           "Calculates <Z_q0 Z_q1 ...> for the target qubits. Does not modify state.");

    m.def("get_expectation_pauli_string",
        [](const RocsvHandleWrapper& handle_wrapper, DeviceBuffer& d_state_buffer, unsigned numQubits,
           const std::string& pauliString, const std::vector<unsigned>& targetQubits_vec) {
            double result = 0.0;
            if (pauliString.length() != targetQubits_vec.size()) {
                throw std::runtime_error("Pauli string length must match the number of target qubits.");
            }
            if (targetQubits_vec.empty()) {
                return 1.0; // Expectation of Identity
            }

            rocqStatus_t status = rocsvGetExpectationPauliString(
                                               handle_wrapper.get(),
                                               d_state_buffer.get_ptr<rocComplex>(),
                                               numQubits,
                                               pauliString.c_str(),
                                               targetQubits_vec.data(),
                                               static_cast<unsigned>(targetQubits_vec.size()),
                                               &result);
            if (status != ROCQ_STATUS_SUCCESS) {
                throw std::runtime_error("rocsvGetExpectationPauliString failed: " + std::to_string(status));
            }
            return result;
        }, py::arg("handle"), py::arg("d_state"), py::arg("num_qubits"), py::arg("pauli_string"), py::arg("target_qubits"),
           "Calculates expectation value for a generic Pauli string (e.g., \"IXYZ\"). Non-destructive.");

    m.def("get_expectation_pauli_string_batch",
        [](const RocsvHandleWrapper& handle_wrapper, DeviceBuffer& d_state_buffer, unsigned numQubits,
           const std::string& pauliString, const std::vector<unsigned>& targetQubits_vec) {
            if (pauliString.length() != targetQubits_vec.size()) {
                throw std::runtime_error("Pauli string length must match the number of target qubits.");
            }
            size_t batch_size = infer_batch_size_from_state_buffer(
                d_state_buffer,
                numQubits,
                "get_expectation_pauli_string_batch");

            py::array_t<double> result(batch_size);
            auto mutable_result = result.mutable_unchecked<1>();
            if (targetQubits_vec.empty()) {
                for (size_t idx = 0; idx < batch_size; ++idx) {
                    mutable_result(static_cast<py::ssize_t>(idx)) = 1.0;
                }
                return result;
            }

            std::string normalized;
            normalized.reserve(pauliString.size());
            for (char pauli : pauliString) {
                const char upper = static_cast<char>(std::toupper(static_cast<unsigned char>(pauli)));
                if (upper != 'I' && upper != 'X' && upper != 'Y' && upper != 'Z') {
                    throw std::runtime_error("Pauli string may only contain I, X, Y, or Z.");
                }
                normalized.push_back(upper);
            }

            std::vector<double> raw_results(batch_size, 0.0);
            rocqStatus_t status = rocsvGetExpectationPauliStringBatch(
                handle_wrapper.get(),
                d_state_buffer.get_ptr<rocComplex>(),
                numQubits,
                normalized.c_str(),
                targetQubits_vec.data(),
                static_cast<unsigned>(targetQubits_vec.size()),
                raw_results.data());
            if (status != ROCQ_STATUS_SUCCESS) {
                throw std::runtime_error("rocsvGetExpectationPauliStringBatch failed: " + std::to_string(status));
            }
            for (size_t idx = 0; idx < batch_size; ++idx) {
                mutable_result(static_cast<py::ssize_t>(idx)) = raw_results[idx];
            }
            return result;
        }, py::arg("handle"), py::arg("d_state"), py::arg("num_qubits"), py::arg("pauli_string"), py::arg("target_qubits"),
           "Calculates Pauli-string expectation values for each local batch state.");

    m.def("get_expectation_matrix",
        [](const RocsvHandleWrapper& handle_wrapper, DeviceBuffer& d_state_buffer, unsigned numQubits,
           const std::vector<unsigned>& targetQubits_vec,
           py::array_t<rocComplex, py::array::c_style | py::array::forcecast> matrix) {
            if (targetQubits_vec.empty()) {
                throw std::runtime_error("target_qubits must not be empty for get_expectation_matrix.");
            }
            if (matrix.ndim() != 2 || matrix.shape(0) != matrix.shape(1)) {
                throw std::runtime_error("matrix must be a square 2D array for get_expectation_matrix.");
            }

            const size_t matrix_dim = static_cast<size_t>(matrix.shape(0));
            if (targetQubits_vec.size() >= sizeof(size_t) * 8 ||
                matrix_dim != (size_t{1} << targetQubits_vec.size())) {
                throw std::runtime_error("matrix dimension must equal 2^len(target_qubits).");
            }

            const rocComplex* matrix_row_major = matrix.data();
            std::vector<rocComplex> matrix_col_major(matrix_dim * matrix_dim);
            for (size_t row = 0; row < matrix_dim; ++row) {
                for (size_t col = 0; col < matrix_dim; ++col) {
                    matrix_col_major[row + col * matrix_dim] =
                        matrix_row_major[row * matrix_dim + col];
                }
            }

            DeviceBuffer matrix_device(matrix_col_major.size(), sizeof(rocComplex));
            if (hipMemcpy(matrix_device.get_ptr<rocComplex>(),
                          matrix_col_major.data(),
                          matrix_col_major.size() * sizeof(rocComplex),
                          hipMemcpyHostToDevice) != hipSuccess) {
                throw std::runtime_error("Failed to copy expectation matrix to device");
            }

            rocComplex result{};
            rocqStatus_t status = rocsvGetExpectationMatrix(
                handle_wrapper.get(),
                d_state_buffer.get_ptr<rocComplex>(),
                numQubits,
                targetQubits_vec.data(),
                static_cast<unsigned>(targetQubits_vec.size()),
                matrix_device.get_ptr<rocComplex>(),
                matrix_dim,
                &result);
            if (status != ROCQ_STATUS_SUCCESS) {
                throw std::runtime_error("rocsvGetExpectationMatrix failed: " + std::to_string(status));
            }
            return std::complex<double>(static_cast<double>(result.x), static_cast<double>(result.y));
        }, py::arg("handle"), py::arg("d_state"), py::arg("num_qubits"), py::arg("target_qubits"), py::arg("matrix"),
           "Calculates <psi|M|psi> for a dense matrix acting on target qubits.");

    m.def("get_expectation_matrix_batch",
        [](const RocsvHandleWrapper& handle_wrapper, DeviceBuffer& d_state_buffer, unsigned numQubits,
           const std::vector<unsigned>& targetQubits_vec,
           py::array_t<rocComplex, py::array::c_style | py::array::forcecast> matrix) {
            if (targetQubits_vec.empty()) {
                throw std::runtime_error("target_qubits must not be empty for get_expectation_matrix_batch.");
            }
            if (matrix.ndim() != 2 || matrix.shape(0) != matrix.shape(1)) {
                throw std::runtime_error("matrix must be a square 2D array for get_expectation_matrix_batch.");
            }

            const size_t matrix_dim = static_cast<size_t>(matrix.shape(0));
            if (targetQubits_vec.size() >= sizeof(size_t) * 8 ||
                matrix_dim != (size_t{1} << targetQubits_vec.size())) {
                throw std::runtime_error("matrix dimension must equal 2^len(target_qubits).");
            }

            const rocComplex* matrix_row_major = matrix.data();
            std::vector<rocComplex> matrix_col_major(matrix_dim * matrix_dim);
            for (size_t row = 0; row < matrix_dim; ++row) {
                for (size_t col = 0; col < matrix_dim; ++col) {
                    matrix_col_major[row + col * matrix_dim] =
                        matrix_row_major[row * matrix_dim + col];
                }
            }

            DeviceBuffer matrix_device(matrix_col_major.size(), sizeof(rocComplex));
            if (hipMemcpy(matrix_device.get_ptr<rocComplex>(),
                          matrix_col_major.data(),
                          matrix_col_major.size() * sizeof(rocComplex),
                          hipMemcpyHostToDevice) != hipSuccess) {
                throw std::runtime_error("Failed to copy batch expectation matrix to device");
            }

            size_t batch_size = infer_batch_size_from_state_buffer(
                d_state_buffer,
                numQubits,
                "get_expectation_matrix_batch");
            std::vector<rocComplex> raw_results(batch_size);
            rocqStatus_t status = rocsvGetExpectationMatrixBatch(
                handle_wrapper.get(),
                d_state_buffer.get_ptr<rocComplex>(),
                numQubits,
                targetQubits_vec.data(),
                static_cast<unsigned>(targetQubits_vec.size()),
                matrix_device.get_ptr<rocComplex>(),
                matrix_dim,
                raw_results.data());
            if (status != ROCQ_STATUS_SUCCESS) {
                throw std::runtime_error("rocsvGetExpectationMatrixBatch failed: " + std::to_string(status));
            }

            py::array_t<std::complex<double>> result(batch_size);
            auto mutable_result = result.mutable_unchecked<1>();
            for (size_t idx = 0; idx < batch_size; ++idx) {
                mutable_result(static_cast<py::ssize_t>(idx)) =
                    std::complex<double>(static_cast<double>(raw_results[idx].x),
                                         static_cast<double>(raw_results[idx].y));
            }
            return result;
        }, py::arg("handle"), py::arg("d_state"), py::arg("num_qubits"), py::arg("target_qubits"), py::arg("matrix"),
           "Calculates <psi|M|psi> for each local batch state.");

    m.def("get_sparse_matrix_moments",
        [](const RocsvHandleWrapper& handle_wrapper, DeviceBuffer& d_state_buffer, unsigned numQubits,
           py::array_t<rocComplex, py::array::c_style | py::array::forcecast> data,
           const std::vector<size_t>& indices_vec,
           const std::vector<size_t>& indptr_vec,
           size_t rows,
           size_t cols) {
            if (data.ndim() != 1) {
                throw std::runtime_error("CSR data must be a 1D array for get_sparse_matrix_moments.");
            }
            const size_t nnz = static_cast<size_t>(data.size());
            if (indices_vec.size() != nnz) {
                throw std::runtime_error("CSR indices length must match data length.");
            }
            if (rows == 0 || cols == 0 || rows != cols) {
                throw std::runtime_error("CSR shape must be square and non-empty.");
            }
            if (numQubits >= sizeof(size_t) * 8 || rows != (size_t{1} << numQubits)) {
                throw std::runtime_error("CSR shape must match 2^num_qubits.");
            }
            if (indptr_vec.size() != rows + 1 || indptr_vec.empty() ||
                indptr_vec.front() != 0 || indptr_vec.back() != nnz) {
                throw std::runtime_error("CSR indptr must start at 0, end at nnz, and have rows + 1 entries.");
            }
            for (size_t row = 0; row < rows; ++row) {
                if (indptr_vec[row] > indptr_vec[row + 1]) {
                    throw std::runtime_error("CSR indptr must be monotonic.");
                }
            }
            for (const size_t col : indices_vec) {
                if (col >= cols) {
                    throw std::runtime_error("CSR column index is out of bounds.");
                }
            }

            DeviceBuffer data_device;
            if (nnz > 0) {
                data_device = DeviceBuffer(nnz, sizeof(rocComplex));
                if (hipMemcpy(data_device.get_ptr<rocComplex>(),
                              data.data(),
                              nnz * sizeof(rocComplex),
                              hipMemcpyHostToDevice) != hipSuccess) {
                    throw std::runtime_error("Failed to copy CSR data to device");
                }
            }

            DeviceBuffer indices_device;
            if (!indices_vec.empty()) {
                indices_device = DeviceBuffer(indices_vec.size(), sizeof(size_t));
                if (hipMemcpy(indices_device.get_ptr<size_t>(),
                              indices_vec.data(),
                              indices_vec.size() * sizeof(size_t),
                              hipMemcpyHostToDevice) != hipSuccess) {
                    throw std::runtime_error("Failed to copy CSR indices to device");
                }
            }

            DeviceBuffer indptr_device(indptr_vec.size(), sizeof(size_t));
            if (hipMemcpy(indptr_device.get_ptr<size_t>(),
                          indptr_vec.data(),
                          indptr_vec.size() * sizeof(size_t),
                          hipMemcpyHostToDevice) != hipSuccess) {
                throw std::runtime_error("Failed to copy CSR indptr to device");
            }

            rocComplex mean{};
            rocComplex second_moment{};
            rocqStatus_t status = rocsvGetSparseMatrixMoments(
                handle_wrapper.get(),
                d_state_buffer.get_ptr<rocComplex>(),
                numQubits,
                nnz > 0 ? data_device.get_ptr<rocComplex>() : nullptr,
                !indices_vec.empty() ? indices_device.get_ptr<size_t>() : nullptr,
                indptr_device.get_ptr<size_t>(),
                rows,
                cols,
                nnz,
                &mean,
                &second_moment);
            if (status != ROCQ_STATUS_SUCCESS) {
                throw std::runtime_error("rocsvGetSparseMatrixMoments failed: " + std::to_string(status));
            }

            return py::make_tuple(
                std::complex<double>(static_cast<double>(mean.x), static_cast<double>(mean.y)),
                std::complex<double>(static_cast<double>(second_moment.x), static_cast<double>(second_moment.y)));
        }, py::arg("handle"), py::arg("d_state"), py::arg("num_qubits"), py::arg("data"),
           py::arg("indices"), py::arg("indptr"), py::arg("rows"), py::arg("cols"),
           "Calculates sparse CSR <psi|H|psi> and <psi|H^2|psi> moments.");

    m.def("get_sparse_matrix_moments_batch",
        [](const RocsvHandleWrapper& handle_wrapper, DeviceBuffer& d_state_buffer, unsigned numQubits,
           py::array_t<rocComplex, py::array::c_style | py::array::forcecast> data,
           const std::vector<size_t>& indices_vec,
           const std::vector<size_t>& indptr_vec,
           size_t rows,
           size_t cols) {
            if (data.ndim() != 1) {
                throw std::runtime_error("CSR data must be a 1D array for get_sparse_matrix_moments_batch.");
            }
            const size_t nnz = static_cast<size_t>(data.size());
            if (indices_vec.size() != nnz) {
                throw std::runtime_error("CSR indices length must match data length.");
            }
            if (rows == 0 || cols == 0 || rows != cols) {
                throw std::runtime_error("CSR shape must be square and non-empty.");
            }
            if (numQubits >= sizeof(size_t) * 8 || rows != (size_t{1} << numQubits)) {
                throw std::runtime_error("CSR shape must match 2^num_qubits.");
            }
            if (indptr_vec.size() != rows + 1 || indptr_vec.empty() ||
                indptr_vec.front() != 0 || indptr_vec.back() != nnz) {
                throw std::runtime_error("CSR indptr must start at 0, end at nnz, and have rows + 1 entries.");
            }
            for (size_t row = 0; row < rows; ++row) {
                if (indptr_vec[row] > indptr_vec[row + 1]) {
                    throw std::runtime_error("CSR indptr must be monotonic.");
                }
            }
            for (const size_t col : indices_vec) {
                if (col >= cols) {
                    throw std::runtime_error("CSR column index is out of bounds.");
                }
            }

            DeviceBuffer data_device;
            if (nnz > 0) {
                data_device = DeviceBuffer(nnz, sizeof(rocComplex));
                if (hipMemcpy(data_device.get_ptr<rocComplex>(),
                              data.data(),
                              nnz * sizeof(rocComplex),
                              hipMemcpyHostToDevice) != hipSuccess) {
                    throw std::runtime_error("Failed to copy batch CSR data to device");
                }
            }

            DeviceBuffer indices_device;
            if (!indices_vec.empty()) {
                indices_device = DeviceBuffer(indices_vec.size(), sizeof(size_t));
                if (hipMemcpy(indices_device.get_ptr<size_t>(),
                              indices_vec.data(),
                              indices_vec.size() * sizeof(size_t),
                              hipMemcpyHostToDevice) != hipSuccess) {
                    throw std::runtime_error("Failed to copy batch CSR indices to device");
                }
            }

            DeviceBuffer indptr_device(indptr_vec.size(), sizeof(size_t));
            if (hipMemcpy(indptr_device.get_ptr<size_t>(),
                          indptr_vec.data(),
                          indptr_vec.size() * sizeof(size_t),
                          hipMemcpyHostToDevice) != hipSuccess) {
                throw std::runtime_error("Failed to copy batch CSR indptr to device");
            }

            size_t batch_size = infer_batch_size_from_state_buffer(
                d_state_buffer,
                numQubits,
                "get_sparse_matrix_moments_batch");
            std::vector<rocComplex> raw_means(batch_size);
            std::vector<rocComplex> raw_second_moments(batch_size);
            rocqStatus_t status = rocsvGetSparseMatrixMomentsBatch(
                handle_wrapper.get(),
                d_state_buffer.get_ptr<rocComplex>(),
                numQubits,
                nnz > 0 ? data_device.get_ptr<rocComplex>() : nullptr,
                !indices_vec.empty() ? indices_device.get_ptr<size_t>() : nullptr,
                indptr_device.get_ptr<size_t>(),
                rows,
                cols,
                nnz,
                raw_means.data(),
                raw_second_moments.data());
            if (status != ROCQ_STATUS_SUCCESS) {
                throw std::runtime_error("rocsvGetSparseMatrixMomentsBatch failed: " + std::to_string(status));
            }

            py::array_t<std::complex<double>> means(batch_size);
            py::array_t<std::complex<double>> second_moments(batch_size);
            auto mutable_means = means.mutable_unchecked<1>();
            auto mutable_second = second_moments.mutable_unchecked<1>();
            for (size_t idx = 0; idx < batch_size; ++idx) {
                mutable_means(static_cast<py::ssize_t>(idx)) =
                    std::complex<double>(static_cast<double>(raw_means[idx].x),
                                         static_cast<double>(raw_means[idx].y));
                mutable_second(static_cast<py::ssize_t>(idx)) =
                    std::complex<double>(static_cast<double>(raw_second_moments[idx].x),
                                         static_cast<double>(raw_second_moments[idx].y));
            }
            return py::make_tuple(means, second_moments);
        }, py::arg("handle"), py::arg("d_state"), py::arg("num_qubits"), py::arg("data"),
           py::arg("indices"), py::arg("indptr"), py::arg("rows"), py::arg("cols"),
           "Calculates batch sparse CSR <psi|H|psi> and <psi|H^2|psi> moments.");

    m.def("sample",
        [](const RocsvHandleWrapper& handle_wrapper, DeviceBuffer& d_state_buffer, unsigned numQubits,
           const std::vector<unsigned>& measuredQubits_vec, unsigned numShots) {
            if (numShots == 0) {
                return py::array_t<uint64_t>(0);
            }

            py::array_t<uint64_t> h_results(numShots);
            
            rocqStatus_t status = rocsvSample(
                                        handle_wrapper.get(),
                                        d_state_buffer.get_ptr<rocComplex>(),
                                        numQubits,
                                        measuredQubits_vec.data(),
                                        static_cast<unsigned>(measuredQubits_vec.size()),
                                        numShots,
                                        h_results.mutable_data());

            if (status != ROCQ_STATUS_SUCCESS) {
                throw std::runtime_error("rocsvSample failed: " + std::to_string(status));
            }
            return h_results;
        }, py::arg("handle"), py::arg("d_state"), py::arg("num_qubits"), py::arg("measured_qubits"), py::arg("num_shots"),
           "Samples from the state vector and returns an array of measurement outcomes (bitstrings).");

    m.def("probabilities",
        [](const RocsvHandleWrapper& handle_wrapper, DeviceBuffer& d_state_buffer, unsigned numQubits,
           const std::vector<unsigned>& measuredQubits_vec) {
            if (measuredQubits_vec.empty()) {
                throw std::invalid_argument("probabilities requires at least one measured qubit.");
            }
            if (measuredQubits_vec.size() > 20) {
                throw std::runtime_error("probabilities currently supports at most 20 measured qubits.");
            }
            const std::size_t num_outcomes = std::size_t{1} << measuredQubits_vec.size();
            py::array_t<double> h_probabilities(num_outcomes);

            rocqStatus_t status = rocsvProbabilities(
                                        handle_wrapper.get(),
                                        d_state_buffer.get_ptr<rocComplex>(),
                                        numQubits,
                                        measuredQubits_vec.data(),
                                        static_cast<unsigned>(measuredQubits_vec.size()),
                                        h_probabilities.mutable_data());

            if (status != ROCQ_STATUS_SUCCESS) {
                throw std::runtime_error("rocsvProbabilities failed: " + std::to_string(status));
            }
            return h_probabilities;
        }, py::arg("handle"), py::arg("d_state"), py::arg("num_qubits"), py::arg("measured_qubits"),
           "Returns normalized computational-basis probabilities for selected qubits.");

    m.def("apply_controlled_matrix",
        [](const RocsvHandleWrapper& handle_wrapper, DeviceBuffer& d_state_buffer, unsigned numQubits,
           const std::vector<unsigned>& controlQubits_vec, const std::vector<unsigned>& targetQubits_vec,
           DeviceBuffer& matrix_device_buffer) {
            
            unsigned numControls = static_cast<unsigned>(controlQubits_vec.size());
            unsigned numTargets = static_cast<unsigned>(targetQubits_vec.size());

            if (numTargets == 0) return ROCQ_STATUS_SUCCESS;
            if (numControls == 0) { // Fallback to regular apply_matrix
                 return rocsvApplyMatrix(handle_wrapper.get(), 
                                    d_state_buffer.get_ptr<rocComplex>(), 
                                    numQubits, 
                                    targetQubits_vec.data(),
                                    numTargets, 
                                    matrix_device_buffer.get_ptr<rocComplex>(), 
                                    1U << numTargets);
            }

            rocqStatus_t status = rocsvApplyControlledMatrix(
                                        handle_wrapper.get(),
                                        d_state_buffer.get_ptr<rocComplex>(),
                                        numQubits,
                                        controlQubits_vec.data(),
                                        numControls,
                                        targetQubits_vec.data(),
                                        numTargets,
                                        matrix_device_buffer.get_ptr<rocComplex>());

            if (status != ROCQ_STATUS_SUCCESS) {
                throw std::runtime_error("rocsvApplyControlledMatrix failed: " + std::to_string(status));
            }
            return status;
        }, py::arg("handle"), py::arg("d_state"), py::arg("num_qubits"), 
           py::arg("control_qubits"), py::arg("target_qubits"), py::arg("matrix_device"),
           "Applies a matrix to target qubits, controlled by control qubits.");

    m.def("get_state_vector_full", [](const RocsvHandleWrapper& handle, DeviceBuffer& d_state_buffer, unsigned num_qubits, size_t batch_size) {
        size_t num_elements = batch_size * (1ULL << num_qubits);
        py::array_t<rocComplex> h_state(num_elements);
        rocqStatus_t status = rocsvGetStateVectorFull(handle.get(), d_state_buffer.get_ptr<rocComplex>(), h_state.mutable_data());
        if (status != ROCQ_STATUS_SUCCESS) {
            throw std::runtime_error("rocsvGetStateVectorFull failed: " + std::to_string(status));
        }
        return h_state;
    }, py::arg("handle"), py::arg("d_state").noconvert(), py::arg("num_qubits"), py::arg("batch_size"));

    m.def("get_state_vector_slice", [](const RocsvHandleWrapper& handle, DeviceBuffer& d_state_buffer, unsigned num_qubits, size_t batch_size, unsigned batch_index) {
        size_t num_elements = 1ULL << num_qubits;
        py::array_t<rocComplex> h_state(num_elements);
        rocqStatus_t status = rocsvGetStateVectorSlice(handle.get(), d_state_buffer.get_ptr<rocComplex>(), h_state.mutable_data(), batch_index);
        if (status != ROCQ_STATUS_SUCCESS) {
            throw std::runtime_error("rocsvGetStateVectorSlice failed: " + std::to_string(status));
        }
        return h_state;
    }, py::arg("handle"), py::arg("d_state").noconvert(), py::arg("num_qubits"), py::arg("batch_size"), py::arg("batch_index"));

    // Add a helper to create a DeviceBuffer and copy a numpy array to it
    m.def("create_device_matrix_from_numpy",
        [](py::array_t<rocComplex, py::array::c_style | py::array::forcecast> np_array) {
            if (np_array.ndim() != 2) throw std::runtime_error("NumPy array must be 2D for matrix.");
            size_t num_elements = np_array.size();
            DeviceBuffer db(num_elements, sizeof(rocComplex)); // Owns memory
            db.copy_from_numpy(np_array);
            return db;
        }, py::arg("numpy_array"), "Creates a DeviceBuffer and copies a NumPy array to it.");

    // --- rocTensorUtil Bindings ---
    py::class_<rocquantum::util::rocTensor>(m, "RocTensor")
        .def(py::init<>(), "Default constructor")
        .def(py::init([](const std::vector<long long>& dims, py::object py_data_np_array) {
            // This constructor is primarily for Python-side creation of metadata.
            // Actual device data allocation should be done via allocate_tensor.
            // If py_data_np_array is a numpy array, we could try to initialize from it,
            // but that complicates ownership. For now, primarily for dimensions.
            auto tensor = rocquantum::util::rocTensor();
            tensor.dimensions_ = dims;
            tensor.calculate_strides(); // Calculate strides based on dimensions
            // Data pointer (tensor.data_) should be set via allocate_tensor or by wrapping existing device memory.
            // tensor.owned_ will be set by allocate_tensor.
            return tensor;
        }), py::arg("dimensions"), py::arg("py_data_np_array") = py::none(), "Constructor with dimensions. Data must be set via allocate_tensor or from existing buffer.")
        .def_property("dimensions",
            [](const rocquantum::util::rocTensor &self) { return self.dimensions_; },
            [](rocquantum::util::rocTensor &self, const std::vector<long long>& dims) {
                self.dimensions_ = dims;
                self.calculate_strides(); // Recalculate strides when dimensions change
            })
        .def_property("labels",
            [](const rocquantum::util::rocTensor &self) { return self.labels_; },
            [](rocquantum::util::rocTensor &self, const std::vector<std::string>& lbls) { self.labels_ = lbls; })
        .def_property_readonly("strides", [](const rocquantum::util::rocTensor &self) { return self.strides_; })
        .def("get_element_count", &rocquantum::util::rocTensor::get_element_count)
        .def("rank", &rocquantum::util::rocTensor::rank)
        // Note: data_ pointer is not directly exposed for safety.
        // owned_ flag is also not directly exposed, managed by allocate/free.
        .def("__repr__", [](const rocquantum::util::rocTensor &t) {
            std::string rep = "<rocq.RocTensor dimensions=[";
            for (size_t i = 0; i < t.dimensions_.size(); ++i) {
                rep += std::to_string(t.dimensions_[i]) + (i == t.dimensions_.size() - 1 ? "" : ", ");
            }
            rep += "], rank=" + std::to_string(t.rank());
            rep += ", elements=" + std::to_string(t.get_element_count());
            rep += (t.data_ ? ", has_data" : ", no_data");
            rep += (t.owned_ ? ", owned" : ", view");
            rep += ">";
            return rep;
        });

    m.def("allocate_tensor", [](rocquantum::util::rocTensor& tensor) {
        // This function will modify tensor in-place to allocate its data
        rocqStatus_t status = rocquantum::util::rocTensorAllocate(&tensor);
        if (status != ROCQ_STATUS_SUCCESS) {
            throw std::runtime_error("rocTensorAllocate failed: " + std::to_string(status));
        }
        // The tensor passed by reference is modified (data_ pointer set, owned_ set to true)
    }, py::arg("tensor").noconvert(), "Allocates device memory for the given RocTensor object (modifies in-place).");

    m.def("free_tensor", [](rocquantum::util::rocTensor& tensor) {
        rocqStatus_t status = rocquantum::util::rocTensorFree(&tensor);
        if (status != ROCQ_STATUS_SUCCESS) {
            // Perhaps just warn or log if freeing non-owned/null data,
            // but rocTensorFree should handle this gracefully.
            // For now, any error from rocTensorFree is an exception.
            throw std::runtime_error("rocTensorFree failed: " + std::to_string(status));
        }
        // Tensor is modified in-place (data_ to nullptr, owned_ to false)
    }, py::arg("tensor").noconvert(), "Frees device memory for the RocTensor if it's owned (modifies in-place).");

    m.def("permute_tensor",
        [](rocquantum::util::rocTensor& output_tensor,
           const rocquantum::util::rocTensor& input_tensor,
           const std::vector<int>& host_permutation_map) {
        // ... (existing implementation)
    }, py::arg("output_tensor").noconvert(), py::arg("input_tensor"), py::arg("permutation_map"));

    // --- hipTensorNet Bindings ---
    // ... (rest of the file)

    // --- hipTensorNet Bindings ---
    // Opaque handle wrapper for rocTensorNetworkHandle_t
    class RocTensorNetworkHandleWrapper {
    public:
        rocTensorNetworkHandle_t handle_ = nullptr;
        RocsvHandleWrapper& sim_handle_ref_; // Keep a reference to the simulator's handle for rocBLAS/stream

        RocTensorNetworkHandleWrapper(RocsvHandleWrapper& sim_handle, py::object dtype_source) 
            : sim_handle_ref_(sim_handle) {
            
            py::dtype dt;
            if (py::isinstance<py::array>(dtype_source)) {
                dt = py::cast<py::array>(dtype_source).dtype();
            } else if (py::isinstance<py::dtype>(dtype_source)) {
                dt = py::cast<py::dtype>(dtype_source);
            } else {
                throw std::invalid_argument("Constructor requires a NumPy array or a NumPy dtype to determine the data type.");
            }

            rocDataType_t rocq_dtype = get_rocq_dtype_from_numpy(dt);
            
            rocqStatus_t status = rocTensorNetworkCreate(&handle_, rocq_dtype);
            if (status != ROCQ_STATUS_SUCCESS) {
                throw std::runtime_error(tensornet_status_message("rocTensorNetworkCreate", status));
            }
        }

        ~RocTensorNetworkHandleWrapper() {
            if (handle_) {
                rocTensorNetworkDestroy(handle_);
            }
        }

        // Disable copy constructor and assignment
        RocTensorNetworkHandleWrapper(const RocTensorNetworkHandleWrapper&) = delete;
        RocTensorNetworkHandleWrapper& operator=(const RocTensorNetworkHandleWrapper&) = delete;

        // Allow move construction
        RocTensorNetworkHandleWrapper(RocTensorNetworkHandleWrapper&& other) noexcept
            : handle_(other.handle_), sim_handle_ref_(other.sim_handle_ref_) {
            other.handle_ = nullptr;
        }
        
        // Note: Move assignment is problematic with reference members and is omitted for simplicity.
        RocTensorNetworkHandleWrapper& operator=(RocTensorNetworkHandleWrapper&& other) = delete;

        rocTensorNetworkHandle_t get() const { return handle_; }
        RocsvHandleWrapper& get_sim_handle() const { return sim_handle_ref_; }
    };

    py::class_<RocTensorNetworkHandleWrapper>(m, "RocTensorNetwork")
        .def(py::init<RocsvHandleWrapper&, py::object>(), py::arg("simulator_handle"), py::arg("dtype_source"),
             "Creates a Tensor Network manager for a specific data type.\n"
             "dtype_source: A NumPy array or np.dtype to specify the precision (e.g., np.float32, np.complex64).")
        
        .def("add_tensor", [](RocTensorNetworkHandleWrapper& self, const rocquantum::util::rocTensor& tensor) {
            rocqStatus_t status = rocTensorNetworkAddTensor(self.get(), &tensor);
            if (status != ROCQ_STATUS_SUCCESS) {
                throw std::runtime_error(tensornet_status_message("rocTensorNetworkAddTensor", status));
            }
        }, py::arg("tensor"))

        .def("contract", [](RocTensorNetworkHandleWrapper& self, 
                                 py::object config_obj, // Can be dict or None
                                 rocquantum::util::rocTensor& result_tensor_py) {
            hipTensorNetContractionOptimizerConfig_t config;
            config.pathfinder_algorithm = ROCTN_PATHFINDER_ALGO_GREEDY;
            config.algo_config.kahypar_config.imbalance_factor = 0.03;
            config.memory_limit_bytes = 0;
            config.num_slices = 0;

            if (!config_obj.is_none()) {
                py::dict config_dict = py::cast<py::dict>(config_obj);
                if (config_dict.contains("pathfinder_algorithm")) {
                    py::object value = config_dict["pathfinder_algorithm"];
                    if (py::isinstance<py::str>(value)) {
                        std::string name = py::cast<std::string>(value);
                        std::transform(name.begin(), name.end(), name.begin(), [](unsigned char ch) {
                            return static_cast<char>(std::tolower(ch));
                        });
                        if (name == "greedy") {
                            config.pathfinder_algorithm = ROCTN_PATHFINDER_ALGO_GREEDY;
                        } else if (name == "kahypar") {
                            config.pathfinder_algorithm = ROCTN_PATHFINDER_ALGO_KAHYPAR;
                        } else if (name == "metis") {
                            config.pathfinder_algorithm = ROCTN_PATHFINDER_ALGO_METIS;
                        } else {
                            throw std::invalid_argument("Unknown TensorNet pathfinder_algorithm: " + name);
                        }
                    } else {
                        config.pathfinder_algorithm =
                            static_cast<rocPathfinderAlgorithm_t>(py::cast<int>(value));
                    }
                }
                if (config_dict.contains("memory_limit_bytes")) {
                    config.memory_limit_bytes = config_dict["memory_limit_bytes"].cast<size_t>();
                } else if (config_dict.contains("memory_limit")) {
                    config.memory_limit_bytes = config_dict["memory_limit"].cast<size_t>();
                }
                if (config_dict.contains("num_slices")) {
                    config.num_slices = config_dict["num_slices"].cast<int>();
                }
            }

            warn_tensornet_pathfinder_fallback(config);
            warn_tensornet_limited_runtime_slicing(config);

            // TODO: The rocblas_handle and hipStream_t should be retrieved from the simulator handle.
            // The current structure of RocsvHandleWrapper makes this difficult without modifying it.
            // Using placeholders for now.
            rocblas_handle blas_h = nullptr; // Placeholder
            hipStream_t stream = 0; // Placeholder

            rocqStatus_t status = rocTensorNetworkContract(self.get(), &config, &result_tensor_py, blas_h, stream);
            
            if (status != ROCQ_STATUS_SUCCESS) {
                throw std::runtime_error(tensornet_contract_status_message(status, config));
            }
        }, py::arg("optimizer_config"), py::arg("result_tensor").noconvert(), "Contracts the tensor network. Result tensor must be pre-allocated.");

    m.def("get_tensornet_capabilities", []() {
        hipTensorNetCapabilities_t caps{};
        rocqStatus_t status = rocTensorNetworkGetCapabilities(&caps);
        if (status != ROCQ_STATUS_SUCCESS) {
            throw std::runtime_error(tensornet_status_message("rocTensorNetworkGetCapabilities", status));
        }
        py::dict result;
        result["supports_c64"] = caps.supports_c64 != 0;
        result["supports_c128"] = caps.supports_c128 != 0;
        result["supports_pathfinder_greedy"] = caps.supports_pathfinder_greedy != 0;
        result["supports_pathfinder_kahypar"] = caps.supports_pathfinder_kahypar != 0;
        result["supports_pathfinder_metis"] = caps.supports_pathfinder_metis != 0;
        result["supports_memory_limit_planning"] = caps.supports_memory_limit_planning != 0;
        result["supports_runtime_slicing"] = caps.supports_runtime_slicing != 0;
        return result;
    }, "Reports TensorNet dtype, optimizer, and slicing capabilities for this build.");

    // --- NEW SVD BINDING ---
    m.def("tensor_svd",
        [](RocTensorNetworkHandleWrapper& handle, const rocquantum::util::rocTensor& A) {
            if (A.rank() != 2) {
                throw std::runtime_error("SVD input must be a 2D tensor (matrix).");
            }
            // NOTE: Workspace management is simplified here. A robust implementation
            // would query rocsolver for the required workspace size.
            DeviceBuffer workspace(1, 1); // Placeholder

            auto U = rocquantum::util::rocTensor();
            auto S = rocquantum::util::rocTensor();
            auto V = rocquantum::util::rocTensor();

            rocqStatus_t status = rocTensorSVD(handle.get(), &U, &S, &V, &A, workspace.ptr_);
            if (status != ROCQ_STATUS_SUCCESS) {
                throw std::runtime_error(tensornet_status_message("rocTensorSVD", status));
            }
            return std::make_tuple(U, S, V);
        }, py::arg("handle"), py::arg("A"), "Performs SVD on a 2D tensor A, returning (U, S, V).");
    // --- END NEW SVD BINDING ---

    // --- GateFusion Bindings ---
    py::class_<rocquantum::GateOp>(m, "GateOp")
        .def(py::init<>())
        .def_readwrite("name", &rocquantum::GateOp::name)
        .def_readwrite("targets", &rocquantum::GateOp::targets)
        .def_readwrite("controls", &rocquantum::GateOp::controls)
        .def_readwrite("params", &rocquantum::GateOp::params);

    py::class_<rocquantum::GateFusion>(m, "GateFusion")
        .def(py::init<RocsvHandleWrapper&, DeviceBuffer&, unsigned>(),
             py::arg("handle"), py::arg("d_state").noconvert(), py::arg("num_qubits"))
        .def("process_queue", [](rocquantum::GateFusion& self, const std::vector<rocquantum::GateOp>& queue) {
            return self.processQueue(queue);
        }, py::arg("queue"));
    // --- End GateFusion Bindings ---

    // --- Conceptual MLIR storage for the legacy API ---
    py::class_<ConceptualMLIRCompiler>(m, "MLIRCompiler")
        .def(py::init<>(), "Initializes the legacy conceptual MLIR holder.")
        .def("initialize_module", &ConceptualMLIRCompiler::initialize_module,
             py::arg("module_name") = "rocq_module",
             "Initializes a conceptual MLIR module string. Returns true on success.")
        .def("dump_module", &ConceptualMLIRCompiler::dump_module,
             "No-op placeholder retained for legacy API compatibility.")
        .def("get_module_string", &ConceptualMLIRCompiler::get_module_string,
             "Returns the stored conceptual MLIR string.")
        .def("load_module_from_string", &ConceptualMLIRCompiler::load_module_from_string,
             py::arg("mlir_string"),
             "Stores a conceptual MLIR string. Returns true on success.")
        .def("run_adjoint_generation_pass", &ConceptualMLIRCompiler::run_adjoint_generation_pass,
             "Returns false because the default legacy binding does not link an MLIR pass pipeline.")
        .def("create_function", &ConceptualMLIRCompiler::create_function,
             py::arg("func_name"),
             py::arg("arg_type_strs"),
             py::arg("result_type_strs") = std::vector<std::string>(),
             "Records a conceptual function declaration for legacy API compatibility.");

}
