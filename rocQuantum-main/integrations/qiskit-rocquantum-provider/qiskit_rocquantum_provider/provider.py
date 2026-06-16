from .backend import RocQuantumBackend

try:
    from qiskit.providers import ProviderV1 as _ProviderBase
except ImportError:  # Qiskit 1.x/2.x no longer requires ProviderV1 for local providers.
    _ProviderBase = object


def _reject_unknown_native_options(options, primitive_name):
    if options:
        option_names = ", ".join(sorted(str(name) for name in options))
        raise ValueError(f"Unsupported native {primitive_name} option(s): {option_names}")


def _normalize_native_option(native):
    if not isinstance(native, bool):
        raise ValueError("native must be a boolean.")
    return native


class RocQuantumProvider(_ProviderBase):
    """
    Provider for the rocQuantum simulator backend.
    """
    def __init__(self):
        try:
            super().__init__()
        except TypeError:
            pass
        self.name = "rocquantum_provider"
        # Instantiate and store the backend instance
        self._backends = {"rocq_simulator": RocQuantumBackend(provider=self)}

    def backends(self, name=None, **kwargs):
        """Return a list of backends."""
        if name is not None:
            backend = self._backends.get(name)
            return [backend] if backend is not None else []
        return list(self._backends.values())

    def get_backend(self, name=None, **kwargs):
        """Return a single backend by name, matching Qiskit provider ergonomics."""
        matches = self.backends(name=name, **kwargs)
        if not matches:
            raise KeyError(f"No rocQuantum backend named {name!r}.")
        if len(matches) > 1:
            raise ValueError(f"More than one backend matches {name!r}.")
        return matches[0]

    def get_sampler(self, name="rocq_simulator", native=True, **options):
        """Return a Qiskit SamplerV2 backed by rocQuantum."""
        native = _normalize_native_option(native)
        if native:
            from .sampler import RocQuantumSampler

            default_shots = options.pop("default_shots", 1024) if options else 1024
            max_dynamic_loop_iterations = (
                options.pop("max_dynamic_loop_iterations", None)
                if options
                else None
            )
            _reject_unknown_native_options(options, "sampler")
            return RocQuantumSampler(
                self.get_backend(name),
                default_shots=default_shots,
                max_dynamic_loop_iterations=max_dynamic_loop_iterations,
            )

        from qiskit.primitives import BackendSamplerV2

        return BackendSamplerV2(backend=self.get_backend(name), options=options or None)

    def get_estimator(self, name="rocq_simulator", native=True, **options):
        """Return a Qiskit EstimatorV2 backed by rocQuantum."""
        native = _normalize_native_option(native)
        if native:
            from .estimator import RocQuantumEstimator

            default_precision = options.pop("default_precision", 0.0) if options else 0.0
            _reject_unknown_native_options(options, "estimator")
            return RocQuantumEstimator(
                self.get_backend(name),
                default_precision=default_precision,
            )

        from qiskit.primitives import BackendEstimatorV2

        return BackendEstimatorV2(backend=self.get_backend(name), options=options or None)

    def estimate_expectation(self, circuit, observable, name="rocq_simulator"):
        """Evaluate a supported observable through the native rocQuantum simulator."""
        return self.get_backend(name).estimate_expectation(circuit, observable)
