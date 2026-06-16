from __future__ import annotations

import numpy as np

from qiskit.primitives import BaseSamplerV2
from qiskit.primitives.containers import BitArray, DataBin, PrimitiveResult, SamplerPub, SamplerPubResult
from qiskit.primitives.primitive_job import PrimitiveJob

from rocquantum.framework_runtime import normalize_positive_integer, normalize_shots, qiskit_sample_plan


def _measurement_plan(circuit, measured_bits):
    if measured_bits:
        measured_items = sorted(measured_bits.items())
    else:
        measured_items = [(idx, idx) for idx in range(circuit.num_qubits)]

    sample_qubits, measured_sample_offsets = qiskit_sample_plan(measured_items)
    sample_offsets = {
        classical_index: measured_sample_offsets[offset]
        for offset, (classical_index, _) in enumerate(measured_items)
    }
    registers = []
    for register in circuit.cregs:
        items = []
        for local_index, bit in enumerate(register):
            classical_index = circuit.find_bit(bit).index
            if classical_index in sample_offsets:
                items.append((local_index, sample_offsets[classical_index]))
        if items:
            registers.append((register.name, items, len(register)))

    if not registers:
        registers.append(("meas", [(idx, idx) for idx in range(len(measured_items))], len(measured_items)))

    return sample_qubits, registers


def _classical_register_plan(circuit, measured_bits):
    measured_items = sorted(measured_bits.items())
    measured_classical_bits = {classical_index for classical_index, _ in measured_items}
    registers = []
    for register in circuit.cregs:
        items = []
        for local_index, bit in enumerate(register):
            classical_index = circuit.find_bit(bit).index
            if classical_index in measured_classical_bits:
                items.append((local_index, classical_index))
        if items:
            registers.append((register.name, items, len(register)))

    if not registers:
        registers.append(
            (
                "meas",
                [(offset, classical_index) for offset, (classical_index, _) in enumerate(measured_items)],
                len(measured_items),
            )
        )

    return registers


def _samples_for_register(raw_samples, items, width):
    rows = []
    for raw_sample in raw_samples:
        row = [False] * width
        for local_index, sample_bit in items:
            if (int(raw_sample) >> sample_bit) & 1:
                row[local_index] = True
        rows.append(row)
    return np.asarray(rows, dtype=bool)


def _samples_from_classical_bits(classical_bits, items, width):
    row = [False] * width
    for local_index, classical_index in items:
        row[local_index] = bool(classical_bits.get(int(classical_index), 0))
    return np.asarray([row], dtype=bool)


def _append_register_array(data, register_name, samples):
    if register_name in data:
        data[register_name].append(samples)
    else:
        data[register_name] = [samples]


class RocQuantumSampler(BaseSamplerV2):
    """Native Qiskit SamplerV2 backed by rocQuantum sampling."""

    def __init__(
        self,
        backend,
        *,
        default_shots: int = 1024,
        max_dynamic_loop_iterations: int | None = None,
    ):
        self._backend = backend
        self._default_shots = normalize_shots(default_shots)
        self._max_dynamic_loop_iterations = (
            None
            if max_dynamic_loop_iterations is None
            else normalize_positive_integer(
                max_dynamic_loop_iterations,
                "max_dynamic_loop_iterations",
            )
        )

    def run(self, pubs, *, shots: int | None = None):
        target_shots = self._default_shots if shots is None else normalize_shots(shots)
        coerced_pubs = [SamplerPub.coerce(pub, target_shots) for pub in pubs]
        job = PrimitiveJob(self._run, coerced_pubs)
        job._submit()
        return job

    def _run(self, pubs):
        return PrimitiveResult([self._run_pub(pub) for pub in pubs], metadata={"version": 2})

    def _run_pub(self, pub):
        batched_result = self._try_run_pub_batched_parameters(pub)
        if batched_result is not None:
            return batched_result

        bound_circuits = np.asarray(pub.parameter_values.bind_all(pub.circuit), dtype=object)
        data = {}
        used_shot_trajectory = False

        for index in np.ndindex(pub.shape):
            circuit = bound_circuits[index]
            trajectory_samples = self._try_run_trajectory_samples(circuit, int(pub.shots))
            if trajectory_samples is not None:
                used_shot_trajectory = True
                for register_name, register_samples in trajectory_samples.items():
                    _append_register_array(data, register_name, register_samples)
                continue

            measured_bits = self._backend._apply_circuit(circuit)
            sample_qubits, registers = _measurement_plan(circuit, measured_bits)
            raw_samples = self._backend._runtime.measure(
                sample_qubits,
                int(pub.shots),
            )
            for register_name, items, width in registers:
                register_samples = _samples_for_register(raw_samples, items, width)
                _append_register_array(data, register_name, register_samples)

        shaped_data = {}
        for register_name, arrays in data.items():
            if pub.shape:
                sample_rows = np.asarray(arrays, dtype=bool).reshape(pub.shape + (int(pub.shots), arrays[0].shape[-1]))
                shaped_data[register_name] = BitArray.from_bool_array(sample_rows, order="little")
            else:
                shaped_data[register_name] = BitArray.from_bool_array(arrays[0], order="little")

        metadata = {
            "shots": int(pub.shots),
            "circuit_metadata": getattr(pub.circuit, "metadata", None) or {},
            "native": True,
        }
        if used_shot_trajectory:
            metadata["shot_trajectory"] = True

        return SamplerPubResult(
            DataBin(**shaped_data, shape=pub.shape),
            metadata=metadata,
        )

    def _try_run_trajectory_samples(self, circuit, shots):
        if self._backend._has_dynamic_circuit(circuit):
            return self._sample_dynamic_circuit(circuit, shots)
        if self._backend._has_runtime_reset(circuit):
            return self._sample_runtime_reset_circuit(circuit, shots)
        return None

    def _sample_runtime_reset_circuit(self, circuit, shots):
        data = {}
        for _ in range(int(shots)):
            measured_bits = self._backend._apply_circuit(
                circuit,
                include_global_phase=False,
                allow_runtime_reset=True,
            )
            sample_qubits, registers = _measurement_plan(circuit, measured_bits)
            raw_samples = self._backend._runtime.measure(sample_qubits, 1)
            for register_name, items, width in registers:
                register_samples = _samples_for_register(raw_samples, items, width)
                _append_register_array(data, register_name, register_samples[0])
        return {
            register_name: np.asarray(rows, dtype=bool)
            for register_name, rows in data.items()
        }

    def _sample_dynamic_circuit(self, circuit, shots):
        data = {}
        max_iterations = self._dynamic_loop_limit()
        for _ in range(int(shots)):
            measured_bits, classical_bits = self._backend._apply_circuit_trajectory(
                circuit,
                max_dynamic_loop_iterations=max_iterations,
            )
            if measured_bits:
                registers = _classical_register_plan(circuit, measured_bits)
                for register_name, items, width in registers:
                    register_samples = _samples_from_classical_bits(classical_bits, items, width)
                    _append_register_array(data, register_name, register_samples[0])
            else:
                sample_qubits, registers = _measurement_plan(circuit, measured_bits)
                raw_samples = self._backend._runtime.measure(sample_qubits, 1)
                for register_name, items, width in registers:
                    register_samples = _samples_for_register(raw_samples, items, width)
                    _append_register_array(data, register_name, register_samples[0])
        return {
            register_name: np.asarray(rows, dtype=bool)
            for register_name, rows in data.items()
        }

    def _dynamic_loop_limit(self):
        if self._max_dynamic_loop_iterations is not None:
            return self._max_dynamic_loop_iterations
        return normalize_positive_integer(
            getattr(
                getattr(self._backend, "options", None),
                "max_dynamic_loop_iterations",
                1024,
            ),
            "max_dynamic_loop_iterations",
        )

    def _try_run_pub_batched_parameters(self, pub):
        if not pub.shape:
            return None

        bound_circuits = np.asarray(pub.parameter_values.bind_all(pub.circuit), dtype=object)
        parameter_indices = list(np.ndindex(pub.shape))
        if len(parameter_indices) <= 1:
            return None

        circuits = [bound_circuits[index] for index in parameter_indices]
        try:
            measured_bits = self._backend._apply_circuit_batch(circuits)
            sample_qubits, registers = _measurement_plan(circuits[0], measured_bits)
            raw_samples_batch = self._backend._runtime.measure_batch(sample_qubits, int(pub.shots))
        except (NotImplementedError, RuntimeError, TypeError, ValueError):
            return None

        raw_samples_batch = np.asarray(raw_samples_batch, dtype=np.int64)
        if raw_samples_batch.shape != (len(parameter_indices), int(pub.shots)):
            return None

        data = {}
        for batch_index, _ in enumerate(parameter_indices):
            raw_samples = raw_samples_batch[batch_index]
            for register_name, items, width in registers:
                register_samples = _samples_for_register(raw_samples, items, width)
                if register_name in data:
                    data[register_name].append(register_samples)
                else:
                    data[register_name] = [register_samples]

        shaped_data = {}
        for register_name, arrays in data.items():
            sample_rows = np.asarray(arrays, dtype=bool).reshape(
                pub.shape + (int(pub.shots), arrays[0].shape[-1])
            )
            shaped_data[register_name] = BitArray.from_bool_array(sample_rows, order="little")

        return SamplerPubResult(
            DataBin(**shaped_data, shape=pub.shape),
            metadata={
                "shots": int(pub.shots),
                "circuit_metadata": getattr(pub.circuit, "metadata", None) or {},
                "native": True,
                "batched_parameters": True,
            },
        )
