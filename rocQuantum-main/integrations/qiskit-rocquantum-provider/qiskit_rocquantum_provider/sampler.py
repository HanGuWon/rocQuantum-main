from __future__ import annotations

import numpy as np

from qiskit.primitives import BaseSamplerV2
from qiskit.primitives.containers import BitArray, DataBin, PrimitiveResult, SamplerPub, SamplerPubResult
from qiskit.primitives.primitive_job import PrimitiveJob

from rocquantum.framework_runtime import qiskit_sample_plan


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


def _samples_for_register(raw_samples, items, width):
    rows = []
    for raw_sample in raw_samples:
        row = [False] * width
        for local_index, sample_bit in items:
            if (int(raw_sample) >> sample_bit) & 1:
                row[local_index] = True
        rows.append(row)
    return np.asarray(rows, dtype=bool)


class RocQuantumSampler(BaseSamplerV2):
    """Native Qiskit SamplerV2 backed by rocQuantum sampling."""

    def __init__(self, backend, *, default_shots: int = 1024):
        self._backend = backend
        self._default_shots = int(default_shots)

    def run(self, pubs, *, shots: int | None = None):
        target_shots = self._default_shots if shots is None else int(shots)
        coerced_pubs = [SamplerPub.coerce(pub, target_shots) for pub in pubs]
        job = PrimitiveJob(self._run, coerced_pubs)
        job._submit()
        return job

    def _run(self, pubs):
        return PrimitiveResult([self._run_pub(pub) for pub in pubs], metadata={"version": 2})

    def _run_pub(self, pub):
        bound_circuits = np.asarray(pub.parameter_values.bind_all(pub.circuit), dtype=object)
        data = {}

        for index in np.ndindex(pub.shape):
            circuit = bound_circuits[index]
            measured_bits = self._backend._apply_circuit(circuit)
            sample_qubits, registers = _measurement_plan(circuit, measured_bits)
            raw_samples = self._backend._runtime.measure(
                sample_qubits,
                int(pub.shots),
            )
            for register_name, items, width in registers:
                register_samples = _samples_for_register(raw_samples, items, width)
                if register_name in data:
                    data[register_name].append(register_samples)
                else:
                    data[register_name] = [register_samples]

        shaped_data = {}
        for register_name, arrays in data.items():
            if pub.shape:
                sample_rows = np.asarray(arrays, dtype=bool).reshape(pub.shape + (int(pub.shots), arrays[0].shape[-1]))
                shaped_data[register_name] = BitArray.from_bool_array(sample_rows, order="little")
            else:
                shaped_data[register_name] = BitArray.from_bool_array(arrays[0], order="little")

        return SamplerPubResult(
            DataBin(**shaped_data, shape=pub.shape),
            metadata={
                "shots": int(pub.shots),
                "circuit_metadata": getattr(pub.circuit, "metadata", None) or {},
                "native": True,
            },
        )
