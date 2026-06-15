# Experimental QEC Helpers

This package contains a small, executable QEC subset over the canonical `rocq`
runtime. It is not a full CUDA-QX QEC library.

Current supported subset:

- `rocquantum.qec.qec_capabilities()` and the package-level `capabilities()`
  alias expose the experimental supported/unsupported QEC contract, entry
  points, code-family scope, measurement-error model, docs path, and ROCm
  validation limit for CUDA-QX comparisons.
- 3 data qubits plus 2 ancilla qubits.
- One bit-flip repetition-code syndrome round plus sequential repeated-round
  aggregation over the same 3-qubit code.
- End-of-circuit ancilla sampling through `rocq.sample()`.
- Generic `QEC_Experiment.run_single_round()` can execute generated canonical
  stabilizer fragments through `rocq.sample()` when legacy `circuit_ref`
  measurement hooks are unavailable.
- Lookup-table correction through `RepetitionCodeDecoder`.
- Syndrome histogram, repeated-round correction summary, and correction-success
  analysis for sampled counts.
- Optional independent syndrome-bit measurement error mitigation through
  `mitigate_repetition_syndrome_counts()` and the
  `measurement_error_probability=` option on repetition-code analysis and
  execution helpers.
- Execution helpers require positive integer `shots`; generic single-round
  orchestration also requires positive integer `num_qubits` plus in-range
  integer `ancilla_qubit_indices`; repeated-round helpers require positive
  integer `rounds`; and count/bit/syndrome inputs are validated as non-empty
  one- or two-bit binary count keys, non-negative integer counts, length-2
  decoder syndrome bits, and non-boolean data/error/logical bits plus finite
  measurement error probabilities in `[0, 0.5)`.

Minimal example:

```python
from rocquantum.qec import run_repetition_code_single_round

result = run_repetition_code_single_round(error_qubit=1, shots=32)
print(result["syndrome"])
print(result["correction_applied"])
print(result["logical_success_rate"])
print(result["most_likely_corrected_data_bits"])
```

Repeated-round example:

```python
from rocquantum.qec import run_repetition_code_rounds

result = run_repetition_code_rounds(error_qubits=[0, 1], rounds=2, shots=32)
print(result["aggregate_syndrome_histogram"])
print(result["correction_summary"])
print(result["logical_success_rate"])
```

Measurement-error mitigation example:

```python
from rocquantum.qec import analyze_repetition_code_counts

analysis = analyze_repetition_code_counts(
    {"01": 81, "00": 9, "11": 9, "10": 1},
    measurement_error_probability=0.1,
)
print(analysis["mitigated_syndrome_scores"])
```

Limitations:

- No mid-circuit measurement or classical feedback.
- Repeated rounds are sequential sampled helper calls with classical
  most-likely feed-forward, not an in-circuit dynamic-control workflow.
- No general noise-aware decoder beyond deterministic single-X syndrome lookup
  plus the independent syndrome readout-error mitigation helper above.
- No performance-tuned syndrome extraction.
