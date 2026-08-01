# Experimental QEC Helpers

This package contains a correctness-oriented QEC subset over NumPy and the
canonical `rocq` runtime. It is not a full CUDA-QX QEC library and makes no
tensor-network, real-time-decoder, device-resident, or native ROCm performance
parity claim.

Current supported subset:

- Official-style host code metadata and registry APIs: `Code`, `CodeMetadata`,
  `@code()` / `register_code()`, `get_code()`, and `get_available_codes()`.
  Built-ins are arbitrary odd-distance `get_code("repetition", distance=d)`
  metadata and the `[[7, 1, 3]]` `get_code("steane")` CSS metadata. Dense
  `uint8` `H`, `Hx`, `Hz`, logical-X, and logical-Z matrices are checked over
  GF(2), including stabilizer commutation and complete logical-family pairing.
  Basis-specific codes may expose only one logical-observable family: the
  repetition code follows CUDA-QX with logical-X shape `(0, d)` and logical-Z
  shape `(1, d)`. `Code.get_stabilizers()` returns `rocq` Pauli operators whose
  `get_pauli_word()` preserves the full code width and trailing identities.
- Official-style host decoder contracts: `DecoderResult`,
  `BatchDecoderResult`, `AsyncDecoderResult.get()`, `Decoder`, decoder
  registration/factories, `SingleErrorLUTDecoder`, and a pure-NumPy
  `BeliefPropagationDecoder`. Scalar, batch, and asynchronous entry points
  validate parity matrices, syndrome probabilities, and result probabilities.
- Reproducible pure-NumPy `generate_random_bit_flips()` and
  `sample_code_capacity()` helpers. Capacity sampling returns
  `(syndromes, errors)` and guarantees `syndromes == errors @ H.T mod 2`.
  A `Code` argument uses its full binary-symplectic `H`; pass `code.Hx` or
  `code.Hz` explicitly to model only one Pauli error channel.
- `rocquantum.qec.qec_capabilities()` and the package-level `capabilities()`
  alias expose the experimental supported/unsupported QEC contract, entry
  points, code-family scope, measurement-error model, execution scope,
  hardware-evidence boundary, docs path, and ROCm validation limit for CUDA-QX
  comparisons. The execution scope is sequential sampled classical
  post-processing over canonical `rocq.sample()` calls, not in-circuit dynamic
  feedback or distributed QEC execution.
- 3 data qubits plus 2 ancilla qubits; concrete repetition-code circuit
  generation requires a positive integer `num_qubits >= 5` and an
  `initial_state_kernel` that is callable or `None`.
- One bit-flip repetition-code syndrome round plus sequential repeated-round
  aggregation over the same 3-qubit code.
- End-of-circuit ancilla sampling through `rocq.sample()`.
- Generic `QEC_Experiment.run_single_round()` can execute generated canonical
  stabilizer fragments through `rocq.sample()` when legacy `circuit_ref`
  measurement hooks are unavailable. Custom code and decoder objects must expose
  callable `generate_stabilizer_circuits()`, `define_logical_operators()`, and
  `decode()` methods, and generated stabilizer fragments must be returned as a
  non-empty, non-string, non-mapping sequence or iterable. The optional
  `initial_state_kernel` must be callable or `None`. Logical operators must be
  returned as a string-keyed mapping of canonical `rocq.operator.PauliOperator`
  values, and decoder results must be canonical `rocq.operator.PauliOperator`
  corrections.
- Lookup-table correction through `RepetitionCodeDecoder`.
- Syndrome histogram, repeated-round correction summary, and correction-success
  analysis for sampled counts.
- Optional independent syndrome-bit measurement error mitigation through
  `mitigate_repetition_syndrome_counts()` and the
  `measurement_error_probability=` option on repetition-code analysis and
  execution helpers.
- Execution helpers require positive integer `shots`; generic single-round
  orchestration also requires canonical runtime backend names, positive integer
  `num_qubits` plus unique in-range integer `ancilla_qubit_indices`, and boolean
  `verbose` options; repeated-round helpers require positive integer `rounds`;
  and count/bit/syndrome inputs are validated as non-empty
  one-bit ancilla sample keys, one- or two-bit repetition-code count keys,
  non-negative integer counts, length-2 decoder syndrome bits, and non-boolean
  data/error/logical bits plus finite measurement error probabilities in
  `[0, 0.5)`.

Minimal example:

```python
from rocquantum.qec import run_repetition_code_single_round

result = run_repetition_code_single_round(error_qubit=1, shots=32)
print(result["syndrome"])
print(result["correction_applied"])
print(result["logical_success_rate"])
print(result["most_likely_corrected_data_bits"])
```

Host code/decoder example:

```python
from rocquantum.qec import get_code, get_decoder, sample_code_capacity

code = get_code("repetition", distance=5)
decoder = get_decoder("single_error_lut", code.Hz)
syndromes, errors = sample_code_capacity(
    code, num_shots=16, error_probability=0.05, seed=7
)
decoded = decoder.decode_batch(syndromes)
print(decoded.converged)
print(decoded.result)
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

- No in-circuit mid-circuit measurement or dynamic classical feedback.
- Repeated rounds are sequential sampled helper calls with classical
  most-likely feed-forward, not an in-circuit dynamic-control workflow.
- The belief-propagation decoder is a host correctness implementation, not a
  performance-tuned production decoder and not CUDA-QX tensor-network decoder
  parity.
- No performance-tuned syndrome extraction.
- No native ROCm execution evidence is claimed until AMD GPU hardware CI is
  available.
