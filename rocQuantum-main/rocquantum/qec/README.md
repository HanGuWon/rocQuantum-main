# Experimental QEC Helpers

This package contains a small, executable QEC subset over the canonical `rocq`
runtime. It is not a full CUDA-QX QEC library.

Current supported subset:

- 3 data qubits plus 2 ancilla qubits.
- One bit-flip repetition-code syndrome round.
- End-of-circuit ancilla sampling through `rocq.sample()`.
- Lookup-table correction through `RepetitionCodeDecoder`.

Minimal example:

```python
from rocquantum.qec import run_repetition_code_single_round

result = run_repetition_code_single_round(error_qubit=1, shots=32)
print(result["syndrome"])
print(result["correction_applied"])
```

Limitations:

- No mid-circuit measurement or classical feedback.
- No repeated rounds.
- No noise-aware decoder.
- No performance-tuned syndrome extraction.
