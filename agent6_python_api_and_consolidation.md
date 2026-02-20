# Agent 6 - Python API Skeleton Cleanup + Consolidation Inputs

## Evidence Table
| Area | Evidence | Observation |
|---|---|---|
| Backend registry includes skeletons as normal targets | `rocQuantum-main/rocquantum/core.py:21`-`30` | Skeleton providers are exposed without explicit policy guard. |
| Skeleton backend implementations | `rocQuantum-main/rocquantum/backends/iqm.py:1`-`7`, `rocQuantum-main/rocquantum/backends/alice_bob.py:1`-`10` | Methods are `pass` placeholders. |
| High-level backend mock fallback | `rocQuantum-main/rocq/backends.py:199`-`204` | Mock state backend can hide missing runtime depth. |
| Core abstract backend contract | `rocQuantum-main/rocquantum/backends/base.py:47`-`59` | Stable interface exists for policy enforcement. |
| Test coverage asymmetry | `rocQuantum-main/tests/test_iqm_backend.py:1`, `rocQuantum-main/tests/test_alice_bob_backend.py:1` | Some provider tests are TODO-only placeholders. |

## API Inventory Snapshot
| API Surface | Status | Evidence |
|---|---|---|
| `rocq.kernel` + MLIR emission | Implemented/Partial | `rocQuantum-main/rocq/kernel.py` |
| `rocq` local simulation backends | Partial | `rocQuantum-main/rocq/backends.py` |
| `rocquantum.core.set_target` provider registry | Partial (policy missing) | `rocQuantum-main/rocquantum/core.py:13`-`50` |
| IonQ/Pasqal/Infleqtion/Quantinuum/Rigetti providers | Partial to Implemented | `rocQuantum-main/rocquantum/backends/*.py` |
| IQM/QuEra/ORCA/SEEQC/QuantumMachines/AliceBob/Xanadu | Skeleton | placeholder files with `pass` |

## Top 5 Findings
- User-facing provider registry does not distinguish stable vs skeleton APIs.
- Skeleton backends can be selected without explicit opt-in.
- Backend capability discovery API is missing.
- Tests do not enforce policy on skeleton exposure.
- Mock fallback in local backend masks backend-depth issues.

## Top 5 Actions
- Introduce backend metadata catalog with `status` and `constraints`.
- Block skeleton backends by default in `set_target()` unless explicit experimental opt-in.
- Add `list_backends(include_experimental=False)` capability endpoint.
- Convert TODO-only provider tests into skip-marked policy tests with explicit status checks.
- Publish and enforce `docs/updates/python_api_policy.md`.

## Consolidation Inputs (for final plan/backlog)
- Dependency order: CI/runtime gating -> distributed completeness -> tensornet optimizer/dtypes -> compiler execute path -> Python exposure tightening.
- P0 should include policy + CI + distributed top-N gaps only.

## Backlog JSON Fragment
```json
[
  {
    "id": "P0-PYAPI-001",
    "area": "PythonAPI",
    "title": "Gate skeleton backends behind explicit experimental opt-in",
    "priority": "P0",
    "impact": "High",
    "effort": "S"
  },
  {
    "id": "P1-PYAPI-002",
    "area": "PythonAPI",
    "title": "Add backend capability listing API and status metadata",
    "priority": "P1",
    "impact": "Med",
    "effort": "S"
  }
]
```

## Acceptance Criteria
- Default API exposure excludes skeleton providers unless user opts in.
- Capability listing clearly reports `Implemented/Partial/Experimental/Skeleton`.
- Tests assert policy behavior.

## Test Plan
- Verified here: source evidence and policy draft.
- Requires ROCm CI: only for local simulation runtime-specific behavior, not provider policy gating.

## Risks
- Tightening default exposure may break existing ad-hoc scripts that used skeleton targets.
- Requires migration note and compatibility switch.
