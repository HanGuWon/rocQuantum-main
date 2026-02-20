# Python API Exposure Policy (Draft)

## Rule
If documented as stable, it must run end-to-end without skeleton behavior.

## Status Labels
- `Implemented`: tested and wired to real backend behavior.
- `Partial`: callable but with explicit functional limits.
- `Experimental`: opt-in, may fail on missing provider/runtime.
- `Skeleton`: not user-exposed by default.

## Enforcement
- Backend registry must include status metadata.
- `set_target()` must block `Skeleton` unless explicit experimental override is set.
- Error messages must include status and required prerequisites.

## Immediate Application
- Keep stable defaults to implemented providers.
- Mark incomplete provider stubs as `Experimental` or remove from default registry.
- Add capability listing API so callers can introspect status before running jobs.
