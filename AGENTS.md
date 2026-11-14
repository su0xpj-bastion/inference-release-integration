# inference-release-integration — agent guardrails

## Scope
- `corestack/` is the system under test. Seeded defects B1-B3 are intentional; never "fix" them in old versions.
  A new defect gets a new commit in `history.json` plus an end-to-end test proving the harness catches it.
- `rit/` is the harness. A harness change must keep `tests/integration/test_release_qualification.py` green.

## Rules
- The golden (`corestack/graph.py::run_reference`) shares no code with compiler/runtime/kernels.
- Retry only `DeviceTimeout` (infra). Never retry product failures.
- `MAX_ATTEMPTS`, never `MAX_RETRIES`. Module-level constants for `ENCODING`, `TOLERANCE`.
- `from __future__ import annotations` in every module. JSON via `Path.open("r", encoding=...)` + `json.load`.
- Version comparison through `corestack.versions.v()`, never string compare.
- If the matrix or tiers change, update the README cost-vs-detection table and `test_detection_by_mode` together.

## Done means
`python -m pytest` green, and `python -m rit qualify` for rc1 = NO-GO (exit 1) and rc2 = GO (exit 0).
