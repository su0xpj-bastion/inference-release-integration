"""First-pass triage: turn N red cases into a few root-cause buckets with an owner.

Three techniques, in order of cost:
  1. localize  — re-run with a device trace; diff every tensor the runtime READ
                 and WROTE against the float64 golden; the first divergence
                 names the call, the tensor, and whether it was a read or write.
                 Read-diverges-but-producer-was-right => memory hazard (who
                 corrupted it: prefetch staging -> runtime, SRAM slot -> planner).
                 Write-diverges-with-correct-inputs => that kernel.
  2. isolate   — drop one feature / dtype / shape at a time; what's left is the
                 minimal trigger. Tells the owner how to reproduce in their suite.
  3. bucket    — group by localization signature, not by error text; 80 red
                 cases from 3 bugs should read as 3 tickets.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from corestack.compiler import CompileError
from corestack.graph import SHAPES
from corestack.versions import History
from rit.matrix import Case
from rit.runner import TOLERANCE, Result, execute_case, run_case

ENCODING = "utf-8"
OWNERS_FILE = Path(__file__).with_name("owners.json")
KNOWN_ISSUES_FILE = Path(__file__).with_name("known_issues.json")


@dataclass(frozen=True)
class Localization:
    layer: str            # compiler | runtime | kernels | infra | unknown
    diagnosis: str
    kernel: str = ""
    tensor_role: str = ""   # normalized tensor name, e.g. 'k' from 'a.k'
    access: str = ""        # "reading" | "writing"

    @property
    def key(self) -> str:
        return f"{self.layer}:{self.diagnosis}:{self.kernel}"


@dataclass
class Bucket:
    key: str
    localization: Localization
    results: list[Result] = field(default_factory=list)
    trigger: list[str] = field(default_factory=list)
    owner: str = ""
    known_issue: str | None = None
    representative: str = ""

    def conditions(self) -> dict[str, dict[str, int]]:
        cases = [Case.parse(r.case_id) for r in self.results]
        return {
            "model": dict(Counter(c.model for c in cases)),
            "shape": dict(Counter(c.shape for c in cases)),
            "dtype": dict(Counter(c.dtype for c in cases)),
        }


def _norm(tensor: str) -> str:
    return tensor.split(".", 1)[-1]


def localize(case: Case, history: History, sha: str) -> Localization:
    try:
        run, ref = execute_case(case, history, sha, trace=True)
    except CompileError as exc:
        return Localization("compiler", "compile error: " + re.sub(r"\d+", "N", str(exc)))
    tol = TOLERANCE[case.dtype]
    written_ok: dict[str, bool] = {"x": True}
    for e in run.trace:
        ref_val = ref[e.tensor]
        val = e.value.astype(np.float64)
        finite = np.isfinite(val).all()
        err = np.abs(val - ref_val).max() / max(np.abs(ref_val).max(), 1e-12) if finite else np.inf
        if e.role == "out":
            written_ok[e.tensor] = err <= tol
            if err > tol:
                diag = "non-finite output" if not finite else "output numerically wrong with correct inputs"
                return Localization("kernels", diag, e.kernel, _norm(e.tensor), "writing")
            continue
        if err > tol and written_ok.get(e.tensor, False):
            if e.prefetched:
                return Localization("runtime", "stale prefetched read (RAW hazard)", e.kernel, _norm(e.tensor), "reading")
            return Localization("compiler", "input clobbered in SRAM after write (buffer aliasing)",
                                e.kernel, _norm(e.tensor), "reading")
    return Localization("unknown", "no tensor diverged under trace (nondeterministic?)")


def isolate(case: Case, history: History, sha: str) -> list[str]:
    """Minimal trigger: each feature/dtype/shape that, when removed, makes the case pass."""
    def passes(c: Case) -> bool:
        return run_case(c, history, sha).status == "pass"

    needed = [f for f in case.flags if passes(replace(case, flags=tuple(x for x in case.flags if x != f)))]
    if case.dtype == "fp16" and passes(replace(case, dtype="fp32")):
        needed.append("dtype=fp16")
    smaller = [s for s in SHAPES if SHAPES[s] < SHAPES[case.shape]]
    if smaller and all(passes(replace(case, shape=s)) for s in smaller):
        needed.append(f"shape>={case.shape}")
    return needed or ["(always fails at this commit)"]


def _load_json(path: Path) -> dict | list:
    with path.open("r", encoding=ENCODING) as fh:
        return json.load(fh)


def triage(results: list[Result], history: History) -> list[Bucket]:
    owners = _load_json(OWNERS_FILE)
    known = _load_json(KNOWN_ISSUES_FILE)
    buckets: dict[str, Bucket] = {}
    for r in results:
        if r.status == "pass":
            continue
        if r.status == "infra":
            loc = Localization("infra", "device timeout persisted through retries")
        else:
            loc = localize(Case.parse(r.case_id), history, r.sha)
        b = buckets.setdefault(loc.key, Bucket(loc.key, loc))
        b.results.append(r)

    for b in buckets.values():
        rep = min(b.results, key=lambda r: (len(Case.parse(r.case_id).flags), SHAPES[Case.parse(r.case_id).shape]))
        b.representative = rep.case_id
        if b.localization.layer not in ("infra", "unknown"):
            b.trigger = isolate(Case.parse(rep.case_id), history, rep.sha)
        b.owner = owners.get(b.localization.layer, owners["unknown"])
        for ki in known:
            if re.search(ki["match"], b.key):
                b.known_issue = ki["id"]
    return sorted(buckets.values(), key=lambda b: -len(b.results))


def group_counts(buckets: list[Bucket]) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for b in buckets:
        out[b.localization.layer] += len(b.results)
    return dict(out)
