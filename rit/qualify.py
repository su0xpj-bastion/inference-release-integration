"""Release qualification: evidence in, GO / NO-GO / INCONCLUSIVE out.

INCONCLUSIVE is deliberate. When infra noise is high, a green run is not
evidence of quality and a red run is not evidence of a defect; shipping or
blocking on that signal is a coin flip. The gate refuses to decide instead.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field

from corestack.versions import History
from rit.bisect import BisectError, BisectResult, bisect
from rit.matrix import Case
from rit.runner import Result
from rit.triage import Bucket, triage

ENCODING = "utf-8"
MAX_INFRA_RATE = 0.02        # above this the run is not trustworthy
MAX_CYCLE_REGRESSION = 0.05  # per-case device-cycle budget vs baseline


@dataclass
class Gate:
    name: str
    passed: bool
    detail: str


@dataclass
class Qualification:
    release: str
    sha: str
    baseline: str
    decision: str
    gates: list[Gate]
    comparison: dict[str, list[str]]
    buckets: list[Bucket] = field(default_factory=list)
    bisects: list[BisectResult] = field(default_factory=list)
    perf: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)


def compare(baseline: list[Result], current: list[Result]) -> dict[str, list[str]]:
    base = {r.case_id: r for r in baseline}
    out: dict[str, list[str]] = {k: [] for k in ("new_failure", "still_failing", "fixed", "flaky", "infra", "new_case")}
    for r in current:
        b = base.get(r.case_id)
        red = r.status in ("fail", "error")
        if r.status == "infra":
            out["infra"].append(r.case_id)
        elif b is None:
            out["new_case"].append(r.case_id)
            if red:
                out["new_failure"].append(r.case_id)
        elif red and b.status == "pass":
            out["new_failure"].append(r.case_id)
        elif red:
            out["still_failing"].append(r.case_id)
        elif b.status in ("fail", "error"):
            out["fixed"].append(r.case_id)
        if r.flaky:
            out["flaky"].append(r.case_id)
    return out


def perf_delta(baseline: list[Result], current: list[Result]) -> dict[str, float]:
    base = {r.case_id: r.cycles for r in baseline if r.status == "pass" and r.cycles}
    ratios = {r.case_id: r.cycles / base[r.case_id] for r in current
              if r.status == "pass" and r.cycles and r.case_id in base}
    if not ratios:
        return {"geomean": 1.0, "worst": 1.0, "regressed": 0}
    return {
        "geomean": math.exp(sum(math.log(x) for x in ratios.values()) / len(ratios)),
        "worst": max(ratios.values()),
        "regressed": sum(1 for x in ratios.values() if x > 1 + MAX_CYCLE_REGRESSION),
    }


def qualify(release: str, current: list[Result], baseline_release: str, baseline: list[Result],
            history: History, *, planned: list[Case] | None = None, do_bisect: bool = True) -> Qualification:
    sha, base_sha = history.resolve(release), history.resolve(baseline_release)
    cmp = compare(baseline, current)
    buckets = triage(current, history)
    waived = {r.case_id for b in buckets if b.known_issue and _waived(b.known_issue) for r in b.results}
    unwaived_red = [r.case_id for r in current if r.status in ("fail", "error") and r.case_id not in waived]
    infra_rate = (len(cmp["infra"]) + len(cmp["flaky"])) / max(len(current), 1)
    perf = perf_delta(baseline, current)
    ran = {r.case_id for r in current}
    missing_p0 = [c.case_id for c in (planned or []) if c.tier == "P0" and c.case_id not in ran]

    gates = [
        Gate("coverage: every planned P0 case executed", not missing_p0,
             f"{len(missing_p0)} P0 case(s) missing" if missing_p0 else "all P0 cases executed"),
        Gate("signal quality: infra + flaky rate <= 2%", infra_rate <= MAX_INFRA_RATE, f"{infra_rate:.1%}"),
        Gate("correctness: zero unwaived failures", not unwaived_red,
             f"{len(unwaived_red)} unwaived failure(s) in {len([b for b in buckets if b.localization.layer != 'infra'])} bucket(s)"),
        Gate("regression: zero new failures vs baseline", not [c for c in cmp["new_failure"] if c not in waived],
             f"{len(cmp['new_failure'])} new vs {baseline_release}"),
        Gate(f"performance: no case > +{MAX_CYCLE_REGRESSION:.0%} device cycles", perf["regressed"] == 0,
             f"geomean x{perf['geomean']:.3f}, worst x{perf['worst']:.3f}, {perf['regressed']} regressed"),
    ]
    if not gates[1].passed or not gates[0].passed:
        decision = "INCONCLUSIVE"
    elif all(g.passed for g in gates):
        decision = "GO"
    else:
        decision = "NO-GO"

    bisects: list[BisectResult] = []
    if do_bisect:
        for b in buckets:
            if b.localization.layer in ("infra", "unknown"):
                continue
            try:
                bisects.append(bisect(b.representative, base_sha, sha, history))
            except BisectError:
                pass
    counts = dict(Counter(r.status for r in current))
    return Qualification(release, sha, baseline_release, decision, gates, cmp, buckets, bisects, perf, counts)


def _waived(issue_id: str) -> bool:
    from rit.triage import KNOWN_ISSUES_FILE
    with KNOWN_ISSUES_FILE.open("r", encoding=ENCODING) as fh:
        return any(k["id"] == issue_id and k.get("waived") for k in json.load(fh))
