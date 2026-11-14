"""Bisect a failing case across the stack history, then say whether the first bad
commit is the ROOT CAUSE or only EXPOSED a latent bug in another component.

The distinction matters for routing: git-bisect blames the commit that turned
the test red; if that commit's component differs from the layer localization
blames, sending the ticket to the bisected author wastes a day.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from corestack.versions import Commit, History
from rit.matrix import Case
from rit.runner import run_case
from rit.triage import Localization, localize


class BisectError(Exception):
    pass


@dataclass
class BisectResult:
    case_id: str
    first_bad: Commit
    steps: list[tuple[str, str]] = field(default_factory=list)   # (sha, status)
    localization: Localization | None = None

    @property
    def verdict(self) -> str:
        if self.localization is None or self.localization.layer in ("unknown", "infra"):
            return "root cause unconfirmed"
        if self.localization.layer == self.first_bad.component:
            return "root cause"
        return f"exposing change; latent defect in {self.localization.layer}"


def _status(case: Case, history: History, sha: str, steps: list) -> str:
    st = run_case(case, history, sha).status
    if st == "infra":
        raise BisectError(f"{sha}: infra failure persisted through retries; cannot classify")
    steps.append((sha, st))
    return st


def bisect(case_id: str, good: str, bad: str, history: History) -> BisectResult:
    case = Case.parse(case_id)
    steps: list[tuple[str, str]] = []
    if _status(case, history, good, steps) != "pass":
        raise BisectError(f"'good' commit {good} does not pass {case_id}")
    if _status(case, history, bad, steps) == "pass":
        raise BisectError(f"'bad' commit {bad} passes {case_id}")
    commits = history.between(good, bad)
    lo, hi = -1, len(commits) - 1          # invariant: lo passes (or is 'good'), hi fails
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if _status(case, history, commits[mid].sha, steps) == "pass":
            lo = mid
        else:
            hi = mid
    first_bad = commits[hi]
    return BisectResult(case_id, first_bad, steps, localize(case, history, first_bad.sha))
