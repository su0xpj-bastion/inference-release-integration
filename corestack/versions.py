"""Component version history: the inference-core 'monorepo' the harness qualifies.

Each commit bumps exactly one component. A commit's stack state is the
cumulative versions up to and including it. Three defects are seeded, each
one invisible to the owning component's own unit tests:

  B1 kernels 2.1   fp16 fused-attention fast path skips max-subtraction
                   -> exp() overflow -> NaN, only with compiler fusion ON,
                   fp16, and un-tempered logits (attention_hot)
  B2 compiler 3.2  planner refactor computes tensor lifetime from FIRST use
                   -> residual input's buffer is reused early -> silent
                   corruption, only when SRAM pressure forces buffer reuse
  B3 runtime 2.4   async prefetch hazard check only looks at inputs[0]
                   -> stale read; LATENT until compiler 3.3's scheduler puts
                   the K projection right before the scores matmul
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ENCODING = "utf-8"
HISTORY_FILE = Path(__file__).with_name("history.json")


def v(version: str) -> tuple[int, ...]:
    """'3.10' > '3.9': compare numerically, never as strings."""
    return tuple(int(p) for p in version.split("."))


@dataclass(frozen=True)
class Commit:
    sha: str
    component: str
    version: str
    message: str


@dataclass(frozen=True)
class StackState:
    sha: str
    compiler: str
    runtime: str
    kernels: str

    def as_dict(self) -> dict[str, str]:
        return {"compiler": self.compiler, "runtime": self.runtime, "kernels": self.kernels}


class History:
    def __init__(self, commits: list[Commit], base: dict[str, str], releases: dict[str, str] | None = None) -> None:
        self.commits = commits
        self.base = base
        self.releases = releases or {}

    @classmethod
    def load(cls, path: Path = HISTORY_FILE) -> History:
        with path.open("r", encoding=ENCODING) as fh:
            data = json.load(fh)
        return cls([Commit(**c) for c in data["commits"]], data["base"], data.get("releases", {}))

    def resolve(self, ref: str) -> str:
        """Release name or sha -> sha."""
        return self.releases.get(ref, ref)

    def index(self, sha: str) -> int:
        for i, c in enumerate(self.commits):
            if c.sha == sha:
                return i
        raise KeyError(f"unknown commit {sha!r}")

    def state(self, sha: str) -> StackState:
        versions = dict(self.base)
        for c in self.commits[: self.index(sha) + 1]:
            versions[c.component] = c.version
        return StackState(sha=sha, **versions)

    def between(self, good: str, bad: str) -> list[Commit]:
        """Commits in (good, bad], oldest first."""
        return self.commits[self.index(good) + 1: self.index(bad) + 1]
