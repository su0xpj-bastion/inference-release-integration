"""Test-case matrix: full release matrix, pairwise presubmit, risk tiers, sharding.

Feature teams own single-feature qualification. RIT's matrix exists for the
*interactions*: a case is high-risk when its enabled features cross layer
boundaries, because no single team's suite exercises that combination.
"""
from __future__ import annotations

import itertools
import zlib
from dataclasses import dataclass

from corestack.graph import MODELS, SHAPES

FLAGS = ("fused_attention", "fuse_mlp", "async_prefetch")
DTYPES = ("fp32", "fp16")
FEATURE_LAYERS = {
    "fused_attention": {"compiler", "kernels"},
    "fuse_mlp": {"compiler", "kernels"},
    "async_prefetch": {"runtime"},
}


@dataclass(frozen=True, order=True)
class Case:
    model: str
    shape: str
    dtype: str
    flags: tuple[str, ...] = ()

    @property
    def case_id(self) -> str:
        return f"{self.model}/{self.shape}/{self.dtype}/{'+'.join(self.flags) or 'none'}"

    @classmethod
    def parse(cls, case_id: str) -> Case:
        model, shape, dtype, flags = case_id.split("/")
        return cls(model, shape, dtype, () if flags == "none" else tuple(flags.split("+")))

    @property
    def layers(self) -> set[str]:
        return set().union(*(FEATURE_LAYERS[f] for f in self.flags)) if self.flags else set()

    @property
    def risk(self) -> int:
        """Layers crossed by enabled features, plus fp16 (numerics) and large (memory pressure)."""
        return len(self.layers) + (self.dtype == "fp16") + (self.shape == "large")

    @property
    def tier(self) -> str:
        return "P0" if self.risk >= 4 else "P1" if self.risk == 3 else "P2"


def _case(model: str, shape: str, dtype: str, bits: tuple[int, ...]) -> Case:
    return Case(model, shape, dtype, tuple(f for f, b in zip(FLAGS, bits) if b))


def full_matrix() -> list[Case]:
    return [_case(m, s, d, bits) for m, s, d, bits in
            itertools.product(MODELS, SHAPES, DTYPES, itertools.product((0, 1), repeat=len(FLAGS)))]


def _params(c: Case) -> tuple:
    return (c.model, c.shape, c.dtype, *(int(f in c.flags) for f in FLAGS))


def tuples(c: Case, t: int) -> set[tuple]:
    """Every t-way (parameter, value) combination this case exercises."""
    p = _params(c)
    return {tuple((i, p[i]) for i in idx) for idx in itertools.combinations(range(len(p)), t)}


def covering(t: int) -> list[Case]:
    """Greedy t-wise covering array: every value combination of any t parameters appears at least once.

    t=2 (pairwise) is cheap but provably blind to 3-way interactions such as
    model x dtype x feature; that is why nightly runs t=3 and release runs everything.
    """
    candidates = full_matrix()
    uncovered = set().union(*(tuples(c, t) for c in candidates))
    chosen: list[Case] = []
    while uncovered:
        best = max(candidates, key=lambda c: (len(tuples(c, t) & uncovered), c.risk, c.case_id))
        chosen.append(best)
        uncovered -= tuples(best, t)
    return sorted(chosen)


MODES = {"presubmit": 2, "nightly": 3, "release": None}


def plan(mode: str) -> list[Case]:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {sorted(MODES)}, got {mode!r}")
    t = MODES[mode]
    return full_matrix() if t is None else covering(t)


def shard(cases: list[Case], index: int, total: int) -> list[Case]:
    """Stable sharding by case-id hash (1-based index), so shards don't reshuffle when cases are added."""
    if not 1 <= index <= total:
        raise ValueError(f"shard {index}/{total}")
    return [c for c in cases if zlib.crc32(c.case_id.encode()) % total == index - 1]
