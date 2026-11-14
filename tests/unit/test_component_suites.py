"""What each feature team's own suite looks like, run at the rc1 component versions.

Every test here PASSES at the buggy versions. That is the point: feature-level
qualification is green, and the release is still broken. These are the gaps
RIT exists for (see tests/integration/test_release_qualification.py).
"""
from __future__ import annotations

import numpy as np

from corestack import kernels
from corestack.compiler import compile_graph
from corestack.graph import build, run_reference
from corestack.runtime import Device, execute
from corestack.versions import History

RC1 = History.load().state("0a7e4c9")


def test_kernels_team_fused_attention_fp16_matches_unfused():
    """kernels 2.1 suite: random logits in a realistic range. B1 needs |logit| > ~11 to show."""
    rng = np.random.default_rng(0)
    scores = rng.standard_normal((2, 16, 16)).astype(np.float16)
    values = rng.standard_normal((2, 16, 8)).astype(np.float16)
    fused = kernels.fused_attention(scores, values, 0.5, RC1.kernels).astype(np.float64)
    s = scores.astype(np.float64) * 0.5
    p = np.exp(s - s.max(-1, keepdims=True))
    want = (p / p.sum(-1, keepdims=True)) @ values.astype(np.float64)
    assert np.abs(fused - want).max() < 2e-2


def test_compiler_team_fusion_rewrites_patterns():
    """compiler suite: structural checks on the emitted program."""
    g = build("attention", "small")
    prog = compile_graph(g, {"fused_attention"}, RC1.compiler, 4 << 20, "fp32")
    kinds = [c.kernel for c in prog.calls]
    assert "fused_attention" in kinds and "softmax" not in kinds


def test_compiler_team_planner_fits_sram_and_live_tensors_distinct_at_small_shapes():
    """compiler suite runs small shapes; B2 only bites when SRAM pressure forces reuse."""
    g = build("decoder", "small")
    prog = compile_graph(g, set(), RC1.compiler, 4 << 20, "fp32")
    assert prog.reused_slots == 0
    assert prog.n_slots <= prog.max_slots


def test_runtime_team_prefetch_matches_sync_on_mlp():
    """runtime 2.4 suite: prefetch vs synchronous on the MLP block (no reordered binary ops)."""
    g = build("mlp", "medium")
    feeds = g.feeds(1)
    prog = compile_graph(g, set(), RC1.compiler, 4 << 20, "fp32")
    sync = execute(prog, feeds, g.params, Device(), set(), RC1.runtime, RC1.kernels).output
    pre = execute(prog, feeds, g.params, Device(), {"async_prefetch"}, RC1.runtime, RC1.kernels).output
    np.testing.assert_array_equal(sync, pre)
    assert np.abs(pre - run_reference(g, feeds)["y"]).max() < 1e-4
