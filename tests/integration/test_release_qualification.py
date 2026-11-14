"""End-to-end: the harness must catch, bucket, route, bisect, and gate all three seeded defects.

If one of these fails, the harness lost the ability to see a class of bug. That is
a harness regression, which is worse than a product regression, because it is silent.
"""
from __future__ import annotations

import pytest

from corestack.versions import History
from rit import cli, matrix, runner
from rit.bisect import BisectError, bisect
from rit.qualify import qualify
from rit.triage import triage

H = History.load()


@pytest.fixture(scope="module")
def baseline():
    return runner.run_cases(matrix.plan("release"), H, H.resolve("2026.10"))


@pytest.fixture(scope="module")
def rc1(baseline):
    cur = runner.run_cases(matrix.plan("release"), H, H.resolve("2026.11-rc1"))
    return qualify("2026.11-rc1", cur, "2026.10", baseline, H, planned=matrix.plan("release"))


def test_baseline_release_is_clean(baseline):
    assert {r.status for r in baseline} == {"pass"}


def test_rc1_is_no_go(rc1):
    assert rc1.decision == "NO-GO"
    failed = {g.name.split(":")[0] for g in rc1.gates if not g.passed}
    assert failed == {"correctness", "regression"}


def test_rc1_failures_collapse_to_three_routed_buckets(rc1):
    by_layer = {b.localization.layer: b for b in rc1.buckets}
    assert set(by_layer) == {"runtime", "kernels", "compiler"}
    assert by_layer["runtime"].owner == "inference-core/runtime"
    assert sum(len(b.results) for b in rc1.buckets) == 80


@pytest.mark.parametrize("layer,diagnosis,where,trigger", [
    ("kernels", "non-finite output", ("fused_attention", "ctx"), ["fused_attention", "dtype=fp16"]),
    ("runtime", "stale prefetched read (RAW hazard)", ("matmul_t", "k"), ["async_prefetch"]),
    ("compiler", "input clobbered in SRAM after write (buffer aliasing)", ("add", "x"), ["shape>=large"]),
])
def test_localization_and_minimal_trigger(rc1, layer, diagnosis, where, trigger):
    b = next(b for b in rc1.buckets if b.localization.layer == layer)
    assert b.localization.diagnosis == diagnosis
    assert (b.localization.kernel, b.localization.tensor_role) == where
    assert b.trigger == trigger


@pytest.mark.parametrize("case_id,first_bad,verdict", [
    ("attention_hot/small/fp16/fused_attention", "b27d9a4", "root cause"),
    ("decoder/large/fp32/none", "c3e81f0", "root cause"),
    # The headline case: bisect blames the compiler scheduler, but the defect is in the runtime.
    ("attention/small/fp32/async_prefetch", "0a7e4c9", "exposing change; latent defect in runtime"),
])
def test_bisect_distinguishes_root_cause_from_exposing_change(case_id, first_bad, verdict):
    r = bisect(case_id, "a1f0c3e", "0a7e4c9", H)
    assert r.first_bad.sha == first_bad
    assert r.verdict == verdict


def test_bisect_refuses_bad_endpoints():
    with pytest.raises(BisectError):
        bisect("attention/small/fp32/async_prefetch", "0a7e4c9", "3dab7f1", H)   # 'good' fails


def test_rc2_is_go(baseline):
    cur = runner.run_cases(matrix.plan("release"), H, H.resolve("2026.11-rc2"), run_seed=3, fault_rate=0.001)
    q = qualify("2026.11-rc2", cur, "2026.10", baseline, H, planned=matrix.plan("release"))
    assert q.decision == "GO", [g for g in q.gates if not g.passed]
    assert q.perf["geomean"] < 1.0, "async prefetch + fusion should be a net cycle win"


def test_noisy_infra_is_inconclusive_not_go(baseline):
    cur = runner.run_cases(matrix.plan("release"), H, H.resolve("2026.11-rc2"), run_seed=7, fault_rate=0.01)
    q = qualify("2026.11-rc2", cur, "2026.10", baseline, H, planned=matrix.plan("release"), do_bisect=False)
    assert q.decision == "INCONCLUSIVE"


def test_missing_p0_coverage_is_inconclusive(baseline):
    plan = matrix.plan("release")
    partial = runner.run_cases(matrix.shard(plan, 1, 3), H, H.resolve("2026.11-rc2"))
    q = qualify("2026.11-rc2", partial, "2026.10", baseline, H, planned=plan, do_bisect=False)
    assert q.decision == "INCONCLUSIVE"


def test_infra_failures_route_to_infra_and_match_waived_issue():
    res = [runner.run_case(matrix.Case("mlp", "small", "fp32"), H, "a1f0c3e", run_seed=s, fault_rate=0.9)
           for s in range(3)]
    buckets = triage([r for r in res if r.status == "infra"], H)
    assert buckets and buckets[0].localization.layer == "infra" and buckets[0].known_issue == "INFRA-311"


@pytest.mark.parametrize("mode,commit,layers", [
    # Documents the cost/coverage tradeoff in the README; if the matrix changes, update both.
    ("presubmit", "0a7e4c9", {"runtime", "kernels"}),
    ("nightly", "c3e81f0", {"kernels", "compiler"}),
    ("release", "0a7e4c9", {"runtime", "kernels", "compiler"}),
])
def test_detection_by_mode(mode, commit, layers):
    res = runner.run_cases(matrix.plan(mode), H, commit)
    assert {b.localization.layer for b in triage(res, H)} == layers


def test_cli_end_to_end(tmp_path):
    out = tmp_path / "rc1.jsonl"
    assert cli.main(["run", "--commit", "2026.11-rc1", "--mode", "presubmit", "--out", str(out)]) == 1
    assert cli.main(["qualify", "--release", "2026.11-rc1", "--baseline-release", "2026.10",
                     "--results", str(out), "--mode", "presubmit", "--no-bisect",
                     "--out", str(tmp_path / "r.md")]) == 1
    assert (tmp_path / "r.json").exists()
    assert cli.main(["bisect", "--case", "attention/small/fp32/async_prefetch",
                     "--good", "2026.10", "--bad", "2026.11-rc1"]) == 0
