from __future__ import annotations

import itertools

import pytest

from corestack.versions import History, v
from rit import matrix
from rit.matrix import Case, covering, full_matrix, shard, tuples
from rit.qualify import compare, perf_delta
from rit.runner import Result, run_case

H = History.load()


def _r(case_id: str, status: str, attempts: int = 1, cycles: int | None = 100) -> Result:
    return Result(case_id, "x", status, attempts, "P2", {}, cycles=cycles)


class TestVersions:
    def test_numeric_not_lexical(self):
        assert v("3.10") > v("3.9") and v("2.4.1") > v("2.4")

    def test_state_is_cumulative(self):
        st = H.state("e5c2d18")
        assert (st.compiler, st.runtime, st.kernels) == ("3.2", "2.4.1", "2.1")

    def test_release_names_resolve(self):
        assert H.resolve("2026.11-rc1") == "0a7e4c9" and H.resolve("abc") == "abc"

    def test_between_is_half_open(self):
        shas = [c.sha for c in H.between("a1f0c3e", "c3e81f0")]
        assert shas == ["b27d9a4", "c3e81f0"]


class TestMatrix:
    def test_full_matrix_unique(self):
        cases = full_matrix()
        assert len(cases) == len(set(cases)) == 4 * 3 * 2 * 8

    @pytest.mark.parametrize("t", [2, 3])
    def test_covering_array_covers_every_t_tuple(self, t):
        want = set().union(*(tuples(c, t) for c in full_matrix()))
        got = set().union(*(tuples(c, t) for c in covering(t)))
        assert got == want

    def test_covering_is_much_smaller(self):
        assert len(covering(2)) < 20 and len(covering(3)) < 40 < len(full_matrix())

    def test_shards_partition_the_plan(self):
        cases = full_matrix()
        parts = [shard(cases, i, 3) for i in (1, 2, 3)]
        assert sorted(itertools.chain(*parts)) == sorted(cases)
        assert all(parts)

    def test_shard_assignment_stable_when_cases_added(self):
        cases = full_matrix()
        before = set(shard(cases, 1, 3))
        assert before <= set(shard(cases + [Case("mlp", "small", "fp32", ("new_flag",))], 1, 3))

    def test_case_id_roundtrip(self):
        for c in full_matrix():
            assert Case.parse(c.case_id) == c

    def test_risk_tiers(self):
        assert Case("decoder", "large", "fp16", ("fused_attention", "async_prefetch")).tier == "P0"
        assert Case("mlp", "small", "fp32").tier == "P2"

    def test_unknown_mode_rejected(self):
        with pytest.raises(ValueError):
            matrix.plan("weekly")


class TestRunner:
    def test_infra_is_retried_and_can_recover(self):
        case = Case("mlp", "small", "fp32")
        results = [run_case(case, H, "a1f0c3e", run_seed=s, fault_rate=0.15) for s in range(30)]
        assert any(r.flaky for r in results), "some cases should pass on retry"
        assert all(r.status in ("pass", "infra") for r in results)
        assert all(r.attempts <= 3 for r in results)

    def test_product_failure_is_not_retried(self):
        r = run_case(Case("attention_hot", "small", "fp16", ("fused_attention",)), H, "b27d9a4")
        assert r.status == "fail" and r.error_type == "NonFinite" and r.attempts == 1

    def test_sram_overflow_is_a_compile_error(self):
        from corestack.compiler import CompileError, compile_graph
        from corestack.graph import build
        with pytest.raises(CompileError):
            compile_graph(build("decoder", "large"), set(), "3.1", 64 << 10, "fp32")


class TestCompare:
    def test_classification(self):
        base = [_r("a", "pass"), _r("b", "fail"), _r("c", "fail"), _r("d", "pass")]
        cur = [_r("a", "fail"), _r("b", "fail"), _r("c", "pass"), _r("d", "pass", attempts=2), _r("e", "fail")]
        out = compare(base, cur)
        assert out["new_failure"] == ["a", "e"]
        assert out["still_failing"] == ["b"]
        assert out["fixed"] == ["c"]
        assert out["flaky"] == ["d"]
        assert out["new_case"] == ["e"]

    def test_infra_is_never_a_regression(self):
        out = compare([_r("a", "pass")], [_r("a", "infra")])
        assert out["infra"] == ["a"] and out["new_failure"] == []

    def test_perf_delta(self):
        base = [_r("a", "pass", cycles=100), _r("b", "pass", cycles=100)]
        cur = [_r("a", "pass", cycles=110), _r("b", "pass", cycles=90)]
        d = perf_delta(base, cur)
        assert d["regressed"] == 1 and d["worst"] == pytest.approx(1.1)
        assert d["geomean"] == pytest.approx((1.1 * 0.9) ** 0.5)
