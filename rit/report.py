"""Render a Qualification as a markdown report (also used as the CI job summary)."""
from __future__ import annotations

from rit.qualify import Qualification

ICON = {"GO": "🟢", "NO-GO": "🔴", "INCONCLUSIVE": "🟡"}


def _fmt_counts(d: dict[str, int]) -> str:
    return ", ".join(f"{k} {v}" for k, v in sorted(d.items(), key=lambda kv: -kv[1]))


def render(q: Qualification) -> str:
    total = sum(q.counts.values())
    lines = [
        f"# Release qualification: {q.release}",
        "",
        f"**Decision: {ICON[q.decision]} {q.decision}**  ·  commit `{q.sha}`  ·  baseline `{q.baseline}`  ·  "
        f"{total} cases: " + ", ".join(f"{k} {v}" for k, v in sorted(q.counts.items())),
        "",
        "## Gates",
        "",
        "| Gate | Result | Detail |",
        "|---|---|---|",
        *(f"| {g.name} | {'✅' if g.passed else '❌'} | {g.detail} |" for g in q.gates),
        "",
        "## Change vs baseline",
        "",
        "| New failures | Still failing | Fixed | Flaky (passed on retry) | Infra |",
        "|---|---|---|---|---|",
        f"| {len(q.comparison['new_failure'])} | {len(q.comparison['still_failing'])} | "
        f"{len(q.comparison['fixed'])} | {len(q.comparison['flaky'])} | {len(q.comparison['infra'])} |",
        "",
    ]
    red = q.buckets
    if red:
        lines += [
            "## Failure buckets (first-pass triage)",
            "",
            "| # | Cases | Layer | Diagnosis | Where | Minimal trigger | Owner | Known issue |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for i, b in enumerate(red, 1):
            loc = b.localization
            where = f"`{loc.kernel}` {loc.access} `{loc.tensor_role}`" if loc.kernel else ""
            lines.append(f"| {i} | {len(b.results)} | {loc.layer} | {loc.diagnosis} | {where} | "
                         f"{', '.join(b.trigger) or '-'} | {b.owner} | {b.known_issue or 'new'} |")
        lines.append("")
        for i, b in enumerate(red, 1):
            cond = b.conditions()
            lines += [
                f"<details><summary>Bucket {i}: {b.localization.key}</summary>",
                "",
                f"- Representative repro: `python -m rit run --commit {b.results[0].sha} --case {b.representative}`",
                f"- Models: {_fmt_counts(cond['model'])}",
                f"- Shapes: {_fmt_counts(cond['shape'])}",
                f"- Dtypes: {_fmt_counts(cond['dtype'])}",
                f"- Sample message: `{b.results[0].message.splitlines()[0] if b.results[0].message else ''}`",
                "",
                "</details>",
                "",
            ]
    if q.bisects:
        lines += [
            "## Bisect (baseline → candidate)",
            "",
            "| Bucket repro | First bad commit | Component | Verdict | Steps |",
            "|---|---|---|---|---|",
        ]
        for r in q.bisects:
            c = r.first_bad
            lines.append(f"| `{r.case_id}` | `{c.sha}` {c.message} | {c.component} {c.version} | "
                         f"**{r.verdict}** | {len(r.steps)} |")
        lines.append("")
    lines += [
        "## Performance (device cycles vs baseline, passing cases)",
        "",
        f"geomean x{q.perf['geomean']:.3f} · worst x{q.perf['worst']:.3f} · "
        f"{int(q.perf['regressed'])} case(s) over budget",
        "",
    ]
    return "\n".join(lines)
