"""rit — Release Integration Testing CLI.

  python -m rit plan     --mode nightly
  python -m rit run      --commit 2026.11-rc1 --mode release --shard 1/3 --out results/rc1-1.jsonl
  python -m rit run      --commit 0a7e4c9 --case attention/small/fp32/async_prefetch
  python -m rit triage   results/rc1-*.jsonl
  python -m rit bisect   --case attention/small/fp32/async_prefetch --good 2026.10 --bad 2026.11-rc1
  python -m rit qualify  --release 2026.11-rc1 --baseline-release 2026.10 --results results/rc1-*.jsonl \\
                         --mode release --out reports/rc1.md

qualify exit codes: 0 GO, 1 NO-GO, 2 INCONCLUSIVE.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from corestack.versions import History
from rit import matrix, runner
from rit.bisect import BisectError, bisect
from rit.qualify import qualify
from rit.report import render
from rit.triage import triage

ENCODING = "utf-8"
EXIT = {"GO": 0, "NO-GO": 1, "INCONCLUSIVE": 2}


def _cases(args: argparse.Namespace) -> list[matrix.Case]:
    if getattr(args, "case", None):
        return [matrix.Case.parse(c) for c in args.case]
    cases = matrix.plan(args.mode)
    if getattr(args, "shard", None):
        i, n = (int(x) for x in args.shard.split("/"))
        cases = matrix.shard(cases, i, n)
    return cases


def cmd_plan(args: argparse.Namespace, history: History) -> int:
    cases = _cases(args)
    print(f"{args.mode}: {len(cases)} cases  tiers {dict(sorted(Counter(c.tier for c in cases).items()))}")
    if args.list:
        for c in cases:
            print(f"  {c.tier}  {c.case_id}")
    return 0


def cmd_run(args: argparse.Namespace, history: History) -> int:
    sha = history.resolve(args.commit)
    results = runner.run_cases(_cases(args), history, sha, run_seed=args.run_seed, fault_rate=args.fault_rate)
    if args.out:
        runner.save(results, Path(args.out))
    counts = Counter(r.status for r in results)
    print(f"{args.commit} ({sha}) {history.state(sha).as_dict()}: {dict(counts)}"
          f"  flaky={sum(r.flaky for r in results)}")
    for r in results:
        if r.status != "pass":
            print(f"  {r.status:5}  {r.case_id:45}  {r.message.splitlines()[0] if r.message else ''}")
    return 0 if counts.keys() <= {"pass"} else 1


def cmd_triage(args: argparse.Namespace, history: History) -> int:
    buckets = triage(runner.load(*map(Path, args.results)), history)
    for i, b in enumerate(buckets, 1):
        loc = b.localization
        print(f"[{i}] {len(b.results):3} case(s)  {loc.layer:8} {loc.diagnosis}"
              f"{f'  @ {loc.kernel} {loc.access} {loc.tensor_role}' if loc.kernel else ''}")
        print(f"      trigger: {', '.join(b.trigger) or '-'}   owner: {b.owner}   known: {b.known_issue or 'new'}")
        print(f"      repro:   {b.representative}")
    return 0


def cmd_bisect(args: argparse.Namespace, history: History) -> int:
    try:
        r = bisect(args.case[0], history.resolve(args.good), history.resolve(args.bad), history)
    except BisectError as exc:
        print(f"bisect aborted: {exc}", file=sys.stderr)
        return 2
    for sha, st in r.steps:
        print(f"  {sha}  {st}")
    c = r.first_bad
    print(f"first bad: {c.sha} [{c.component} {c.version}] {c.message}")
    if r.localization:
        print(f"localized: {r.localization.layer}: {r.localization.diagnosis} @ {r.localization.kernel}")
    print(f"verdict:   {r.verdict}")
    return 0


def cmd_qualify(args: argparse.Namespace, history: History) -> int:
    planned = matrix.plan(args.mode)
    current = runner.load(*map(Path, args.results))
    if args.baseline_results:
        baseline = runner.load(*map(Path, args.baseline_results))
    else:
        baseline = runner.run_cases(planned, history, history.resolve(args.baseline_release))
    q = qualify(args.release, current, args.baseline_release, baseline, history,
                planned=planned, do_bisect=not args.no_bisect)
    text = render(q)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding=ENCODING)
        summary = {"release": q.release, "sha": q.sha, "decision": q.decision, "counts": q.counts,
                   "gates": [asdict(g) for g in q.gates],
                   "buckets": [{"key": b.key, "cases": len(b.results), "owner": b.owner,
                                "trigger": b.trigger, "known_issue": b.known_issue} for b in q.buckets],
                   "bisect": [{"case": r.case_id, "first_bad": r.first_bad.sha, "verdict": r.verdict}
                              for r in q.bisects]}
        out.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding=ENCODING)
    return EXIT[q.decision]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rit", description="Release Integration Testing for the inference core")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp: argparse.ArgumentParser, mode: bool = True) -> None:
        if mode:
            sp.add_argument("--mode", choices=sorted(matrix.MODES), default="presubmit")
        sp.add_argument("--case", action="append", help="case id (repeatable); overrides --mode")

    sp = sub.add_parser("plan")
    common(sp)
    sp.add_argument("--list", action="store_true")
    sp.set_defaults(fn=cmd_plan)

    sp = sub.add_parser("run")
    common(sp)
    sp.add_argument("--commit", required=True, help="sha or release name")
    sp.add_argument("--shard", help="i/n, 1-based")
    sp.add_argument("--fault-rate", type=float, default=0.0, help="injected device-timeout rate per kernel call")
    sp.add_argument("--run-seed", type=int, default=0)
    sp.add_argument("--out")
    sp.set_defaults(fn=cmd_run)

    sp = sub.add_parser("triage")
    sp.add_argument("results", nargs="+")
    sp.set_defaults(fn=cmd_triage)

    sp = sub.add_parser("bisect")
    common(sp, mode=False)
    sp.add_argument("--good", required=True)
    sp.add_argument("--bad", required=True)
    sp.set_defaults(fn=cmd_bisect)

    sp = sub.add_parser("qualify")
    sp.add_argument("--release", required=True)
    sp.add_argument("--baseline-release", required=True)
    sp.add_argument("--results", nargs="+", required=True)
    sp.add_argument("--baseline-results", nargs="+")
    sp.add_argument("--mode", choices=sorted(matrix.MODES), default="release")
    sp.add_argument("--no-bisect", action="store_true")
    sp.add_argument("--out")
    sp.set_defaults(fn=cmd_qualify)

    args = p.parse_args(argv)
    if args.cmd == "bisect" and not args.case:
        p.error("bisect needs --case")
    return args.fn(args, History.load())


if __name__ == "__main__":
    sys.exit(main())
