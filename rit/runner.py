"""Execute cases against a stack commit; compare to the float64 golden; retry infra only."""
from __future__ import annotations

import json
import time
import traceback
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from corestack.compiler import CompileError, compile_graph
from corestack.graph import build, run_reference
from corestack.runtime import Device, DeviceTimeout, Execution, execute
from corestack.versions import History
from rit.matrix import Case

ENCODING = "utf-8"
MAX_ATTEMPTS = 3
TOLERANCE = {"fp32": 1e-4, "fp16": 3e-2}   # max |err| / max |ref|
FEED_SEED = 1


@dataclass
class Result:
    case_id: str
    sha: str
    status: str                   # pass | fail | error | infra
    attempts: int
    tier: str
    versions: dict[str, str]
    error_type: str | None = None
    message: str = ""
    max_rel_err: float | None = None
    cycles: int | None = None
    duration_ms: float = 0.0
    infra_errors: list[str] = field(default_factory=list)

    @property
    def flaky(self) -> bool:
        return self.status == "pass" and self.attempts > 1


def rel_err(out: np.ndarray, ref: np.ndarray) -> float:
    out = out.astype(np.float64)
    if not np.isfinite(out).all():
        return float("inf")
    return float(np.abs(out - ref).max() / max(np.abs(ref).max(), 1e-12))


def execute_case(case: Case, history: History, sha: str, *, fault_rate: float = 0.0,
                 rng: np.random.Generator | None = None, trace: bool = False) -> tuple[Execution, dict]:
    """One attempt. Raises CompileError / DeviceTimeout / anything the stack raises."""
    st = history.state(sha)
    graph = build(case.model, case.shape)
    feeds = graph.feeds(FEED_SEED)
    device = Device(dtype=case.dtype, fault_rate=fault_rate)
    flags = set(case.flags)
    prog = compile_graph(graph, flags, st.compiler, device.sram_bytes, case.dtype)
    run = execute(prog, feeds, graph.params, device, flags, st.runtime, st.kernels, rng=rng, trace=trace)
    return run, run_reference(graph, feeds)


def run_case(case: Case, history: History, sha: str, *, run_seed: int = 0, fault_rate: float = 0.0) -> Result:
    res = Result(case.case_id, sha, "error", 0, case.tier, history.state(sha).as_dict())
    t0 = time.perf_counter()
    for attempt in range(1, MAX_ATTEMPTS + 1):
        res.attempts = attempt
        rng = np.random.default_rng([run_seed, zlib.crc32(case.case_id.encode()), attempt])
        try:
            run, ref_env = execute_case(case, history, sha, fault_rate=fault_rate, rng=rng)
        except DeviceTimeout as exc:
            # Infra, not product: retry. Never retry a product failure; that hides real bugs.
            res.infra_errors.append(str(exc))
            res.status, res.error_type, res.message = "infra", "DeviceTimeout", str(exc)
            continue
        except CompileError as exc:
            res.status, res.error_type, res.message = "error", "CompileError", str(exc)
            break
        except Exception as exc:  # noqa: BLE001  any other crash is a product error with a traceback
            res.status, res.error_type = "error", type(exc).__name__
            res.message = f"{exc}\n{traceback.format_exc(limit=3)}"
            break
        ref = ref_env[build(case.model, case.shape).output]
        err = rel_err(run.output, ref)
        res.max_rel_err = err
        res.cycles = run.cycles
        tol = TOLERANCE[case.dtype]
        if err == float("inf"):
            bad = int((~np.isfinite(run.output)).sum())
            res.status, res.error_type = "fail", "NonFinite"
            res.message = f"non-finite output: {bad}/{run.output.size} elements"
        elif err > tol:
            res.status, res.error_type = "fail", "NumericMismatch"
            res.message = f"max relative error {err:.2e} > tolerance {tol:.0e}"
        else:
            res.status, res.error_type, res.message = "pass", None, ""
        break
    res.duration_ms = round((time.perf_counter() - t0) * 1000, 2)
    return res


def run_cases(cases: list[Case], history: History, sha: str, **kw) -> list[Result]:
    return [run_case(c, history, sha, **kw) for c in cases]


def save(results: list[Result], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding=ENCODING) as fh:
        for r in results:
            fh.write(json.dumps(asdict(r)) + "\n")


def load(*paths: Path) -> list[Result]:
    out: list[Result] = []
    for p in paths:
        with p.open("r", encoding=ENCODING) as fh:
            out.extend(Result(**json.loads(line)) for line in fh if line.strip())
    return out
