"""Compiler: graph -> kernel program. Passes: fusion, scheduling, memory planning."""
from __future__ import annotations

from dataclasses import dataclass, field

from corestack.graph import Graph, Op, infer_shapes
from corestack.versions import v

DTYPE_BYTES = {"fp32": 4, "fp16": 2}


class CompileError(Exception):
    """Raised by the compiler; triage attributes it to the compiler layer."""


@dataclass
class KernelCall:
    kernel: str
    inputs: list[str]
    output: str
    attrs: dict = field(default_factory=dict)


@dataclass
class Program:
    calls: list[KernelCall]
    slot: dict[str, int]                 # activation tensor -> SRAM buffer slot
    shapes: dict[str, tuple[int, ...]]
    n_slots: int
    max_slots: int
    reused_slots: int                    # how many allocations reused a freed slot


# ---- fusion ---------------------------------------------------------------
def _single_consumer(graph: Graph, tensor: str, kind: str) -> Op | None:
    cons = graph.consumers(tensor)
    return cons[0] if len(cons) == 1 and cons[0].kind == kind else None


def _fuse(graph: Graph, flags: set[str]) -> list[KernelCall]:
    calls: list[KernelCall] = []
    skip: set[str] = set()
    for op in graph.ops:
        if op.name in skip:
            continue
        if "fuse_mlp" in flags and op.kind == "matmul" and op.inputs[1] in graph.params:
            add = _single_consumer(graph, op.output, "add")
            gelu = add and add.inputs[1] in graph.params and _single_consumer(graph, add.output, "gelu")
            if gelu:
                calls.append(KernelCall("matmul_bias_gelu", [op.inputs[0], op.inputs[1], add.inputs[1]], gelu.output))
                skip |= {add.name, gelu.name}
                continue
        if "fused_attention" in flags and op.kind == "scale":
            sm = _single_consumer(graph, op.output, "softmax")
            mm = sm and _single_consumer(graph, sm.output, "matmul")
            if mm and mm.inputs[0] == sm.output:
                calls.append(KernelCall("fused_attention", [op.inputs[0], mm.inputs[1]], mm.output,
                                        {"factor": op.attrs["factor"]}))
                skip |= {sm.name, mm.name}
                continue
        calls.append(KernelCall(op.kind, list(op.inputs), op.output, dict(op.attrs)))
    return calls


# ---- scheduling -----------------------------------------------------------
def _schedule(calls: list[KernelCall], params: dict, version: str) -> list[KernelCall]:
    """compiler >= 3.3: order runs of independent weight matmuls by weight name, descending,
    so weights stream in address order. Semantically legal; it changes which call precedes which."""
    if v(version) < v("3.3"):
        return calls
    out: list[KernelCall] = []
    i = 0
    while i < len(calls):
        j = i
        run: list[KernelCall] = []
        while (j < len(calls) and calls[j].kernel == "matmul" and calls[j].inputs[1] in params
               and all(calls[j].inputs[0] != r.output for r in run)):
            run.append(calls[j])
            j += 1
        if len(run) > 1:
            out.extend(sorted(run, key=lambda c: c.inputs[1], reverse=True))
            i = j
        else:
            out.append(calls[i])
            i += 1
    return out


# ---- memory planning ------------------------------------------------------
def _plan(calls: list[KernelCall], graph: Graph, shapes: dict, sram_bytes: int,
          dtype: str, version: str) -> tuple[dict[str, int], int, int, int]:
    acts = [t for t in shapes if t not in graph.params]
    max_bytes = max(int(_numel(shapes[t])) for t in acts) * DTYPE_BYTES[dtype]
    max_slots = sram_bytes // max_bytes
    if max_slots < 3:
        raise CompileError(f"SRAM {sram_bytes}B holds {max_slots} buffers of {max_bytes}B; need >= 3")

    first_use: dict[str, int] = {}
    last_use: dict[str, int] = {}
    for i, c in enumerate(calls):
        for t in c.inputs:
            first_use.setdefault(t, i)
            last_use[t] = i
    last_use[graph.output] = len(calls)
    # B2: compiler 3.2-3.3 'single-pass liveness' records lifetime end as FIRST use.
    buggy = v("3.2") <= v(version) < v("3.4")
    end = {t: (first_use.get(t, last_use.get(t, -1)) if buggy and t != graph.output else last_use.get(t, -1))
           for t in acts}

    slot: dict[str, int] = {}
    live: dict[int, str] = {}
    free: list[int] = []
    n_slots = 0
    reused = 0

    def alloc(t: str) -> None:
        nonlocal n_slots, reused
        # Prefer a fresh slot (keeps buffers independent for prefetch overlap); reuse only under pressure.
        if n_slots < max_slots:
            s = n_slots
            n_slots += 1
        elif free:
            s = free.pop(0)
            reused += 1
        else:
            raise CompileError(f"out of SRAM: {len(live)} live buffers, max {max_slots}")
        slot[t] = s
        live[s] = t

    for t in graph.inputs:
        alloc(t)
    for i, c in enumerate(calls):
        for s, t in list(live.items()):
            if end[t] < i:
                del live[s]
                free.append(s)
        alloc(c.output)
    return slot, n_slots, max_slots, reused


def _numel(shape: tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def compile_graph(graph: Graph, flags: set[str], version: str, sram_bytes: int, dtype: str) -> Program:
    shapes = infer_shapes(graph)
    calls = _fuse(graph, flags)
    calls = _schedule(calls, graph.params, version)
    slot, n_slots, max_slots, reused = _plan(calls, graph, shapes, sram_bytes, dtype, version)
    return Program(calls, slot, shapes, n_slots, max_slots, reused)
