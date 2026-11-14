"""Runtime + device model: executes a Program against slot-addressed SRAM.

SRAM is modelled literally: each slot is a flat buffer, a tensor is the first
numel elements of its slot. If the planner hands one slot to two live tensors,
the later write silently corrupts the earlier one, exactly like real memory.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from corestack import kernels
from corestack.compiler import Program
from corestack.versions import v


class DeviceTimeout(Exception):
    """Transient infrastructure failure (lost heartbeat). Retryable; not a product defect."""


@dataclass
class Device:
    dtype: str = "fp32"
    sram_bytes: int = 4 << 20
    fault_rate: float = 0.0
    flops_per_cycle: int = 256
    bytes_per_cycle: int = 64


@dataclass
class TraceEntry:
    call: int
    kernel: str
    tensor: str
    role: str            # "in" or "out"
    slot: int | None
    prefetched: bool
    value: np.ndarray


@dataclass
class Execution:
    output: np.ndarray
    cycles: int
    trace: list[TraceEntry] = field(default_factory=list)


def execute(prog: Program, feeds: dict[str, np.ndarray], params: dict[str, np.ndarray], device: Device,
            flags: set[str], runtime_version: str, kernels_version: str,
            rng: np.random.Generator | None = None, trace: bool = False) -> Execution:
    dt = kernels.DTYPES[device.dtype]
    cap = max(int(np.prod(s)) for t, s in prog.shapes.items() if t not in params)
    sram: dict[int, np.ndarray] = {}
    dparams = {k: p.astype(dt) for k, p in params.items()}
    log: list[TraceEntry] = []

    def write(t: str, arr: np.ndarray) -> None:
        buf = sram.setdefault(prog.slot[t], np.zeros(cap, dtype=dt))
        buf[: arr.size] = arr.ravel()

    def read(t: str) -> np.ndarray:
        if t in dparams:
            return dparams[t]
        n = int(np.prod(prog.shapes[t]))
        return sram.setdefault(prog.slot[t], np.zeros(cap, dtype=dt))[:n].reshape(prog.shapes[t]).copy()

    for t, arr in feeds.items():
        write(t, arr.astype(dt))

    prefetch = "async_prefetch" in flags and v(runtime_version) >= v("2.4")
    full_hazard_check = v(runtime_version) >= v("2.5")
    staged: dict[tuple[int, int], np.ndarray] = {}
    cycles = 0

    for i, call in enumerate(prog.calls):
        if rng is not None and device.fault_rate and rng.random() < device.fault_rate:
            raise DeviceTimeout(f"device heartbeat lost after {int(rng.integers(800, 5000))}ms in {call.kernel} (call {i})")
        args = []
        for j, t in enumerate(call.inputs):
            was_staged = (i, j) in staged
            val = staged.pop((i, j)) if was_staged else read(t)
            args.append(val)
            if trace and t not in dparams:
                log.append(TraceEntry(i, call.kernel, t, "in", prog.slot.get(t), was_staged, val))

        if prefetch and i + 1 < len(prog.calls):
            # Overlap: DMA the next call's activations into staging while this call computes.
            # Must not stage a tensor this call is about to write (read-after-write hazard).
            nxt = prog.calls[i + 1]
            for j, t in enumerate(nxt.inputs):
                if t in dparams:
                    continue
                # B3: runtime 2.4.x only checks the first operand (written when every kernel was unary).
                hazard = t == call.output if full_hazard_check else (j == 0 and t == call.output)
                if not hazard:
                    staged[(i + 1, j)] = read(t)

        out = kernels.run(call.kernel, args, call.attrs, kernels_version).astype(dt)
        write(call.output, out)
        if trace:
            log.append(TraceEntry(i, call.kernel, call.output, "out", prog.slot[call.output], False, out))

        compute = kernels.flops(call.kernel, [a.shape for a in args]) // device.flops_per_cycle
        dma = sum(a.nbytes for a in args) // device.bytes_per_cycle
        cycles += max(compute, dma) if prefetch else compute + dma

    return Execution(read(prog.calls[-1].output) if prog.calls else feeds["x"], cycles, log)
