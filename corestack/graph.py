"""Model graphs and the float64 golden reference.

The reference executes the *unfused, unscheduled* graph op by op in float64.
It shares no code with compiler/runtime/kernels on purpose: a golden that
reuses the code under test can only ever agree with it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SHAPES: dict[str, tuple[int, int, int]] = {   # (batch, seq, hidden)
    "small": (1, 16, 32),
    "medium": (2, 64, 64),
    "large": (4, 128, 128),
}
MODELS = ("mlp", "attention", "attention_hot", "decoder")


@dataclass
class Op:
    name: str
    kind: str                     # layernorm matmul matmul_t add gelu scale softmax
    inputs: tuple[str, ...]
    output: str
    attrs: dict = field(default_factory=dict)


@dataclass
class Graph:
    name: str
    shape: tuple[int, int, int]
    ops: list[Op]
    params: dict[str, np.ndarray]
    inputs: tuple[str, ...] = ("x",)
    output: str = "y"

    def feeds(self, seed: int) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        return {"x": rng.standard_normal(self.shape)}

    def consumers(self, tensor: str) -> list[Op]:
        return [op for op in self.ops if tensor in op.inputs]


class _Builder:
    def __init__(self, hidden: int, rng: np.random.Generator) -> None:
        self.h = hidden
        self.rng = rng
        self.ops: list[Op] = []
        self.params: dict[str, np.ndarray] = {}

    def param(self, name: str, shape: tuple[int, ...], std: float) -> str:
        self.params[name] = self.rng.standard_normal(shape) * std
        return name

    def op(self, kind: str, inputs: tuple[str, ...], output: str, **attrs) -> str:
        self.ops.append(Op(f"{kind}:{output}", kind, inputs, output, attrs))
        return output

    def mlp(self, x: str, p: str, out: str) -> str:
        h = self.h
        n = self.op("layernorm", (x,), f"{p}ln")
        a = self.op("matmul", (n, self.param(f"{p}W1", (h, 2 * h), h ** -0.5)), f"{p}up")
        a = self.op("add", (a, self.param(f"{p}b1", (2 * h,), 0.1)), f"{p}up_b")
        a = self.op("gelu", (a,), f"{p}act")
        a = self.op("matmul", (a, self.param(f"{p}W2", (2 * h, h), (2 * h) ** -0.5)), f"{p}down")
        a = self.op("add", (a, self.param(f"{p}b2", (h,), 0.1)), f"{p}down_b")
        return self.op("add", (a, x), out)                       # residual: x has 2 consumers

    def attention(self, x: str, p: str, out: str, hot: bool) -> str:
        h = self.h
        n = self.op("layernorm", (x,), f"{p}ln")
        q = self.op("matmul", (n, self.param(f"{p}Wq", (h, h), h ** -0.5)), f"{p}q")
        k = self.op("matmul", (n, self.param(f"{p}Wk", (h, h), h ** -0.5)), f"{p}k")
        v = self.op("matmul", (n, self.param(f"{p}Wv", (h, h), h ** -0.5)), f"{p}v")
        s = self.op("matmul_t", (q, k), f"{p}scores")
        # "hot" = a checkpoint exported without the 1/sqrt(d) temperature folded in.
        # Logits reach ~3*sqrt(h): harmless in fp32, overflows exp() in fp16.
        s = self.op("scale", (s,), f"{p}scaled", factor=1.0 if hot else h ** -0.5)
        s = self.op("softmax", (s,), f"{p}probs")
        a = self.op("matmul", (s, v), f"{p}ctx")
        a = self.op("matmul", (a, self.param(f"{p}Wo", (h, h), h ** -0.5)), f"{p}proj")
        return self.op("add", (a, x), out)


def build(model: str, shape: str, seed: int = 0) -> Graph:
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}")
    b, s, h = SHAPES[shape]
    g = _Builder(h, np.random.default_rng(seed))
    if model == "mlp":
        g.mlp("x", "m.", "y")
    elif model in ("attention", "attention_hot"):
        g.attention("x", "a.", "y", hot=model == "attention_hot")
    else:
        mid = g.attention("x", "a.", "h1", hot=False)
        g.mlp(mid, "m.", "y")
    return Graph(f"{model}/{shape}", (b, s, h), g.ops, g.params)


# ---- golden reference (float64, op by op) --------------------------------
def _ref_op(kind: str, args: list[np.ndarray], attrs: dict) -> np.ndarray:
    if kind == "layernorm":
        x = args[0]
        return (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5)
    if kind == "matmul":
        return args[0] @ args[1]
    if kind == "matmul_t":
        return args[0] @ np.swapaxes(args[1], -1, -2)
    if kind == "add":
        return args[0] + args[1]
    if kind == "gelu":
        x = args[0]
        return 0.5 * x * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))
    if kind == "scale":
        return args[0] * attrs["factor"]
    if kind == "softmax":
        x = args[0] - args[0].max(-1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(-1, keepdims=True)
    raise ValueError(kind)


def run_reference(graph: Graph, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Every tensor in the graph, float64. Used for the final check AND per-tensor localization."""
    env: dict[str, np.ndarray] = {**{k: v.astype(np.float64) for k, v in graph.params.items()},
                                  **{k: v.astype(np.float64) for k, v in feeds.items()}}
    for op in graph.ops:
        env[op.output] = _ref_op(op.kind, [env[i] for i in op.inputs], op.attrs)
    return env


def infer_shapes(graph: Graph) -> dict[str, tuple[int, ...]]:
    env = run_reference(graph, {"x": np.zeros(graph.shape)})
    return {k: v.shape for k, v in env.items()}
