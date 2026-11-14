"""Device kernels. Compute in the device dtype; layernorm accumulates in fp32 like real hardware."""
from __future__ import annotations

import numpy as np

from corestack.versions import v

DTYPES = {"fp32": np.float32, "fp16": np.float16}


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1 + np.tanh(x.dtype.type(0.7978845608) * (x + x.dtype.type(0.044715) * x ** 3)))


def _softmax_stable(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def fused_attention(scores: np.ndarray, values: np.ndarray, factor: float, version: str) -> np.ndarray:
    """softmax(scores * factor) @ values in one kernel (saves a round trip of the s x s matrix).

    kernels 2.1-2.1.x: fp16 'fast path' drops the max-subtraction to save a reduction.
    Correct for |logit| < ~11; overflows to inf -> NaN beyond that.  (B1)
    """
    s = scores * scores.dtype.type(factor)
    fast_path = scores.dtype == np.float16 and v("2.1") <= v(version) < v("2.2")
    with np.errstate(over="ignore", invalid="ignore"):
        if fast_path:
            e = np.exp(s)
            p = e / e.sum(-1, keepdims=True)
        else:
            p = _softmax_stable(s)
    return p @ values


def run(kernel: str, args: list[np.ndarray], attrs: dict, version: str) -> np.ndarray:
    if kernel == "layernorm":
        x = args[0].astype(np.float32)
        y = (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5)
        return y.astype(args[0].dtype)
    if kernel == "matmul":
        return args[0] @ args[1]
    if kernel == "matmul_t":
        return args[0] @ np.swapaxes(args[1], -1, -2)
    if kernel == "add":
        return args[0] + args[1]
    if kernel == "gelu":
        return _gelu(args[0])
    if kernel == "scale":
        return args[0] * args[0].dtype.type(attrs["factor"])
    if kernel == "softmax":
        return _softmax_stable(args[0])
    if kernel == "matmul_bias_gelu":
        return _gelu(args[0] @ args[1] + args[2])
    if kernel == "fused_attention":
        return fused_attention(args[0], args[1], attrs["factor"], version)
    raise ValueError(f"no kernel {kernel!r}")


def flops(kernel: str, shapes: list[tuple[int, ...]]) -> int:
    """Rough FLOP count for the device cycle model."""
    if kernel in ("matmul", "matmul_bias_gelu", "fused_attention", "matmul_t"):
        a, b = shapes[0], shapes[1]
        k = a[-1]
        n = b[-2] if kernel == "matmul_t" else b[-1]
        return 2 * int(np.prod(a[:-1])) * k * n
    return 5 * int(np.prod(shapes[0]))
