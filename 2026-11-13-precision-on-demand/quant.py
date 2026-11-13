"""Simulated weight quantization: store int8, read either all 8 bits or only the top 4.

Each weight matrix is quantized per output channel (last axis) with a
symmetric int8 code q in [-127, 127].

* High pass  : w_hi = q * s                        (8 bits read per weight)
* Low pass   : keep only the top nibble of q (q >> 4, an arithmetic shift), i.e.
               the value is known to lie in [16*hi, 16*hi + 15]; reconstruct at the
               bucket midpoint: w_lo = (16 * (q >> 4) + 7.5) * s   (4 bits read)

Escalating a step only requires reading the remaining low nibble, so the
storage format is a single int8 tensor shared by both passes.
"""
import numpy as np

from model import WEIGHT_NAMES


def quantize_int8(w):
    """Per-output-channel symmetric int8. Returns (q:int8, scale:float32 broadcastable)."""
    amax = np.abs(w).max(axis=0, keepdims=True)
    scale = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
    q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
    return q, scale


def dequant_high(q, scale):
    return q.astype(np.float32) * scale


def dequant_low(q, scale):
    top = np.right_shift(q.astype(np.int16), 4)  # floor(q / 16), arithmetic for negatives
    return (16.0 * top + 7.5).astype(np.float32) * scale


def make_views(params):
    """Return (high_params, low_params) sharing the same int8 storage."""
    hi, lo = dict(params), dict(params)
    for k in WEIGHT_NAMES:
        q, s = quantize_int8(params[k])
        hi[k] = dequant_high(q, s)
        lo[k] = dequant_low(q, s)
    return hi, lo


def weight_count(params):
    return sum(params[k].size for k in WEIGHT_NAMES)


def bytes_per_token(escalation_rate, mode="nibble"):
    """Weight bytes read per token, normalized so that all-high-precision = 1.0.

    mode="nibble": escalation reads only the extra low 4 bits (0.5 + 0.5 r)
    mode="rerun" : escalation re-reads the full 8-bit weights   (0.5 + 1.0 r)
    """
    if mode == "nibble":
        return 0.5 + 0.5 * escalation_rate
    if mode == "rerun":
        return 0.5 + 1.0 * escalation_rate
    raise ValueError(mode)
