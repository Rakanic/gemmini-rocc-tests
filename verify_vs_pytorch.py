#!/usr/bin/env python3
"""Place spike's output on the ladder between the hardware model and plain PyTorch.

The `matmul_tiled_*` tests already establish the top rung: each one compares spike's output
element-by-element against `C_out_bf16` in the generated header and reports 0 mismatches, and the
requant tests do the same for `C_out` / `C_scales_out`. So

    spike  ==  tiled_matmul_hwlike(...)          bit-exact, asserted by the test itself
    spike  ==  quantize_mx_block32(C_bf16)       bit-exact, asserted by the requant test

What those do NOT say is how far that agreed-upon answer sits from what PyTorch would compute for
the same real llama tensors. This script reports that, for every shape, on four rungs:

    C_fp32      A @ B in fp32          the true llama matmul, no quantization anywhere
    C_mxq       A_hat @ B_hat in fp32  PyTorch's answer for MXQuant-quantized operands, with an
                                       exact accumulator -- "the PyTorch expected output"
    C_mesh      tiled_matmul_hwlike    the bit-exact mesh model == the header == spike
    C_requant   decode(C_out codes * C_scales_out) -- what a chained stage would consume

so the error is attributed: `C_fp32 -> C_mxq` is the cost of quantizing the operands, and
`C_mxq -> C_mesh` is the cost of the mesh's truncating products, reduced-precision per-lane
accumulators and BF16 accumulate. Only the second is hardware; the first is MX itself.

    PATH=../../npu-exploration/.venv/bin:$PATH \
      ../../npu-exploration/.venv/bin/python3 verify_vs_pytorch.py
"""
from __future__ import annotations

import sys

import numpy as np
import torch

from gen_matmul_llama import (  # noqa: E402
    SHAPES, BY_NAME, FORMATS, build, load_pair, quantize, _decode_codes,
)
sys.path.insert(0, str(__import__("gen_matmul_llama").NPU))
from app.mxwire import e8m0_decode  # noqa: E402


def dequant(codes: np.ndarray, scales: np.ndarray, *, axis: str, f, shape) -> np.ndarray:
    """``decode(codes) * broadcast(decode(scales))`` -- the values the mesh's operands stand for."""
    import fp4_matmul_model as _m
    e_bits, m_bits = _m.parse_fp_spec(f.name)
    vals = _decode_codes(codes, e_bits, m_bits)
    sc = e8m0_decode(scales).astype(np.float32)
    tiles = (np.repeat(sc, 32, axis=1)[:, :codes.shape[1]] if axis == "row"
             else np.repeat(sc, 32, axis=0)[:codes.shape[0], :])
    return vals * tiles


def rel(a: np.ndarray, b: np.ndarray) -> float:
    """Relative Frobenius error of ``a`` against reference ``b``."""
    den = np.linalg.norm(b.ravel())
    return float(np.linalg.norm((a - b).ravel()) / den) if den else 0.0


def report(shape) -> dict:
    f = FORMATS[shape.fmt]
    A, B = load_pair(shape)
    d = build(shape, verbose=False)

    # Rung 1: the true llama matmul, fp32, nothing quantized.
    C_fp32 = (A.astype(np.float64) @ B.astype(np.float64)).astype(np.float32)

    # Rung 2: PyTorch's answer for the SAME quantized operands the hardware got, but with an exact
    # fp32 accumulator -- the values the mesh's operand codes stand for.
    A_hat = dequant(d["A_codes"], d["A_scales"], axis="row", f=f, shape=shape)
    B_hat = dequant(d["B_codes"], d["B_scales"], axis="col", f=f, shape=shape)
    C_mxq = (A_hat.astype(np.float64) @ B_hat.astype(np.float64)).astype(np.float32)

    # Rung 3: the bit-exact mesh model == the header's C_out_bf16 == what spike returns.
    C_mesh = d["C_bf16"]

    # Rung 4: the requantized output a chained next stage would consume.
    C_requant = dequant(d["C_codes"], d["C_scales"], axis="row", f=f, shape=shape)

    return dict(
        shape=shape,
        operand_err=rel(C_mxq, C_fp32),
        mesh_err=rel(C_mesh, C_mxq),
        total_err=rel(C_mesh, C_fp32),
        requant_err=rel(C_requant, C_mesh),
        peak=float(np.abs(C_fp32).max()),
    )


def main() -> int:
    want = sys.argv[1:]
    todo = [BY_NAME[w] for w in want] if want else SHAPES

    rows = []
    for shape in todo:
        r = report(shape)
        rows.append(r)
        s = r["shape"]
        print(f"{s.header:<28} M={s.M:<4} K={s.K:<4} N={s.N:<4} {s.layer}/{s.proj}")
        print(f"    operands quantized   C_fp32 -> C_mxq   rel {r['operand_err']:.4%}   (MX itself)")
        print(f"    mesh arithmetic      C_mxq  -> C_mesh  rel {r['mesh_err']:.4%}   (hardware)")
        print(f"    end to end           C_fp32 -> C_mesh  rel {r['total_err']:.4%}")
        print(f"    output requantized   C_mesh -> C_out   rel {r['requant_err']:.4%}\n")

    print(f"{'':28} {'operand':>10} {'mesh':>10} {'end-to-end':>12} {'requant':>10}")
    for r in rows:
        print(f"{r['shape'].header:<28} {r['operand_err']:>9.3%} {r['mesh_err']:>10.3%} "
              f"{r['total_err']:>11.3%} {r['requant_err']:>10.3%}")

    print("\nspike == C_mesh and spike == C_out are asserted bit-exactly by the tests themselves\n"
          "(matmul_tiled_* report 0 mismatches against these very arrays), so every number above\n"
          "is equally a statement about spike.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
