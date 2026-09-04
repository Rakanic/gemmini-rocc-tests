#!/usr/bin/env python3
"""Generate include/matmul_fp8_64x64.h from REAL TinyLlama values, quantized by MXQuant.

Every number in the header is derived, nothing is synthetic:

  operands   real TinyLlama activation/weight tiles logged by MXQuant's own eval run
             (MXQuant/end_to_end_linear/systolic_simulation/data_evalrun_01), quantized by
             MXQuant itself via app/mxq_golden.py -> quantize_mx_block32.
  C_out_bf16 those operands through fp8_matmul_model.tiled_matmul_hwlike, the bit-exact mesh
             model (per-lane accumulator precision, truncating product quantization, BF16
             accumulate). This is what the NON-requant test checks.
  C_out      MXQuant's quantizer applied to that exact BF16 tile. This is what the REQUANT test
  C_scales   checks, and it isolates the requantizer: if C_out_bf16 matches, then C_out matching
             means the requantizer agrees with MXQuant, with no accumulator modelling in the way.

This replaces the previous generator path (`fp8_matmul_model.run`), which used `torch.randn`
operands, random power-of-two scales, and `matrix_mx_requantize`'s `log2_pmax = emax = 8` -- the
OCP convention, which places the block max at 448 and is NOT what MXQuant's e2e evaluations use.
See ../../npu-exploration/planning/chain_seam_hw_notes.md §8.

Run it with the npu-exploration venv, which has torch and the MXQuant repo wired up:

    cd generators/gemmini/software/gemmini-rocc-tests
    ../../npu-exploration/.venv/bin/python gen_matmul_fp8_64x64_llama.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NPU = HERE.parent.parent / "npu-exploration"
if not (NPU / "app" / "mxq_golden.py").exists():
    raise SystemExit(f"npu-exploration not found at {NPU}")
sys.path.insert(0, str(NPU))
sys.path.insert(0, str(HERE))

import torch  # noqa: E402
from app.mxq_golden import golden, MXQ_ROOT  # noqa: E402
from app.mxquant import e8m0_decode  # noqa: E402

M = K = N = 64
BLOCK = 32
GK, GN = K // BLOCK, N // BLOCK

#: Which logged tiles to assemble. Real 32x32 tiles are stitched into a 2x2 grid to reach 64x64 --
#: deterministic, and every value stays a real logged one (no upsampling or resampling).
LAYER = "layer0"
PROJ = "mlp.gate_proj"
A_TILES = ["A_square_000.npz", "A_square_001.npz", "A_square_002.npz", "A_square_003.npz"]
W_TILES = ["W_square_000.npz", "W_square_001.npz", "W_square_002.npz", "W_square_003.npz"]

#: Mesh precision, from ConfigsFP.scala's meshProdPrecisionList / meshAccPrecisionList. Written
#: (exp_bits, frac_bits); the same schedule appears in gemmini.cc as acc_e[]/acc_m[].
PROD_PRECISION = [(4, 3)] * 16
ACC_PRECISION = [(4, 4)] * 8 + [(4, 5)] * 2 + [(4, 6)] * 5 + [(8, 7)] * 1

DATA = MXQ_ROOT / "end_to_end_linear" / "systolic_simulation" / "data_evalrun_01"


def _stitch(names: list[str]) -> np.ndarray:
    """Four real 32x32 tiles -> one 64x64, as [[t0, t1], [t2, t3]]."""
    parts = []
    for n in names:
        p = DATA / LAYER / PROJ / n
        if not p.exists():
            raise SystemExit(f"missing logged tile {p}")
        with np.load(p) as z:
            t = z["data"].astype(np.float32)
        if t.shape != (32, 32):
            raise SystemExit(f"{p} is {t.shape}, expected (32, 32)")
        parts.append(t)
    top = np.concatenate([parts[0], parts[1]], axis=1)
    bot = np.concatenate([parts[2], parts[3]], axis=1)
    return np.ascontiguousarray(np.concatenate([top, bot], axis=0))


def bf16_bits(x: np.ndarray) -> np.ndarray:
    """fp32 -> BF16 bit pattern, RNE, matching mx_fp_math.h::f32_to_bf16_rne."""
    x = np.asarray(x, dtype=np.float32)
    u = x.view(np.uint32).astype(np.uint64)
    nan = ((u >> 23) & 0xFF) == 0xFF
    lsb = (u >> 16) & 1
    r = (u + 0x7FFF + lsb) >> 16
    r = np.where(nan, u >> 16, r).astype(np.uint32)
    out = (r & 0xFFFF).astype(np.uint16)
    return np.where(x == 0, np.uint16(0), out).astype(np.uint16)


def _rows(a: np.ndarray, width: int) -> str:
    """Format a 2-D array as C initializer rows of 0x-prefixed values."""
    fmt = f"0x%0{width}x"
    return ",\n".join("    { " + ", ".join(fmt % int(v) for v in row) + " }" for row in a)


def main() -> None:
    A = _stitch(A_TILES)
    B = _stitch(W_TILES)
    print(f"operands from {LAYER}/{PROJ}: A |max| = {np.abs(A).max():.6g}, "
          f"B |max| = {np.abs(B).max():.6g}")

    # A is blocked along K (its columns); B along K (its rows).
    gA = golden(A, axis="row")     # scales [M][GK]
    gB = golden(B, axis="col")     # scales [GK][N]

    # The mesh model takes the code VALUES and the block scales separately, exactly as the
    # datapath does (raw code products accumulate first, scales apply afterwards).
    C_bf16 = torch.zeros(M, N)
    import fp8_matmul_model as mm
    C_bf16 = mm.tiled_matmul_hwlike(
        torch.from_numpy(gA.P), torch.from_numpy(gB.P),
        torch.from_numpy(e8m0_decode(gA.scales).astype(np.float32)),      # [M][GK]
        torch.from_numpy(e8m0_decode(gB.scales).astype(np.float32)),      # [GK][N]
        verbose=False, prod_precision_list=PROD_PRECISION,
        acc_precision_list=ACC_PRECISION,
    ).numpy().astype(np.float32)

    if not np.isfinite(C_bf16).all():
        raise SystemExit("mesh model produced non-finite output -- operands exceed the "
                         "accumulator bound, pick different tiles")
    print(f"C_out_bf16 |max| = {np.abs(C_bf16).max():.6g}")

    # Requantize that exact BF16 tile with MXQuant. axis="row" is the requantizer's own layout:
    # one E8M0 byte per row per 32 output columns, i.e. [M][N/32].
    gC = golden(C_bf16, axis="row")
    peak = int(np.max(gC.codes & 0x7F))
    print(f"requant: peak code 0x{peak:02X}, E8M0 codes "
          f"{int(gC.scales.min())}..{int(gC.scales.max())}")

    hdr = HERE / "include" / "matmul_fp8_64x64.h"
    with open(hdr, "w") as f:
        f.write(f"""// GENERATED by gen_matmul_fp8_64x64_llama.py -- do not edit by hand.
//
// Operands are real TinyLlama tiles ({LAYER}/{PROJ}, four logged 32x32 tiles stitched into a
// 2x2 grid), quantized by MXQuant itself (quantize_mx_block32, block 32, E4M3). C_out_bf16 is
// those operands through the bit-exact mesh model; C_out/C_scales_out are MXQuant's quantizer
// applied to that BF16 tile, in the requantizer's own [M][N/32] scale layout.
//
// Block scales follow MXQuant's e2e convention (X = 2^floor(log2 amax), block max in [1,2)),
// NOT OCP Algorithm 1's `- emax`. See npu-exploration/planning/chain_seam_hw_notes.md §8.
#ifndef INCLUDE_MATMUL_FP8_64X64_H
#define INCLUDE_MATMUL_FP8_64X64_H

#include <stdint.h>

#define MATMUL_M {M}
#define MATMUL_K {K}
#define MATMUL_N {N}
#define MATMUL_GK {GK}
#define MATMUL_GN {GN}

// Input precision: fp8:e4m3
static const uint8_t A_in[MATMUL_M][MATMUL_K] = {{
{_rows(gA.codes, 2)}
}};

static const uint8_t B_in[MATMUL_K][MATMUL_N] = {{
{_rows(gB.codes, 2)}
}};

// A's scale is per row per K-group: a_off = group * M + row
static const uint8_t A_scales_row[MATMUL_GK][MATMUL_M] = {{
{_rows(gA.scales.T, 2)}
}};

// B's scale is per column per K-group: b_off = group * N + col
static const uint8_t B_scales_col[MATMUL_GK][MATMUL_N] = {{
{_rows(gB.scales, 2)}
}};

// Non-requant output: BF16 bit patterns.
static const uint16_t C_out_bf16[MATMUL_M][MATMUL_N] = {{
{_rows(bf16_bits(C_bf16), 4)}
}};

// Requant output: E4M3 codes, and the E8M0 block scales in the layout the requantizer WRITES
// (one byte per row per 32 output columns).
static const uint8_t C_out[MATMUL_M][MATMUL_N] = {{
{_rows(gC.codes, 2)}
}};

static const uint8_t C_scales_out[MATMUL_M][MATMUL_GN] = {{
{_rows(gC.scales, 2)}
}};

#endif // INCLUDE_MATMUL_FP8_64X64_H
""")
    print(f"wrote {hdr}")


if __name__ == "__main__":
    main()
