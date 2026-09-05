#!/usr/bin/env python3
"""Generate EVERY include/matmul_*.h from real TinyLlama tensors, quantized by MXQuant.

This generalizes `gen_matmul_fp8_64x64_llama.py` from one 64x64 header to the whole
`matmul_tiled_*` family. Nothing in any generated header is synthetic:

  operands   real TinyLlama activations and weights, sliced out of ONE contiguous 512x512
             (A_square, W_square) pair captured by `npu-exploration/app/capture_llama_tiles.py`
             (which reuses MXQuant's own `log_pairs_from_eval.PairLogger`). The pair shares the
             in-feature offset `i0`, so `A @ B` is a genuine sub-block of that projection's real
             output -- not a collage of unrelated 32x32 windows.
  quantizer  MXQuant's `quantize_mx_block32`, reached through `app/mxq_golden.py`, which converts
             its (P, X) value output into the hardware wire format losslessly.
  C_out_bf16 those operands through `fp8_matmul_model.tiled_matmul_hwlike`, the bit-exact mesh
             model (per-lane accumulator precision, truncating product quantization, BF16
             accumulate). This is what the NON-requant tests check.
  C_out      the requantizer's own convention applied to that exact BF16 tile. This is what the
  C_scales   REQUANT tests check, and it isolates the requantizer: if C_out_bf16 matches, then
             C_out matching means the requantizer agrees with the reference, with no accumulator
             modelling in the way.

BLOCK SCALE CONVENTION, per format. FP8 follows MXQuant's e2e convention
(`X = 2^floor(log2 amax)`, block max in [1,2)) on both operands and output, which is what
`MxRequantizer.scala` and `gemmini.cc`'s FP8 post-pass now implement -- see
`npu-exploration/planning/chain_seam_hw_notes.md` section 8. FP4 and FP6 were NOT migrated: their
spike paths still subtract `log2_pmax` (`gemmini.cc:1366` FP6 -> 4, `gemmini.cc:1478` FP4 -> 2), so
their OUTPUT requantization keeps that shift. Operands are the e2e convention for every format.

Run it with the npu-exploration venv, which has torch, ninja and the MXQuant repo wired up:

    cd generators/gemmini/software/gemmini-rocc-tests
    PATH=../../npu-exploration/.venv/bin:$PATH \
      ../../npu-exploration/.venv/bin/python3 gen_matmul_llama.py            # all headers
      ../../npu-exploration/.venv/bin/python3 gen_matmul_llama.py fp8_64x64  # just one
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
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

BLOCK = 32

#: Contiguous 512x512 (A,W) pairs from a real forward pass. See app/capture_llama_tiles.py.
DATA = MXQ_ROOT / "end_to_end_linear" / "systolic_simulation" / "data_evalrun_512"
#: The N=32 random-window tiles the older generator stitched. Only used if DATA is absent.
DATA_LEGACY = MXQ_ROOT / "end_to_end_linear" / "systolic_simulation" / "data_evalrun_01"

#: Mesh precision, from ConfigsFP.scala's meshProdPrecisionList / meshAccPrecisionList. Written
#: (exp_bits, frac_bits); the same schedule appears in gemmini.cc as acc_e[]/acc_m[].
PROD_PRECISION = [(4, 3)] * 16
ACC_PRECISION = [(4, 4)] * 8 + [(4, 5)] * 2 + [(4, 6)] * 5 + [(8, 7)] * 1


# --- formats -----------------------------------------------------------------------------------

@dataclass(frozen=True)
class Format:
    """One MX element format, as both oracles implement it."""
    name: str
    mxq: str          #: the alias quantize_mx_block32 understands
    bits: int         #: element width; 4 and 6 are nibble-packed / LUT-indexed in the headers
    #: `log2_pmax_floor` the requantizer subtracts when it commits OUTPUT blocks. 0 is MXQuant's
    #: e2e convention (FP8, after chain_seam_hw_notes.md section 8); FP6/FP4 still use the OCP
    #: `- emax` shift in `gemmini.cc`, so their goldens must reproduce it.
    out_pmax: int
    #: mesh tile geometry. FP8 marches 16x16; the nibble formats pack 32x32 (fp4_matmul_model.py:28,
    #: gemmini.cc:1303 `const int TM = 32, TN = 32`).
    tile_m: int = 16
    tile_n: int = 16
    tile_k: int = 16
    #: which model module carries the matching mesh + code encoder
    model: str = "fp8_matmul_model"
    #: who requantizes the OUTPUT. "mxquant" is MXQuant's `_quantize_elemwise`, which FP8's
    #: requantizer was migrated to. FP4/FP6 were not: `gemmini.cc:1508` requantizes with
    #: `q_bf16_rne(v/scale)` then `hw_bf16_to_e2m1`, a two-stage RNE->E3M1->E2M1 that rounds ties
    #: differently from MXQuant (`mx_fp_math.h:402-404` says so outright). Their goldens must use
    #: the hardware's quantizer or they disagree on ~14% of elements in the mantissa LSB.
    out_requant: str = "mxquant"

FORMATS = {
    "fp8": Format("fp8:e4m3", "MXFP8_E4M3", 8, out_pmax=0),
    "fp6": Format("fp6:e3m2", "MXFP6_E3M2", 6, out_pmax=4,
                  tile_m=32, tile_n=32, model="fp4_matmul_model", out_requant="model"),
    "fp4": Format("fp4:e2m1", "MXFP4", 4, out_pmax=2,
                  tile_m=32, tile_n=32, model="fp4_matmul_model", out_requant="model"),
}


# --- the shape table ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Shape:
    """One generated header: its dimensions, format, and where its operands come from."""
    header: str                 #: file under include/
    M: int
    K: int
    N: int
    fmt: str
    layer: str = "layer0"
    proj: str = "mlp.gate_proj"
    guard: str = ""
    #: tests that consume this header, for the report
    tests: tuple[str, ...] = field(default_factory=tuple)

    @property
    def GK(self) -> int: return self.K // BLOCK

    @property
    def GN(self) -> int: return self.N // BLOCK


SHAPES = [
    Shape("matmul_fp8_32x32x32.h", 32, 32, 32, "fp8",
          guard="INCLUDE_MATMUL_FP8_32X32X32_H",
          tests=("matmul_tiled_fp8_32x32x32", "matmul_tiled_fp8_32x32x32_requant")),
    Shape("matmul_fp8_64x64.h", 64, 64, 64, "fp8",
          guard="INCLUDE_MATMUL_FP8_64X64_H",
          tests=("matmul_tiled_fp8_64x64", "matmul_tiled_fp8_64x64_requant",
                 "matmul_tiled_fp8_64x64_DRAMMvout", "matmul_tiled_fp8_64x64_smem_mvout")),
    Shape("matmul_fp8_96x32x32.h", 96, 32, 32, "fp8", proj="mlp.up_proj",
          guard="INCLUDE_MATMUL_FP8_96X32X32_H",
          tests=("matmul_tiled_fp8_96x32x32", "matmul_tiled_fp8_96x32x32_requant")),
    Shape("matmul_fp8_64x96x64.h", 64, 64, 96, "fp8", proj="attn.q_proj",
          guard="INCLUDE_MATMUL_FP8_64X96X64_H",
          tests=("matmul_tiled_fp8_64x96x64", "matmul_tiled_fp8_64x96x64_requant")),
    Shape("matmul_fp8_96x96x64.h", 96, 64, 96, "fp8", proj="attn.o_proj",
          guard="INCLUDE_MATMUL_FP8_96X96X64_H",
          tests=("matmul_tiled_fp8_96x96x64", "matmul_tiled_fp8_96x96x64_requant")),
    Shape("matmul_fp8_128x128.h", 128, 128, 128, "fp8", proj="mlp.down_proj",
          guard="INCLUDE_MATMUL_FP8_128X128_H",
          tests=("matmul_tiled_fp8_128x128", "matmul_tiled_fp8_128x128_requant")),
    Shape("matmul_fp8_128x128x256.h", 128, 256, 128, "fp8", layer="layer1",
          guard="INCLUDE_MATMUL_FP8_128X128X256_H",
          tests=("matmul_tiled_fp8_128x128x256", "matmul_tiled_fp8_128x128x256_DRAMMvout")),

    Shape("matmul_fp4_64x64.h", 64, 64, 64, "fp4", layer="layer2",
          guard="MATMUL_DATA_FP4_H",
          tests=("matmul_tiled_fp4_64x64", "matmul_tiled_fp4_64x64_requant",
                 "matmul_tiled_fp4_64x64_DRAMMvout")),
    Shape("matmul_fp4_128x128.h", 128, 128, 128, "fp4", layer="layer2", proj="mlp.up_proj",
          guard="INCLUDE_MATMUL_FP4_128X128_H",
          tests=("matmul_tiled_fp4_128x128", "matmul_tiled_fp4_128x128_requant")),
    Shape("matmul_fp4_128x128x512.h", 128, 512, 128, "fp4", layer="layer3", proj="mlp.down_proj",
          guard="INCLUDE_MATMUL_FP4_128X128X512_H",
          tests=("matmul_tiled_fp4_128x128x512", "matmul_tiled_fp4_128x128x512_requant")),
]

BY_NAME = {s.header.removeprefix("matmul_").removesuffix(".h"): s for s in SHAPES}


# --- operand sourcing --------------------------------------------------------------------------

def load_pair(shape: Shape) -> tuple[np.ndarray, np.ndarray]:
    """Real ``(A[M][K], B[K][N])`` sliced from one contiguous logged (A_square, W_square) pair.

    ``A_square`` is ``[tokens][in_features]`` and ``W_square`` is ``[out_features][in_features]``
    (``log_pairs_from_eval.py:193-195``), both taken at the same in-feature offset, so
    ``A_square[:M,:K] @ W_square[:N,:K].T`` is a real sub-block of that projection's own output.
    """
    d = DATA / shape.layer / shape.proj
    if not d.is_dir():
        raise SystemExit(
            f"missing {d}\nRun the capture first:\n"
            f"    cd {NPU} && .venv/bin/python3 -m app.capture_llama_tiles")

    def tile(which: str) -> np.ndarray:
        p = d / f"{which}_square.npz"
        if not p.exists():
            raise SystemExit(f"missing logged tile {p}")
        with np.load(p) as z:
            return z["data"].astype(np.float32)

    A_full, W_full = tile("A"), tile("W")
    need = (max(shape.M, shape.N), shape.K)
    if A_full.shape[0] < shape.M or A_full.shape[1] < shape.K:
        raise SystemExit(f"{d}/A_square.npz is {A_full.shape}, need at least {(shape.M, shape.K)}")
    if W_full.shape[0] < shape.N or W_full.shape[1] < shape.K:
        raise SystemExit(f"{d}/W_square.npz is {W_full.shape}, need at least {(shape.N, shape.K)}")
    del need

    A = np.ascontiguousarray(A_full[:shape.M, :shape.K])
    B = np.ascontiguousarray(W_full[:shape.N, :shape.K].T)     # [out][in] -> [K][N]
    return A, B


# --- emission ----------------------------------------------------------------------------------

def _decode_codes(codes: np.ndarray, e_bits: int, m_bits: int) -> np.ndarray:
    """Decode raw (1+e+m)-bit float codes. Mirrors mx_fp_math.h's decoders, format-generically."""
    c = np.asarray(codes, dtype=np.uint32)
    bias = (1 << (e_bits - 1)) - 1
    s = (c >> (e_bits + m_bits)) & 1
    e = (c >> m_bits) & ((1 << e_bits) - 1)
    m = c & ((1 << m_bits) - 1)
    scale = float(1 << m_bits)
    val = np.where(e == 0,
                   (m / scale) * 2.0 ** (1 - bias),
                   (1.0 + m / scale) * 2.0 ** (e.astype(np.int64) - bias))
    return np.where(s == 1, -val, val).astype(np.float32)


def quantize(V: np.ndarray, *, axis: str, f: Format, pmax_shift: int = 0):
    """``(codes, scales, P)`` for ``V`` in ``f``'s wire format.

    FP8 goes through ``app.mxq_golden.golden``, which encodes by exact table lookup and asserts
    losslessness. The nibble formats have no such table, so they use the repo's own encoder --
    ``tensor_to_custom_fp_codes``, the one that wrote the existing headers -- and the losslessness
    assertion is made here instead, by decoding the codes back and requiring ``P`` exactly.
    """
    if f.bits == 8:
        g = golden(V, axis=axis, fmt=f.mxq, pmax_shift=pmax_shift)
        return g.codes, g.scales, g.P

    from app.mxq_golden import e8m0_encode_exact, _shift_scale
    from end_to_end_linear.mx_block_quant import quantize_mx_block32
    model = __import__(f.model)

    out = quantize_mx_block32(torch.from_numpy(V), fmt=f.mxq, axis=axis)
    if pmax_shift:
        out = _shift_scale(V, out, fmt=f.mxq, axis=axis, pmax_shift=pmax_shift)
    P = out.P.numpy().astype(np.float32)
    scales = e8m0_encode_exact(out.X.numpy().astype(np.float32))

    raw, total_bits = model.tensor_to_custom_fp_codes(torch.from_numpy(P), f.name)
    codes = np.array(raw, dtype=np.uint8)
    assert total_bits == f.bits, f"{f.name} encoder returned {total_bits} bits, expected {f.bits}"

    e_bits, m_bits = model.parse_fp_spec(f.name)
    back = _decode_codes(codes, e_bits, m_bits)
    if not np.array_equal(back, P):
        i = int(np.argmax((back != P).ravel()))
        raise AssertionError(
            f"{f.name} wire encoding is not lossless at flat index {i}: reference P = "
            f"{P.ravel()[i]!r}, decode(code 0x{int(codes.ravel()[i]):x}) = {back.ravel()[i]!r}")
    return codes, scales, P


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


def build(shape: Shape, verbose: bool = True) -> dict[str, np.ndarray]:
    """Quantize, run the mesh model, requantize. Returns everything the header needs."""
    f = FORMATS[shape.fmt]
    A, B = load_pair(shape)
    if verbose:
        print(f"  operands {shape.layer}/{shape.proj}  A{A.shape} |max|={np.abs(A).max():.6g}  "
              f"B{B.shape} |max|={np.abs(B).max():.6g}")

    # A is blocked along K (its columns); B along K (its rows).
    A_codes, A_scales, A_P = quantize(A, axis="row", f=f)     # scales [M][GK]
    B_codes, B_scales, B_P = quantize(B, axis="col", f=f)     # scales [GK][N]

    # The mesh model takes the code VALUES and the block scales separately, exactly as the
    # datapath does (raw code products accumulate first, scales apply afterwards).
    mm = __import__(f.model)
    C_bf16 = mm.tiled_matmul_hwlike(
        torch.from_numpy(A_P), torch.from_numpy(B_P),
        torch.from_numpy(e8m0_decode(A_scales).astype(np.float32)),       # [M][GK]
        torch.from_numpy(e8m0_decode(B_scales).astype(np.float32)),       # [GK][N]
        verbose=False, prod_precision_list=PROD_PRECISION,
        acc_precision_list=ACC_PRECISION,
    ).numpy().astype(np.float32)

    if not np.isfinite(C_bf16).all():
        raise SystemExit(f"{shape.header}: mesh model produced non-finite output -- operands "
                         "exceed the accumulator bound, pick a different (layer, proj)")

    # Requantize that exact BF16 tile. axis="row" is the requantizer's own layout: one E8M0 byte
    # per row per 32 output columns, i.e. [M][N/32].
    if f.out_requant == "model":
        from app.mxq_golden import e8m0_encode_exact
        C_q, C_sc = mm.matrix_mx_requantize(torch.from_numpy(C_bf16), f.name)
        C_codes = np.array(mm.tensor_to_custom_fp_codes(C_q, f.name)[0], dtype=np.uint8)
        C_scales = e8m0_encode_exact(C_sc.numpy().astype(np.float32))
    else:
        C_codes, C_scales, _ = quantize(C_bf16, axis="row", f=f, pmax_shift=f.out_pmax)
    if verbose:
        mask = (1 << (f.bits - 1)) - 1
        print(f"  C_bf16 |max|={np.abs(C_bf16).max():.6g}   requant peak code "
              f"0x{int(np.max(C_codes & mask)):02X}, E8M0 {int(C_scales.min())}.."
              f"{int(C_scales.max())}")

    return dict(A_codes=A_codes, B_codes=B_codes,
                A_scales=A_scales, B_scales=B_scales,
                C_bf16=C_bf16, C_codes=C_codes, C_scales=C_scales)


def emit(shape: Shape, d: dict[str, np.ndarray]) -> Path:
    f = FORMATS[shape.fmt]
    path = HERE / "include" / shape.header
    with open(path, "w") as fh:
        fh.write(f"""// GENERATED by gen_matmul_llama.py -- do not edit by hand.
//
// Operands are REAL TinyLlama tensors: A is activations and B is weights, both sliced out of one
// contiguous 512x512 (A_square, W_square) pair logged at {shape.layer}/{shape.proj} by
// npu-exploration/app/capture_llama_tiles.py. The pair shares its in-feature offset, so A @ B is a
// genuine sub-block of that projection's own output. Quantized by MXQuant itself
// (quantize_mx_block32, block 32, {f.name}).
//
// C_out_bf16 is those operands through the bit-exact mesh model; C_out / C_scales_out are the
// requantizer's convention applied to that BF16 tile, in the requantizer's own [M][N/32] scale
// layout. Block scales use X = 2^(floor(log2 amax) - {f.out_pmax}); see
// npu-exploration/planning/chain_seam_hw_notes.md section 8.
#ifndef {shape.guard}
#define {shape.guard}

#include <stdint.h>

#define MATMUL_M {shape.M}
#define MATMUL_K {shape.K}
#define MATMUL_N {shape.N}
#define MATMUL_GK {shape.GK}
#define MATMUL_GN {shape.GN}

// Input precision: {f.name}
static const uint8_t A_in[MATMUL_M][MATMUL_K] = {{
{_rows(d['A_codes'], 2)}
}};

static const uint8_t B_in[MATMUL_K][MATMUL_N] = {{
{_rows(d['B_codes'], 2)}
}};

// A's scale is per row per K-group: a_off = group * M + row
static const uint8_t A_scales_row[MATMUL_GK][MATMUL_M] = {{
{_rows(d['A_scales'].T, 2)}
}};

// B's scale is per column per K-group: b_off = group * N + col
static const uint8_t B_scales_col[MATMUL_GK][MATMUL_N] = {{
{_rows(d['B_scales'], 2)}
}};

// Non-requant output: BF16 bit patterns.
static const uint16_t C_out_bf16[MATMUL_M][MATMUL_N] = {{
{_rows(bf16_bits(d['C_bf16']), 4)}
}};

// Requant output: {f.name} codes, and the E8M0 block scales in the layout the requantizer WRITES
// (one byte per row per 32 output columns).
static const uint8_t C_out[MATMUL_M][MATMUL_N] = {{
{_rows(d['C_codes'], 2)}
}};

static const uint8_t C_scales_out[MATMUL_M][MATMUL_GN] = {{
{_rows(d['C_scales'], 2)}
}};

// Transpose of C_scales_out, kept for tests written against the older [GN][M] declaration.
static const uint8_t C_scales_row[MATMUL_GN][MATMUL_M] = {{
{_rows(d['C_scales'].T, 2)}
}};

#endif // {shape.guard}
""")
    return path


def emit_fp4(shape: Shape, d: dict[str, np.ndarray]) -> Path:
    """FP4 header: nibble-packed operands in the layouts the FP4 datapath reads.

    A and C go into the HW-tiled layout (`lut_golden_model._a_indices_to_hw_layout`): within each
    32x16 block, two m-rows share a byte, low nibble = even row. B is packed along N, low nibble =
    even column. Same packers `fp4_matmul_model.write_c_header_fp4_direct` used, so only the DATA
    changes -- real llama tensors through MXQuant instead of `torch.randn` and random scales.
    """
    from lut_golden_model import _a_indices_to_hw_layout
    f = FORMATS[shape.fmt]
    M, K, N = shape.M, shape.K, shape.N

    A_hw = np.array(_a_indices_to_hw_layout(d["A_codes"].tolist(), f.tile_m, f.tile_k),
                    dtype=np.uint8)                                    # [M/2][K]
    B = d["B_codes"]
    B_packed = ((B[:, 1::2].astype(np.uint16) << 4) | B[:, 0::2]).astype(np.uint8)   # [K][N/2]
    C_hw = np.array(_a_indices_to_hw_layout(d["C_codes"].tolist(), f.tile_m, f.tile_k),
                    dtype=np.uint8)                                    # [M/2][N]

    path = HERE / "include" / shape.header
    with open(path, "w") as fh:
        fh.write(f"""// GENERATED by gen_matmul_llama.py -- do not edit by hand.
//
// Operands are REAL TinyLlama tensors: A is activations and B is weights, both sliced out of one
// contiguous 512x512 (A_square, W_square) pair logged at {shape.layer}/{shape.proj} by
// npu-exploration/app/capture_llama_tiles.py. Quantized by MXQuant itself (quantize_mx_block32,
// block 32, {f.name}).
//
// Block scales use X = 2^(floor(log2 amax) - {f.out_pmax}) on the OUTPUT: unlike FP8, the FP4
// requantizer was not migrated to MXQuant's e2e convention and still subtracts log2_pmax
// (gemmini.cc:1478). Operands use the e2e convention.
#ifndef {shape.guard}
#define {shape.guard}

#include <stdint.h>

#define MATMUL_M   {M}
#define MATMUL_K   {K}
#define MATMUL_N   {N}
#define MATMUL_GK  {shape.GK}
#define MATMUL_GN  {shape.GN}
#define A_TILE_M   {f.tile_m}
#define K_TILE     {f.tile_k}

// Input precision: {f.name} (direct 4-bit FP4 codes, no LUT)
// A stored in HW tiled layout: per (A_TILE_M x K_TILE) block,
//   pairs of m-rows interleaved per k, then adjacent column pairs nibble-packed.
//   Dimensions: [M/2][K] = [{M // 2}][{K}]
//   byte layout: bits[7:4]=fp4(row 2r+1, k), bits[3:0]=fp4(row 2r, k)
static const uint8_t A_in_hw[{M // 2}][{K}] = {{
{_rows(A_hw, 2)}
}};

// B stored in standard layout: odd-col nibble in high bits, even-col in low bits
//   Dimensions: [K][N/2] = [{K}][{N // 2}]
static const uint8_t B_in[{K}][{N // 2}] = {{
{_rows(B_packed, 2)}
}};

// Per-row per-32-K-group A scales in fpe8m0
static const uint8_t A_scales_row[{shape.GK}][{M}] = {{
{_rows(d['A_scales'].T, 2)}
}};

// Per-col per-32-K-group B scales in fpe8m0
static const uint8_t B_scales_col[{shape.GK}][{N}] = {{
{_rows(d['B_scales'], 2)}
}};

// Final output requantized to {f.name} (direct 4-bit codes)
// HW tiled layout: [M/2][N] = [{M // 2}][{N}]
//   byte layout: bits[7:4]=fp4(row 2r+1, n), bits[3:0]=fp4(row 2r, n)
static const uint8_t C_out[{M // 2}][{N}] = {{
{_rows(C_hw, 2)}
}};

// Per-row per-32-N-group C output scales in fpe8m0, as the requantizer WRITES them [M][GN]
static const uint8_t C_scales_out[{M}][{shape.GN}] = {{
{_rows(d['C_scales'], 2)}
}};

// Transpose of C_scales_out, kept for tests written against the older [GN][M] declaration.
static const uint8_t C_scales_row[{shape.GN}][{M}] = {{
{_rows(d['C_scales'].T, 2)}
}};

// Final output (scaled+accumulated), bf16
static const uint16_t C_out_bf16[{M}][{N}] = {{
{_rows(bf16_bits(d['C_bf16']), 4)}
}};

#endif // {shape.guard}
""")
    return path


EMITTERS = {"fp8": emit, "fp4": emit_fp4}


# --- chained (back-to-back) matmul --------------------------------------------------------------
#
# C2 = (A1 @ B1) @ B2, all fp8. MM1's requantized fp8 output C1 stays RESIDENT in the scratchpad
# and is fed to MM2 as operand A -- exactly as the hardware reads it: the fp8 code VALUES multiply
# in the mesh and C1's block scales apply afterwards. MM1's output block-scales are reused as MM2's
# input A-scales. This is the golden for matmul_tiled_fp8_64x64_chain.c (Step 2 of the standalone
# plan): C1_out/C1_scales_out check residency, C2_out/C2_scales_out check the final chained result.


def _run_mesh(A_P: np.ndarray, A_scales: np.ndarray,
              B_P: np.ndarray, B_scales: np.ndarray, f: Format) -> np.ndarray:
    """A_P @ B_P through the bit-exact mesh model, block scales applied after the raw products."""
    mm = __import__(f.model)
    C = mm.tiled_matmul_hwlike(
        torch.from_numpy(A_P), torch.from_numpy(B_P),
        torch.from_numpy(e8m0_decode(A_scales).astype(np.float32)),
        torch.from_numpy(e8m0_decode(B_scales).astype(np.float32)),
        verbose=False, prod_precision_list=PROD_PRECISION, acc_precision_list=ACC_PRECISION,
    ).numpy().astype(np.float32)
    if not np.isfinite(C).all():
        raise SystemExit("chain: mesh model produced non-finite output -- operands exceed the "
                         "accumulator bound, pick a different (layer, proj) or B2 slice")
    return C


def _requant_fp8(C_bf16: np.ndarray, f: Format):
    """Requantize a BF16 tile to fp8. Returns (codes [M][N], scales [M][N/32], P [M][N] values).

    P is the fp8 code VALUE of each element -- what a subsequent matmul multiplies when this tile
    is reused as an operand -- so it is captured (build() discards it).
    """
    assert f.out_requant == "mxquant", "chain golden is fp8-only for now"
    codes, scales, P = quantize(C_bf16, axis="row", f=f, pmax_shift=f.out_pmax)
    return codes, scales, P


def load_B2(shape: Shape) -> np.ndarray:
    """A DISTINCT real weight slice for MM2, [K][N]. Same logged W_square, different out-features.

    B1 is W_full[:N, :K].T (see load_pair). B2 takes the NEXT N out-features at the same in-feature
    offset -> W_full[N:2N, :K].T, so it is genuinely different real weight data of the right shape,
    not a reshuffle of B1. (This is a distinct real weight sub-block; it does not claim to be the
    literal next layer -- the point is bit-exact Spike/RTL/model consistency on real values.)
    """
    d = DATA / shape.layer / shape.proj
    with np.load(d / "W_square.npz") as z:
        W_full = z["data"].astype(np.float32)
    off = shape.N
    if W_full.shape[0] < off + shape.N:
        raise SystemExit(f"{d}/W_square.npz is {W_full.shape}, need >= {off + shape.N} rows for B2")
    return np.ascontiguousarray(W_full[off:off + shape.N, :shape.K].T)   # [K][N]


def build_chain(shape: Shape, verbose: bool = True) -> dict[str, np.ndarray]:
    """MM1 = A1@B1 -> C1 (fp8, resident); MM2 = C1@B2 -> C2 (fp8). Returns all header arrays."""
    f = FORMATS[shape.fmt]
    assert f.name.startswith("fp8"), "chain golden is fp8-only for now"

    A1, B1 = load_pair(shape)
    B2 = load_B2(shape)
    if verbose:
        print(f"  MM1 operands {shape.layer}/{shape.proj}  A1{A1.shape} B1{B1.shape}   "
              f"MM2 B2{B2.shape} (W out-features {shape.N}..{2*shape.N})")

    # MM1 -------------------------------------------------------------------------------------
    A1_codes, A1_scales, A1_P = quantize(A1, axis="row", f=f)   # [M][GK]
    B1_codes, B1_scales, B1_P = quantize(B1, axis="col", f=f)   # [GK][N]
    C1_bf16 = _run_mesh(A1_P, A1_scales, B1_P, B1_scales, f)
    C1_codes, C1_scales, C1_P = _requant_fp8(C1_bf16, f)        # C1_scales [M][N/32]

    # MM2: C1 (resident fp8) @ B2. A2 fed exactly as the mesh reads C1 -----------------------
    B2_codes, B2_scales, B2_P = quantize(B2, axis="col", f=f)   # [GK][N]
    # A2_P = C1's fp8 code values; A2_scales = C1's output block scales ([M][N1/32] == [M][K2/32]).
    C2_bf16 = _run_mesh(C1_P, C1_scales, B2_P, B2_scales, f)
    C2_codes, C2_scales, C2_P = _requant_fp8(C2_bf16, f)

    if verbose:
        mask = (1 << (f.bits - 1)) - 1
        print(f"  C1 |max|={np.abs(C1_bf16).max():.6g} peak 0x{int(np.max(C1_codes & mask)):02X} "
              f"E8M0 {int(C1_scales.min())}..{int(C1_scales.max())}")
        print(f"  C2 |max|={np.abs(C2_bf16).max():.6g} peak 0x{int(np.max(C2_codes & mask)):02X} "
              f"E8M0 {int(C2_scales.min())}..{int(C2_scales.max())}")

    return dict(A_codes=A1_codes, A_scales=A1_scales, B_codes=B1_codes, B_scales=B1_scales,
                B2_codes=B2_codes, B2_scales=B2_scales,
                C1_codes=C1_codes, C1_scales=C1_scales, C1_bf16=C1_bf16,
                C2_codes=C2_codes, C2_scales=C2_scales, C2_bf16=C2_bf16)


def emit_chain(shape: Shape, d: dict[str, np.ndarray]) -> Path:
    """Emit include/matmul_fp8_<M>x<N>_chain.h -- MM1 operands, B2, and both C1 and C2 goldens."""
    f = FORMATS[shape.fmt]
    stem = f"matmul_fp8_{shape.M}x{shape.N}_chain"
    guard = f"INCLUDE_{stem.upper()}_H"
    path = HERE / "include" / f"{stem}.h"
    with open(path, "w") as fh:
        fh.write(f"""// GENERATED by gen_matmul_llama.py chain_fp8_64x64 -- do not edit by hand.
//
// CHAINED back-to-back fp8 matmul: C2 = (A1 @ B1) @ B2, all {f.name}. MM1's requantized fp8 output
// C1 stays RESIDENT in the scratchpad and is reused as MM2's operand A; MM1's output block-scales
// are reused as MM2's input A-scales. A1/B1 are real TinyLlama tiles from {shape.layer}/{shape.proj}
// (A_square/W_square, block 32, MXQuant). B2 is a distinct real weight sub-block (next {shape.N}
// out-features of the same W_square). C_out_bf16/C1 are the mesh model + requantizer on MM1; C2 the
// same on MM2 with C1 fed as the mesh reads it (fp8 code values, block scales applied after).
#ifndef {guard}
#define {guard}

#include <stdint.h>

#define MATMUL_M {shape.M}
#define MATMUL_K {shape.K}
#define MATMUL_N {shape.N}
#define MATMUL_GK {shape.GK}
#define MATMUL_GN {shape.GN}

// ---- MM1 operands (A1 @ B1) ----
// Input precision: {f.name}
static const uint8_t A_in[MATMUL_M][MATMUL_K] = {{
{_rows(d['A_codes'], 2)}
}};

static const uint8_t B_in[MATMUL_K][MATMUL_N] = {{
{_rows(d['B_codes'], 2)}
}};

// A's scale is per row per K-group: a_off = group * M + row
static const uint8_t A_scales_row[MATMUL_GK][MATMUL_M] = {{
{_rows(d['A_scales'].T, 2)}
}};

// B's scale is per column per K-group: b_off = group * N + col
static const uint8_t B_scales_col[MATMUL_GK][MATMUL_N] = {{
{_rows(d['B_scales'], 2)}
}};

// ---- MM2 weight operand (C1 @ B2); C1 comes from the scratchpad, not DRAM ----
static const uint8_t B2_in[MATMUL_K][MATMUL_N] = {{
{_rows(d['B2_codes'], 2)}
}};

static const uint8_t B2_scales_col[MATMUL_GK][MATMUL_N] = {{
{_rows(d['B2_scales'], 2)}
}};

// ---- MM1 output C1 = requant(A1 @ B1): the RESIDENT operand + residency check ----
// C1 fp8 codes (what mvout of the scratchpad must reproduce).
static const uint8_t C1_out[MATMUL_M][MATMUL_N] = {{
{_rows(d['C1_codes'], 2)}
}};

// C1 block scales in the requantizer's [M][N/32] output layout. These ARE MM2's A-scales:
// C1_scales_out[m][b] == A2's per-row per-K-group scale (N1/32 == K2/32).
static const uint8_t C1_scales_out[MATMUL_M][MATMUL_GN] = {{
{_rows(d['C1_scales'], 2)}
}};

static const uint16_t C1_out_bf16[MATMUL_M][MATMUL_N] = {{
{_rows(bf16_bits(d['C1_bf16']), 4)}
}};

// ---- MM2 output C2 = requant(C1 @ B2): the final chained result ----
static const uint8_t C2_out[MATMUL_M][MATMUL_N] = {{
{_rows(d['C2_codes'], 2)}
}};

static const uint8_t C2_scales_out[MATMUL_M][MATMUL_GN] = {{
{_rows(d['C2_scales'], 2)}
}};

static const uint16_t C2_out_bf16[MATMUL_M][MATMUL_N] = {{
{_rows(bf16_bits(d['C2_bf16']), 4)}
}};

#endif // {guard}
""")
    return path


def main() -> int:
    want = sys.argv[1:]
    if len(want) == 1 and want[0].startswith("chain_"):
        key = want[0].removeprefix("chain_")           # e.g. chain_fp8_64x64 -> fp8_64x64
        if key not in BY_NAME:
            raise SystemExit(f"unknown chain shape {key}; known: {sorted(k for k in BY_NAME)}")
        shape = BY_NAME[key]
        t0 = time.time()
        print(f"matmul_fp8_{shape.M}x{shape.N}_chain.h  CHAIN C2=(A1@B1)@B2  "
              f"M={shape.M} K={shape.K} N={shape.N} {FORMATS[shape.fmt].name}")
        d = build_chain(shape)
        p = emit_chain(shape, d)
        print(f"  wrote {p.relative_to(HERE)}  ({time.time() - t0:.1f}s)\n")
        return 0
    if want:
        unknown = [w for w in want if w not in BY_NAME]
        if unknown:
            raise SystemExit(f"unknown shape(s) {unknown}; known: {sorted(BY_NAME)} + chain_fp8_64x64")
        todo = [BY_NAME[w] for w in want]
    else:
        todo = SHAPES

    for shape in todo:
        t0 = time.time()
        print(f"{shape.header}  M={shape.M} K={shape.K} N={shape.N} {FORMATS[shape.fmt].name}")
        d = build(shape)
        p = EMITTERS[shape.fmt](shape, d)
        print(f"  wrote {p.relative_to(HERE)}  ({time.time() - t0:.1f}s)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
