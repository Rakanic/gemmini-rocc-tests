#!/usr/bin/env python3
"""ASYMMETRIC FP4-activation x FP8_E4M3-SINGLE-weight matmul (mode2, dual throughput 32x16).

FP4 activation is quad (2 rows/lane, direct nibbles); E4M3 weight is DIRECT single (1 col/lane, no
weight LUT). => 2 products/PE, 32x16 output tile. Spike selects single-weight via wgt_single (fp8/
code0, altfmt0, no weight LUT) in the FP4-act branch, giving TN=16 + direct 8-bit B decode.
  A (activation) = FP4 E2M1 4-bit codes, HW-tiled [M/2][K] (direct). TM=32.
  B (weight)     = FP8_E4M3 DIRECT 8-bit codes, row-major [K][N] (exact spike grid, no LUT). TN=16.
Output is BF16. Golden is the throughput-agnostic mesh model (fp8_matmul_model.tiled_matmul_hwlike).
"""
import os
import torch

from lut_golden_model import (
    make_fp_quantizer, tensor_to_custom_fp_codes, codes_to_hex_rows, _a_indices_to_hw_layout,
)

# ---- Parameters ------------------------------------------------------------
SEED     = int(os.environ.get("ASYM_SEED", "0"))
M        = int(os.environ.get("ASYM_M", "64"))
K        = int(os.environ.get("ASYM_K", "64"))
N        = int(os.environ.get("ASYM_N", "64"))
ACT_SPEC = "fp4:e2m1"     # FP4 E2M1 activation (direct, quad)
WEI_SPEC = "fp8:e4m3"     # FP8 E4M3 weight, DIRECT single (no LUT)
SCALE_SPEC = "fpe8m0"
GROUP    = 32
A_TILE_M = 32
K_TILE   = 16
HEADER   = os.environ.get("ASYM_HEADER_PATH", "./include/matmul_data_asym_fp4_e4m3s.h")
DEV      = torch.device("cpu")

Gk = K // GROUP
assert M % A_TILE_M == 0 and K % K_TILE == 0 and K % GROUP == 0

# ---- Inputs ----------------------------------------------------------------
torch.manual_seed(SEED)
A = torch.randn(M, K, device=DEV, dtype=torch.float32)
B = torch.randn(K, N, device=DEV, dtype=torch.float32)

# A: FP4 E2M1 direct (exact spike grid mx_fp_math.h fp4_e2m1_decode).
def _fp4_e2m1_decode_code(code: int) -> float:
    s = (code >> 3) & 1
    e = (code >> 1) & 0x3
    m = code & 0x1
    v = (m / 2.0) if e == 0 else (1.0 + m / 2.0) * (2.0 ** (e - 1))
    return -v if s else v

_FP4_GRID = torch.tensor([_fp4_e2m1_decode_code(c) for c in range(16)], dtype=torch.float32)
A_code_t = (A.unsqueeze(-1) - _FP4_GRID).abs().argmin(dim=-1).to(torch.int64)  # (M,K) nearest code
A_fp = _FP4_GRID[A_code_t]                                 # (M,K) decoded == spike decode
A_codes = A_code_t.tolist()
A_in_hw = _a_indices_to_hw_layout(A_codes, A_TILE_M, K_TILE)  # [M/2][K] nibble-packed HW-tiled

# B: FP8 E4M3 DIRECT single, quantized against the EXACT spike grid (fp8_e4m3_decode).
def _fp8_e4m3_decode_code(code: int) -> float:
    if (code & 0x7F) == 0x7F:
        return float("nan")
    s = (code >> 7) & 1
    e = (code >> 3) & 0xF
    m = code & 0x7
    bias = 7
    val = (m / 8.0) * (2.0 ** (1 - bias)) if e == 0 else (1.0 + m / 8.0) * (2.0 ** (e - bias))
    return -val if s else val

_E4M3_CODES = [c for c in range(256) if (c & 0x7F) != 0x7F]
_E4M3_GRID = torch.tensor([_fp8_e4m3_decode_code(c) for c in _E4M3_CODES], dtype=torch.float32)
_b_idx = (B.unsqueeze(-1) - _E4M3_GRID).abs().argmin(dim=-1)   # (K,N) nearest finite-grid index
B_fp = _E4M3_GRID[_b_idx]                                      # (K,N) == spike decode
B_in = [[_E4M3_CODES[j] for j in row] for row in _b_idx.tolist()]  # [K][N] direct 8-bit codes

# ---- Scales (power-of-two, fpe8m0) -----------------------------------------
torch.manual_seed(SEED + 123)
A_scale_exp = torch.randint(-4, 4, (M, Gk), device=DEV)
B_scale_exp = torch.randint(-4, 4, (Gk, N), device=DEV)
sq = make_fp_quantizer(SCALE_SPEC, "nearest")
A_scales_row_q = sq(torch.pow(2.0, A_scale_exp.to(torch.float32)))   # (M, Gk)
B_scales_col_q = sq(torch.pow(2.0, B_scale_exp.to(torch.float32)))   # (Gk, N)

# ---- Golden ----------------------------------------------------------------
import fp8_matmul_model
from fp8_matmul_model import tiled_matmul_hwlike

fp8_matmul_model.INPUT_SPEC = ACT_SPEC
PROD_PRECISION = [(4, 3)] * 16
ACC_PRECISION = [(4, 4)] * 8 + [(4, 5)] * 2 + [(4, 6)] * 5 + [(8, 7)] * 1

C_out_bf16 = tiled_matmul_hwlike(
    A_fp, B_fp, A_scales_row_q, B_scales_col_q,
    verbose=False, prod_precision_list=PROD_PRECISION, acc_precision_list=ACC_PRECISION,
)

# ---- Encode header arrays --------------------------------------------------
As_codes, As_bits = tensor_to_custom_fp_codes(A_scales_row_q.transpose(0, 1), SCALE_SPEC)
Bs_codes, Bs_bits = tensor_to_custom_fp_codes(B_scales_col_q, SCALE_SPEC)
As_bits -= 1
Bs_bits -= 1
As_hex = codes_to_hex_rows(As_codes, As_bits)
Bs_hex = codes_to_hex_rows(Bs_codes, Bs_bits)

C_codes, C_bits = tensor_to_custom_fp_codes(C_out_bf16, "bf16")
C_hex = codes_to_hex_rows(C_codes, C_bits)

# ---- Write header ----------------------------------------------------------
guard = HEADER.upper()
for ch in [".", "/", "\\", "-"]:
    guard = guard.replace(ch, "_")


def fmt2d_u8(rows):
    return ",\n".join("    { " + ", ".join(f"0x{b:02x}" for b in row) + " }" for row in rows)


def fmt2d_hex(hex_rows):
    return ",\n".join("    { " + ", ".join(f"0x{h}" for h in row) + " }" for row in hex_rows)


with open(HEADER, "w") as f:
    f.write(f"#ifndef {guard}\n#define {guard}\n\n#include <stdint.h>\n\n")
    f.write(f"#define MATMUL_M   {M}\n#define MATMUL_K   {K}\n#define MATMUL_N   {N}\n")
    f.write(f"#define MATMUL_GK  {Gk}\n#define MATMUL_GN  {N // GROUP}\n")
    f.write(f"#define A_TILE_M   {A_TILE_M}\n#define K_TILE     {K_TILE}\n\n")

    f.write("// A (activation) = FP4 E2M1 4-bit codes, HW-tiled [M/2][K] (direct, quad 2 rows/lane).\n")
    f.write(f"static const uint8_t A_in_hw[{M // 2}][{K}] = {{\n{fmt2d_u8(A_in_hw)}\n}};\n\n")

    f.write("// B (weight) = FP8_E4M3 DIRECT 8-bit codes, row-major [K][N] (single, 1 col/lane, no LUT).\n")
    f.write(f"static const uint8_t B_in[MATMUL_K][MATMUL_N] = {{\n{fmt2d_u8(B_in)}\n}};\n\n")

    f.write(f"// Per-row per-{GROUP}-K-group activation scales in {SCALE_SPEC}\n")
    f.write(f"static const uint8_t A_scales_row[MATMUL_GK][MATMUL_M] = {{\n{fmt2d_hex(As_hex)}\n}};\n\n")

    f.write(f"// Per-col per-{GROUP}-K-group weight scales in {SCALE_SPEC}\n")
    f.write(f"static const uint8_t B_scales_col[MATMUL_GK][MATMUL_N] = {{\n{fmt2d_hex(Bs_hex)}\n}};\n\n")

    f.write("// Golden output (scaled + bf16-accumulated), bf16\n")
    f.write(f"static const uint16_t C_out_bf16[MATMUL_M][MATMUL_N] = {{\n{fmt2d_hex(C_hex)}\n}};\n\n")

    f.write(f"#endif // {guard}\n")

print(f"Wrote {HEADER}  (M={M} K={K} N={N}, Gk={Gk}, FP4 quad act x E4M3-single wei -> mode2)")
