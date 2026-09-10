#!/usr/bin/env python3
"""Generate the data header for the ASYMMETRIC FP8_E4M3-SINGLE-activation x FP4-weight matmul (mode6).

This is the "single x quad" dual-throughput case: the E4M3 activation is used WITHOUT the LUT
(single throughput, direct 8-bit, 1 row/lane) while the FP4 weight is quad (2 col/lane). Per PE this
produces 2 products (a rectangular 16x32 output tile), between single (1) and quad (4).

  A (activation) = FP8_E4M3 DIRECT 8-bit codes, row-major [M][K] (no LUT, mx_lut_en=0). TM=16.
  B (weight)     = FP4 E2M1 DIRECT 4-bit codes, nibble-packed [K][N/2] (no LUT).             TN=32.
Output is BF16 (compared against C_out_bf16).

Golden is the bit-exact mesh model (fp8_matmul_model.tiled_matmul_hwlike) -- throughput-agnostic,
identical to the symmetric/quad path (it is just an MxN matmul of decoded floats + fpe8m0 scales).
"""
import os
import torch

from lut_golden_model import (
    make_fp_quantizer, tensor_to_custom_fp_codes, codes_to_hex_rows,
)

# ---- Parameters ------------------------------------------------------------
SEED     = int(os.environ.get("ASYM_SEED", "0"))
M        = int(os.environ.get("ASYM_M", "64"))
K        = int(os.environ.get("ASYM_K", "64"))
N        = int(os.environ.get("ASYM_N", "64"))
ACT_SPEC = "fp8:e4m3"     # FP8 E4M3 activation, DIRECT single (no LUT)
WEI_SPEC = "fp4:e2m1"     # FP4 E2M1 weight (direct)
SCALE_SPEC = "fpe8m0"
GROUP    = 32             # K-group size for scales
HEADER   = os.environ.get("ASYM_HEADER_PATH", "./include/matmul_data_asym_e4m3s_fp4.h")
DEV      = torch.device("cpu")

Gk = K // GROUP
assert K % 16 == 0 and K % GROUP == 0

# ---- Inputs ----------------------------------------------------------------
torch.manual_seed(SEED)
A = torch.randn(M, K, device=DEV, dtype=torch.float32)
B = torch.randn(K, N, device=DEV, dtype=torch.float32)

# A: FP8 E4M3 DIRECT (all 256 codes, single throughput -- no LUT). Quantize against the EXACT spike
# grid (mx_fp_math.h fp8_e4m3_decode) so A_fp == what spike decodes byte-for-byte. qtorch's generic
# float_quantize does NOT implement E4M3's special top-of-range/NaN codes, which the direct path (unlike
# the 16-entry LUT path) can reach -- so build the grid from spike's decode directly.
def _fp8_e4m3_decode_code(code: int) -> float:
    if (code & 0x7F) == 0x7F:
        return float("nan")
    s = (code >> 7) & 1
    e = (code >> 3) & 0xF
    m = code & 0x7
    bias = 7
    val = (m / 8.0) * (2.0 ** (1 - bias)) if e == 0 else (1.0 + m / 8.0) * (2.0 ** (e - bias))
    return -val if s else val

_E4M3_CODES = [c for c in range(256) if (c & 0x7F) != 0x7F]   # finite codes (exclude NaN 0x7F/0xFF)
_E4M3_GRID = torch.tensor([_fp8_e4m3_decode_code(c) for c in _E4M3_CODES], dtype=torch.float32)  # (254,)
_a_idx = (A.unsqueeze(-1) - _E4M3_GRID).abs().argmin(dim=-1)   # (M, K) nearest finite-grid index
A_fp = _E4M3_GRID[_a_idx]                                      # (M, K) decoded floats == spike's decode
A_in = [[_E4M3_CODES[j] for j in row] for row in _a_idx.tolist()]  # [M][K] direct 8-bit codes

# B: FP4 E2M1 direct. Encode with the EXACT spike grid (mx_fp_math.h fp4_e2m1_decode).
def _fp4_e2m1_decode_code(code: int) -> float:
    s = (code >> 3) & 1
    e = (code >> 1) & 0x3
    m = code & 0x1
    v = (m / 2.0) if e == 0 else (1.0 + m / 2.0) * (2.0 ** (e - 1))
    return -v if s else v

_FP4_GRID = torch.tensor([_fp4_e2m1_decode_code(c) for c in range(16)], dtype=torch.float32)  # (16,)
B_code_t = (B.unsqueeze(-1) - _FP4_GRID).abs().argmin(dim=-1).to(torch.int64)  # (K,N) nearest code
B_fp = _FP4_GRID[B_code_t]                                # (K, N) decoded floats == spike's decode
B_codes = B_code_t.tolist()                               # List[K][N], 4-bit codes

# ---- Scales (power-of-two, fpe8m0) -----------------------------------------
torch.manual_seed(SEED + 123)
A_scale_exp = torch.randint(-4, 4, (M, Gk), device=DEV)
B_scale_exp = torch.randint(-4, 4, (Gk, N), device=DEV)
sq = make_fp_quantizer(SCALE_SPEC, "nearest")
A_scales_row_q = sq(torch.pow(2.0, A_scale_exp.to(torch.float32)))   # (M, Gk)
B_scales_col_q = sq(torch.pow(2.0, B_scale_exp.to(torch.float32)))   # (Gk, N)

# ---- Golden: bit-exact mesh model (matches gemmini.cc / spike) --------------
import fp8_matmul_model
from fp8_matmul_model import tiled_matmul_hwlike

fp8_matmul_model.INPUT_SPEC = ACT_SPEC   # only used for verbose prints
PROD_PRECISION = [(4, 3)] * 16
ACC_PRECISION = [(4, 4)] * 8 + [(4, 5)] * 2 + [(4, 6)] * 5 + [(8, 7)] * 1

C_out_bf16 = tiled_matmul_hwlike(
    A_fp, B_fp,
    A_scales_row_q,                          # (M, Gk) decoded scale floats
    B_scales_col_q,                          # (Gk, N) decoded scale floats
    verbose=False,
    prod_precision_list=PROD_PRECISION,
    acc_precision_list=ACC_PRECISION,
)

# ---- Encode header arrays --------------------------------------------------
# B_in: FP4 direct codes, standard nibble-packed [K][N/2] (odd col high nibble)
B_hw = [[((r[i + 1] << 4) | r[i]) for i in range(0, len(r), 2)] for r in B_codes]

# Scales: unsigned fpe8m0 codes (drop sign bit)
As_codes, As_bits = tensor_to_custom_fp_codes(A_scales_row_q.transpose(0, 1), SCALE_SPEC)  # (Gk, M)
Bs_codes, Bs_bits = tensor_to_custom_fp_codes(B_scales_col_q, SCALE_SPEC)                  # (Gk, N)
As_bits -= 1
Bs_bits -= 1
As_hex = codes_to_hex_rows(As_codes, As_bits)
Bs_hex = codes_to_hex_rows(Bs_codes, Bs_bits)

# Output bf16 codes
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
    f.write(f"#define MATMUL_GK  {Gk}\n#define MATMUL_GN  {N // GROUP}\n\n")

    f.write("// A (activation) = FP8_E4M3 DIRECT 8-bit codes, row-major [M][K] (single throughput, no LUT).\n")
    f.write(f"static const uint8_t A_in[MATMUL_M][MATMUL_K] = {{\n{fmt2d_u8(A_in)}\n}};\n\n")

    f.write("// B (weight) = FP4 E2M1 DIRECT 4-bit codes, nibble-packed [K][N/2] (odd col high nibble)\n")
    f.write(f"static const uint8_t B_in[MATMUL_K][MATMUL_N / 2] = {{\n{fmt2d_u8(B_hw)}\n}};\n\n")

    f.write(f"// Per-row per-{GROUP}-K-group activation scales in {SCALE_SPEC}\n")
    f.write(f"static const uint8_t A_scales_row[MATMUL_GK][MATMUL_M] = {{\n{fmt2d_hex(As_hex)}\n}};\n\n")

    f.write(f"// Per-col per-{GROUP}-K-group weight scales in {SCALE_SPEC}\n")
    f.write(f"static const uint8_t B_scales_col[MATMUL_GK][MATMUL_N] = {{\n{fmt2d_hex(Bs_hex)}\n}};\n\n")

    f.write("// Golden output (scaled + bf16-accumulated), bf16\n")
    f.write(f"static const uint16_t C_out_bf16[MATMUL_M][MATMUL_N] = {{\n{fmt2d_hex(C_hex)}\n}};\n\n")

    f.write(f"#endif // {guard}\n")

print(f"Wrote {HEADER}  (M={M} K={K} N={N}, Gk={Gk}, E4M3-single act x FP4 quad wei -> mode6)")
