#!/usr/bin/env python3
"""Generate the data header for the ASYMMETRIC FP8_E5M2-activation x FP6_E3M2-weight matmul (mode4).

BOTH operands are LUT formats (both deproject):
  A (activation) is FP8_E5M2 -> 4-bit LUT indices + per-row-group E5M2 codebook (8-bit codes).
  B (weight) is FP6_E3M2 -> 4-bit LUT indices + per-col-group E3M2 codebook (6-bit codes).
The two codebooks have DIFFERENT native widths (8-bit vs 6-bit), so A_lut and B_lut are packed
at their own entry widths. Output is BF16 (compared against C_out_bf16).

Golden is the bit-exact mesh model (fp8_matmul_model.tiled_matmul_hwlike): per K-chunk unscaled
accumulate with the gemmini.cc acc_e[]/acc_m[] schedule, then fpe8m0 scale, then bf16-accumulate.
"""
import os
import torch

from lut_golden_model import (
    make_lut, quantize_lut_indices, make_fp_quantizer,
    tensor_to_custom_fp_codes, codes_to_hex_rows, pack_lut_hw_words,
    _a_indices_to_hw_layout, parse_fp_spec,
)

# ---- Parameters ------------------------------------------------------------
SEED     = int(os.environ.get("ASYM_SEED", "0"))
M        = int(os.environ.get("ASYM_M", "64"))
K        = int(os.environ.get("ASYM_K", "64"))
N        = int(os.environ.get("ASYM_N", "64"))
ACT_SPEC = "fp6:e2m3"     # FP6_E2M3 activation (LUT)
WEI_SPEC = "fp8:e5m2"     # FP8_E5M2 weight (LUT)
SCALE_SPEC = "fpe8m0"
LUT_INDEX_BITS = 4
GROUP    = 32             # K-group size for scales
G        = 1              # QUANT_LUT_UPDATE_GRANULARITY: 2^G rows/cols share one LUT
A_TILE_M = 32
K_TILE   = 16
HEADER   = os.environ.get("ASYM_HEADER_PATH", "./include/matmul_data_asym_e2m3_e5m2.h")
DEV      = torch.device("cpu")

Gk = K // GROUP
assert M % A_TILE_M == 0, "M must be a multiple of A_TILE_M for the HW-tiled A layout"
assert K % K_TILE == 0 and K % GROUP == 0 and (M >> G) >= 1 and (N >> G) >= 1

# ---- Inputs ----------------------------------------------------------------
torch.manual_seed(SEED)
A = torch.randn(M, K, device=DEV, dtype=torch.float32)
B = torch.randn(K, N, device=DEV, dtype=torch.float32)

# A: FP8_E5M2 via per-row-group LUT (activation deproject).
A_luts = [make_lut(ACT_SPEC, LUT_INDEX_BITS, DEV) for _ in range(M >> G)]
A_luts_t = torch.stack(A_luts)                            # (M>>G, 16)
A_indices = torch.stack(
    [quantize_lut_indices(A_luts[i >> G], A[i, :]) for i in range(M)]
)                                                          # (M, K)
mi_full = (torch.arange(M, device=DEV) >> G).unsqueeze(1).expand(-1, K)
A_fp = A_luts_t[mi_full, A_indices]                        # (M, K) e5m2 floats
A_in_hw = _a_indices_to_hw_layout(A_indices.tolist(), A_TILE_M, K_TILE)  # [M//2][K]

# B: FP6_E3M2 via per-col-group LUT (weight deproject).
B_luts = [make_lut(WEI_SPEC, LUT_INDEX_BITS, DEV) for _ in range(N >> G)]
B_luts_t = torch.stack(B_luts)                            # (N>>G, 16)
B_indices = torch.stack(
    [quantize_lut_indices(B_luts[j >> G], B[:, j]) for j in range(N)], dim=1
)                                                          # (K, N)
ni_full = (torch.arange(N, device=DEV) >> G).unsqueeze(0).expand(K, -1)
B_fp = B_luts_t[ni_full, B_indices]                        # (K, N) e3m2 floats

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
# B_in: E3M2 LUT indices, nibble-packed [K][N/2] (odd col high nibble)
B_idx_l = B_indices.tolist()
B_hw = [[((r[i + 1] << 4) | r[i]) for i in range(0, len(r), 2)] for r in B_idx_l]

# Codebook code values (E5M2 codes are 8-bit; E3M2 codes are 6-bit).
A_lut_codes, _ = tensor_to_custom_fp_codes(A_luts_t, ACT_SPEC)       # (M>>G, 16), 8-bit codes
B_lut_codes, _ = tensor_to_custom_fp_codes(B_luts_t, WEI_SPEC)       # (N>>G, 16), 6-bit codes
# Both LUTs are LOADED at 8-bit entries: the RTL weight LUT stores 8-bit slots and slices the low 6
# for E3M2 at deproject time, so each 6-bit E3M2 code is zero-extended into an 8-bit slot (high 2 = 0).
# Spike's fp6_e3m2_decode also reads only the low 6 bits, so both agree.
_LUT_LOAD_BITS = max(1 + sum(parse_fp_spec(ACT_SPEC)), 1 + sum(parse_fp_spec(WEI_SPEC)))
A_LUT_ENTRY_BITS = _LUT_LOAD_BITS  # both LUTs loaded at the wider native width (8 if any 8-bit fmt, else 6)
B_LUT_ENTRY_BITS = _LUT_LOAD_BITS
A_LUT_WORDS = (16 * A_LUT_ENTRY_BITS + 31) // 32                     # 4
B_LUT_WORDS = (16 * B_LUT_ENTRY_BITS + 31) // 32                     # 4

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


def fmt_lut_packed(lut_codes, entry_bits):
    lines = []
    for grp in lut_codes:
        w = pack_lut_hw_words(grp, entry_bits)
        lines.append("    { " + ", ".join(f"0x{x:08x}" for x in w) + " }")
    return ",\n".join(lines)


n_a = M >> G
n_b = N >> G
A_lut_packed = fmt_lut_packed(A_lut_codes, A_LUT_ENTRY_BITS)   # activation codebook (loaded, real, 8-bit)
B_lut_packed = fmt_lut_packed(B_lut_codes, B_LUT_ENTRY_BITS)   # weight codebook (loaded, real, 6-bit)
# C_lut is unused (output is bf16); emit an 8-bit placeholder so the .c C_lut load (8-bit) compiles.
C_lut_packed = A_lut_packed

with open(HEADER, "w") as f:
    f.write(f"#ifndef {guard}\n#define {guard}\n\n#include <stdint.h>\n\n")
    f.write(f"#define MATMUL_M   {M}\n#define MATMUL_K   {K}\n#define MATMUL_N   {N}\n")
    f.write(f"#define MATMUL_GK  {Gk}\n#define MATMUL_GN  {N // GROUP}\n")
    f.write(f"#define A_TILE_M   {A_TILE_M}\n#define K_TILE     {K_TILE}\n\n")

    f.write("// A (activation) = FP6_E2M3 4-bit LUT indices, HW-tiled [M/2][K].\n")
    f.write("//   byte layout: bits[7:4]=a(2r+1,k), bits[3:0]=a(2r,k) for hw-row r\n")
    f.write(f"static const uint8_t A_in_hw[{M // 2}][{K}] = {{\n{fmt2d_u8(A_in_hw)}\n}};\n\n")

    f.write("// B (weight) = FP8_E5M2 4-bit LUT indices, nibble-packed [K][N/2] (odd col high nibble)\n")
    f.write(f"static const uint8_t B_in[MATMUL_K][MATMUL_N / 2] = {{\n{fmt2d_u8(B_hw)}\n}};\n\n")

    f.write(f"// A_lut = E5M2 codebook (16x{A_LUT_ENTRY_BITS}-bit -> {A_LUT_WORDS}x uint32). Loaded (sel=1).\n")
    f.write(f"static const uint32_t A_lut[{n_a}][{A_LUT_WORDS}] = {{\n{A_lut_packed}\n}};\n\n")
    f.write(f"// B_lut = E3M2 codebook, 6-bit codes zero-extended to {B_LUT_ENTRY_BITS}-bit slots "
            f"(RTL slices low 6) -> {B_LUT_WORDS}x uint32. Loaded (sel=0).\n")
    f.write(f"static const uint32_t B_lut[{n_b}][{B_LUT_WORDS}] = {{\n{B_lut_packed}\n}};\n\n")
    f.write(f"// C_lut placeholder (output is bf16, never read); 8-bit like A_lut.\n")
    f.write(f"static const uint32_t C_lut[{n_a}][{A_LUT_WORDS}] = {{\n{C_lut_packed}\n}};\n\n")

    f.write(f"// Per-row per-{GROUP}-K-group activation scales in {SCALE_SPEC}\n")
    f.write(f"static const uint8_t A_scales_row[MATMUL_GK][MATMUL_M] = {{\n{fmt2d_hex(As_hex)}\n}};\n\n")

    f.write(f"// Per-col per-{GROUP}-K-group weight scales in {SCALE_SPEC}\n")
    f.write(f"static const uint8_t B_scales_col[MATMUL_GK][MATMUL_N] = {{\n{fmt2d_hex(Bs_hex)}\n}};\n\n")

    f.write("// Golden output (scaled + bf16-accumulated), bf16\n")
    f.write(f"static const uint16_t C_out_bf16[MATMUL_M][MATMUL_N] = {{\n{fmt2d_hex(C_hex)}\n}};\n\n")

    f.write(f"#endif // {guard}\n")

print(f"Wrote {HEADER}  (M={M} K={K} N={N}, Gk={Gk}, A LUTs={n_a}, B LUTs={n_b}, both loaded 8b)")
