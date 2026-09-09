#!/usr/bin/env python3
"""Generate the data header for the ASYMMETRIC FP4-activation x FP6_E3M2-weight matmul test.

A (activation) is FP4 E2M1 stored as DIRECT 4-bit codes (fed straight to the mesh, no LUT).
B (weight) is FP6_E3M2 stored as 4-bit LUT indices + a per-group fp6 codebook (LUT-deprojected).
Output is BF16 (compared against C_out_bf16).

Golden arithmetic mirrors lut_mapping_demo.py (the passing symmetric fp6 test):
  A_fp_scaled = A_fp * A_scales ; B_fp_scaled = B_fp * B_scales ;
  C_out_bf16  = matmul_outer_quantized_hwlike(A_fp_scaled, B_fp_scaled)   # bf16 accumulate

Header layout matches include/matmul_data_mx_lut_hw.h so the test .c can mirror
matmul_tiled_fp6_128x128.c with only the A operand + act format changed. A_lut/C_lut are
emitted (a fp6 codebook) only so both compile branches build; the HW never reads A's LUT
because fp4 activations take the direct path.
"""
import os
import torch

from lut_golden_model import (
    make_lut, quantize_lut_indices, make_fp_quantizer,
    tensor_to_custom_fp_codes, codes_to_hex_rows, pack_lut_hw_words,
    _a_indices_to_hw_layout, parse_fp_spec,
    tiled_matmul_scaled_accum_hwlike,
)

# ---- Parameters ------------------------------------------------------------
SEED     = int(os.environ.get("ASYM_SEED", "0"))
M        = int(os.environ.get("ASYM_M", "64"))
K        = int(os.environ.get("ASYM_K", "64"))
N        = int(os.environ.get("ASYM_N", "64"))
ACT_SPEC = "fp4:e4m1"     # FP4 E2M1 (exp2, man1) activation
WEI_SPEC = "fp6:e3m2"     # FP6 E3M2 weight
SCALE_SPEC = "fpe8m0"
LUT_INDEX_BITS = 4
GROUP    = 32             # K-group size for scales
G        = 1              # QUANT_LUT_UPDATE_GRANULARITY: 2^G cols share one weight LUT
A_TILE_M = 32
K_TILE   = 16
HEADER   = os.environ.get("ASYM_HEADER_PATH", "./include/matmul_data_asym_fp4_fp6.h")
DEV      = torch.device("cpu")

Gk = K // GROUP
assert M % A_TILE_M == 0, "M must be a multiple of A_TILE_M for the HW-tiled A layout"
assert K % K_TILE == 0 and K % GROUP == 0 and (N >> G) >= 1

# ---- Inputs ----------------------------------------------------------------
torch.manual_seed(SEED)
A = torch.randn(M, K, device=DEV, dtype=torch.float32)
B = torch.randn(K, N, device=DEV, dtype=torch.float32)

# A: FP4 E2M1 direct. Encode with the EXACT spike grid (mx_fp_math.h fp4_e2m1_decode:
#   e==0 -> m/2 (subnormal 0/0.5); else (1+m/2)*2^(e-1); sign in bit 3), and take A_fp as the
# decoded value so the golden matches what spike decodes from the emitted codes.
def _fp4_e2m1_decode_code(code: int) -> float:
    s = (code >> 3) & 1
    e = (code >> 1) & 0x3
    m = code & 0x1
    v = (m / 2.0) if e == 0 else (1.0 + m / 2.0) * (2.0 ** (e - 1))
    return -v if s else v

_FP4_GRID = torch.tensor([_fp4_e2m1_decode_code(c) for c in range(16)], dtype=torch.float32)  # (16,)
A_code_t = (A.unsqueeze(-1) - _FP4_GRID).abs().argmin(dim=-1).to(torch.int64)  # (M,K) nearest code
A_fp = _FP4_GRID[A_code_t]                                 # (M,K) decoded floats == spike's decode
A_codes = A_code_t.tolist()                                # List[M][K], 4-bit codes
A_in_hw = _a_indices_to_hw_layout(A_codes, A_TILE_M, K_TILE)  # [M//2][K]

# B: FP6_E3M2 via per-col-group LUT.
B_luts = [make_lut(WEI_SPEC, LUT_INDEX_BITS, DEV) for _ in range(N >> G)]
B_luts_t = torch.stack(B_luts)                            # (N>>G, 16)
B_indices = torch.stack(
    [quantize_lut_indices(B_luts[j >> G], B[:, j]) for j in range(N)], dim=1
)                                                          # (K, N)
ni_full = (torch.arange(N, device=DEV) >> G).unsqueeze(0).expand(K, -1)
B_fp = B_luts_t[ni_full, B_indices]                        # (K, N) fp6 floats

# ---- Scales (power-of-two, fpe8m0) -----------------------------------------
torch.manual_seed(SEED + 123)
A_scale_exp = torch.randint(-4, 4, (M, Gk), device=DEV)
B_scale_exp = torch.randint(-4, 4, (Gk, N), device=DEV)
sq = make_fp_quantizer(SCALE_SPEC, "nearest")
A_scales_row_q = sq(torch.pow(2.0, A_scale_exp.to(torch.float32)))   # (M, Gk)
B_scales_col_q = sq(torch.pow(2.0, B_scale_exp.to(torch.float32)))   # (Gk, N)

# ---- Golden: bit-exact mesh model (matches gemmini.cc / spike) --------------
# tiled_matmul_hwlike accumulates UNSCALED e4m3-truncated products with the per-position
# accumulator precision schedule, then scales the tile and bf16-accumulates. prod/acc lists
# are the exact acc_e[]/acc_m[] schedule from gemmini.cc.
import fp8_matmul_model
from fp8_matmul_model import tiled_matmul_hwlike

fp8_matmul_model.INPUT_SPEC = WEI_SPEC   # only used for verbose prints
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
# B_in: standard nibble-packed [K][N/2], odd-col index in high nibble
B_idx_l = B_indices.tolist()
B_hw = [[((r[i + 1] << 4) | r[i]) for i in range(0, len(r), 2)] for r in B_idx_l]

# Weight codebook -> 6-bit HW-packed words (16 entries -> 3 uint32)
B_lut_codes, _ = tensor_to_custom_fp_codes(B_luts_t, WEI_SPEC)       # (N>>G, 16)
LUT_ENTRY_BITS = 1 + sum(parse_fp_spec(WEI_SPEC))                     # fp6 -> 6
LUT_WORDS = (16 * LUT_ENTRY_BITS + 31) // 32                          # 3

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


def fmt_lut_packed(lut_codes):
    lines = []
    for grp in lut_codes:
        w = pack_lut_hw_words(grp, LUT_ENTRY_BITS)
        lines.append("    { " + ", ".join(f"0x{x:08x}" for x in w) + " }")
    return ",\n".join(lines)


n_lut = N >> G
B_lut_packed = fmt_lut_packed(B_lut_codes)   # weight codebook (loaded)
# A_lut / C_lut are unused by the HW here (A is direct, output is bf16) but must exist so both
# compile branches build. Reuse the weight codebook as a harmless placeholder.
placeholder_lut = B_lut_packed

with open(HEADER, "w") as f:
    f.write(f"#ifndef {guard}\n#define {guard}\n\n#include <stdint.h>\n\n")
    f.write(f"#define MATMUL_M   {M}\n#define MATMUL_K   {K}\n#define MATMUL_N   {N}\n")
    f.write(f"#define MATMUL_GK  {Gk}\n#define MATMUL_GN  {N // GROUP}\n")
    f.write(f"#define A_TILE_M   {A_TILE_M}\n#define K_TILE     {K_TILE}\n\n")

    f.write("// A (activation) = FP4 E2M1 DIRECT 4-bit codes, HW-tiled [M/2][K].\n")
    f.write("//   byte layout: bits[7:4]=a(2r+1,k), bits[3:0]=a(2r,k) for hw-row r\n")
    f.write(f"static const uint8_t A_in_hw[{M // 2}][{K}] = {{\n{fmt2d_u8(A_in_hw)}\n}};\n\n")

    f.write("// B (weight) = FP6_E3M2 4-bit LUT indices, nibble-packed [K][N/2] (odd col high nibble)\n")
    f.write(f"static const uint8_t B_in[MATMUL_K][MATMUL_N / 2] = {{\n{fmt2d_u8(B_hw)}\n}};\n\n")

    f.write(f"// LUTs: 16x{LUT_ENTRY_BITS}-bit entries -> {LUT_WORDS}x uint32 per group. Only B_lut is used\n")
    f.write("// (weight deproject); A_lut/C_lut are placeholders so both build paths compile.\n")
    f.write(f"static const uint32_t A_lut[{n_lut}][{LUT_WORDS}] = {{\n{placeholder_lut}\n}};\n\n")
    f.write(f"static const uint32_t B_lut[{n_lut}][{LUT_WORDS}] = {{\n{B_lut_packed}\n}};\n\n")
    f.write(f"static const uint32_t C_lut[{n_lut}][{LUT_WORDS}] = {{\n{placeholder_lut}\n}};\n\n")

    f.write(f"// Per-row per-{GROUP}-K-group activation scales in {SCALE_SPEC}\n")
    f.write(f"static const uint8_t A_scales_row[MATMUL_GK][MATMUL_M] = {{\n{fmt2d_hex(As_hex)}\n}};\n\n")

    f.write(f"// Per-col per-{GROUP}-K-group weight scales in {SCALE_SPEC}\n")
    f.write(f"static const uint8_t B_scales_col[MATMUL_GK][MATMUL_N] = {{\n{fmt2d_hex(Bs_hex)}\n}};\n\n")

    f.write("// Golden output (scaled + bf16-accumulated), bf16\n")
    f.write(f"static const uint16_t C_out_bf16[MATMUL_M][MATMUL_N] = {{\n{fmt2d_hex(C_hex)}\n}};\n\n")

    f.write(f"#endif // {guard}\n")

print(f"Wrote {HEADER}  (M={M} K={K} N={N}, Gk={Gk}, weight LUTs={n_lut})")
