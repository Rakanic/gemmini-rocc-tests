#!/usr/bin/env python3
"""ASYMMETRIC FP8_E4M3-SINGLE-activation x FP6_E3M2-weight matmul (mode7, dual throughput 16x32).

E4M3 activation is DIRECT (single, 1 row/lane, no act LUT); E3M2 weight is quad (LUT-deprojected,
2 col/lane). => 2 products/PE, 16x32 output tile. Spike selects single-act via mx_lut_a_loaded=0
(no activation LUT) even though a WEIGHT LUT is loaded (mx_lut_b_loaded=1).
  A (activation) = FP8_E4M3 DIRECT 8-bit codes, row-major [M][K] (exact spike grid). TM=16.
  B (weight)     = FP6_E3M2 4-bit LUT indices, nibble-packed [K][N/2] + per-col-group codebook. TN=32.
Output is BF16. Golden is the throughput-agnostic mesh model (fp8_matmul_model.tiled_matmul_hwlike).
"""
import os
import torch

from lut_golden_model import (
    make_lut, quantize_lut_indices, make_fp_quantizer,
    tensor_to_custom_fp_codes, codes_to_hex_rows, pack_lut_hw_words,
)

# ---- Parameters ------------------------------------------------------------
SEED     = int(os.environ.get("ASYM_SEED", "0"))
M        = int(os.environ.get("ASYM_M", "64"))
K        = int(os.environ.get("ASYM_K", "64"))
N        = int(os.environ.get("ASYM_N", "64"))
ACT_SPEC = "fp8:e4m3"     # FP8 E4M3 activation, DIRECT single (no LUT)
WEI_SPEC = "fp6:e3m2"     # FP6_E3M2 weight (LUT)
SCALE_SPEC = "fpe8m0"
LUT_INDEX_BITS = 4
GROUP    = 32
G        = 1              # QUANT_LUT_UPDATE_GRANULARITY
HEADER   = os.environ.get("ASYM_HEADER_PATH", "./include/matmul_data_asym_e4m3s_e3m2.h")
DEV      = torch.device("cpu")

Gk = K // GROUP
assert K % 16 == 0 and K % GROUP == 0 and (N >> G) >= 1

# ---- Inputs ----------------------------------------------------------------
torch.manual_seed(SEED)
A = torch.randn(M, K, device=DEV, dtype=torch.float32)
B = torch.randn(K, N, device=DEV, dtype=torch.float32)

# A: FP8 E4M3 DIRECT, quantized against the EXACT spike grid (mx_fp_math.h fp8_e4m3_decode).
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
_a_idx = (A.unsqueeze(-1) - _E4M3_GRID).abs().argmin(dim=-1)
A_fp = _E4M3_GRID[_a_idx]                                     # (M, K) == spike decode
A_in = [[_E4M3_CODES[j] for j in row] for row in _a_idx.tolist()]  # [M][K] direct 8-bit codes

# B: FP6_E3M2 via per-col-group LUT (weight deproject).
B_luts = [make_lut(WEI_SPEC, LUT_INDEX_BITS, DEV) for _ in range(N >> G)]
B_luts_t = torch.stack(B_luts)                            # (N>>G, 16)
B_indices = torch.stack(
    [quantize_lut_indices(B_luts[j >> G], B[:, j]) for j in range(N)], dim=1
)                                                          # (K, N)
ni_full = (torch.arange(N, device=DEV) >> G).unsqueeze(0).expand(K, -1)
B_fp = B_luts_t[ni_full, B_indices]                       # (K, N) e3m2 floats

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
B_idx_l = B_indices.tolist()
B_hw = [[((r[i + 1] << 4) | r[i]) for i in range(0, len(r), 2)] for r in B_idx_l]

B_lut_codes, _ = tensor_to_custom_fp_codes(B_luts_t, WEI_SPEC)       # (N>>G, 16), 6-bit codes
B_LUT_ENTRY_BITS = 8   # 6-bit E3M2 codes zero-extended to 8-bit slots (RTL slices low 6)
B_LUT_WORDS = (16 * B_LUT_ENTRY_BITS + 31) // 32

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


def fmt_lut_packed(lut_codes, entry_bits):
    return ",\n".join("    { " + ", ".join(f"0x{x:08x}" for x in pack_lut_hw_words(grp, entry_bits)) + " }"
                      for grp in lut_codes)


n_b = N >> G
B_lut_packed = fmt_lut_packed(B_lut_codes, B_LUT_ENTRY_BITS)

with open(HEADER, "w") as f:
    f.write(f"#ifndef {guard}\n#define {guard}\n\n#include <stdint.h>\n\n")
    f.write(f"#define MATMUL_M   {M}\n#define MATMUL_K   {K}\n#define MATMUL_N   {N}\n")
    f.write(f"#define MATMUL_GK  {Gk}\n#define MATMUL_GN  {N // GROUP}\n\n")

    f.write("// A (activation) = FP8_E4M3 DIRECT 8-bit codes, row-major [M][K] (single throughput, no LUT).\n")
    f.write(f"static const uint8_t A_in[MATMUL_M][MATMUL_K] = {{\n{fmt2d_u8(A_in)}\n}};\n\n")

    f.write("// B (weight) = FP6_E3M2 4-bit LUT indices, nibble-packed [K][N/2] (odd col high nibble)\n")
    f.write(f"static const uint8_t B_in[MATMUL_K][MATMUL_N / 2] = {{\n{fmt2d_u8(B_hw)}\n}};\n\n")

    f.write(f"// B_lut = E3M2 codebook, 6-bit codes zero-extended to {B_LUT_ENTRY_BITS}-bit slots -> "
            f"{B_LUT_WORDS}x uint32. Loaded (sel=0). No A_lut/C_lut (act direct, output bf16).\n")
    f.write(f"static const uint32_t B_lut[{n_b}][{B_LUT_WORDS}] = {{\n{B_lut_packed}\n}};\n\n")

    f.write(f"// Per-row per-{GROUP}-K-group activation scales in {SCALE_SPEC}\n")
    f.write(f"static const uint8_t A_scales_row[MATMUL_GK][MATMUL_M] = {{\n{fmt2d_hex(As_hex)}\n}};\n\n")

    f.write(f"// Per-col per-{GROUP}-K-group weight scales in {SCALE_SPEC}\n")
    f.write(f"static const uint8_t B_scales_col[MATMUL_GK][MATMUL_N] = {{\n{fmt2d_hex(Bs_hex)}\n}};\n\n")

    f.write("// Golden output (scaled + bf16-accumulated), bf16\n")
    f.write(f"static const uint16_t C_out_bf16[MATMUL_M][MATMUL_N] = {{\n{fmt2d_hex(C_hex)}\n}};\n\n")

    f.write(f"#endif // {guard}\n")

print(f"Wrote {HEADER}  (M={M} K={K} N={N}, Gk={Gk}, E4M3-single act x E3M2 quad wei -> mode7)")
