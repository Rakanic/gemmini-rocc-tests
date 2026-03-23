#!/usr/bin/env python3
"""
LUT Mapping Demo
================
Shows how every 4-bit nibble in A_in/B_in is used as an index into A_lut/B_lut
to recover the actual float values.

Packing convention (from write_c_header_tiled in golden_model.py, line 729):
  A_codes = [[(r[i+1] << 4) | r[i] for i in range(0, len(r), 2)] for r in A_codes]

So each byte stores two indices:
  - bits [3:0] (low nibble)  -> element at column c
  - bits [7:4] (high nibble) -> element at column c+1
"""

import sys
import random
import numpy as np
import torch

sys.path.insert(0, ".")
from fp_decoder import ieee_to_recfn
from fp_decoder import recfn_to_ieee
import fp8_matmul_model
from fp8_matmul_model import matmul_outer_quantized_hwlike, compute_tile_scale_matrix, tiled_matmul_hwlike
from golden_model import (
    make_lut,
    quantize_lut_indices,
    lut_lookup,
    tensor_to_custom_fp_codes,
    codes_to_hex_rows,
    make_fp_quantizer,
    tiled_matmul_scaled_accum_hwlike,
    q_bf16_rne,
    hw_add_bf16,
    parse_fp_spec,
    write_c_header_tiled_hw,
    compute_tile_scale_matrix_fpe8m0,
    _a_indices_to_hw_layout,
)

def bf16_tensor_to_recfn_hex(t: torch.Tensor) -> list:
    """Convert float32 tensor (on BF16 grid) to HardFloat recFN hex strings (17-bit, 5 hex digits)."""
    raw = t.to(torch.bfloat16).view(torch.uint8)
    # reconstruct 16-bit IEEE BF16 values (little-endian pairs)
    rows = []
    flat = t.to(torch.bfloat16).reshape(-1)
    import struct
    for val in flat:
        ieee = struct.unpack('<H', struct.pack('<e', float(val.to(torch.float32))))[0]
        # bfloat16 raw bits via view
        buf = val.view(torch.int16).item() & 0xFFFF
        rec = ieee_to_recfn(buf, exp_bits=8, mant_bits_total=8)
        rows.append(f"{rec:05x}")
    shape = t.shape
    result, idx = [], 0
    for i in range(shape[0]):
        result.append([rows[idx + j] for j in range(shape[1])])
        idx += shape[1]
    return result

# ── Parameters (edit to match your run) ───────────────────────────────────────
SEED           = 0
M, K, N        = 128, 128, 128
INPUT_SPEC     = "fp6:e3m2"
LUT_INDEX_BITS = 4          # 2^4 = 16-entry LUT
DEV            = torch.device("cpu")
PROD_SPEC      = INPUT_SPEC  # product quantization spec (match hardware)
SCALE_SPEC     = "fpe8m0"    # scale factor format
GROUP          = 32          # K-group size for scales
TILE           = 16          # systolic tile size
QUANT_LUT_UPDATE_GRANULARITY = 1  # every 2^g rows of A / cols of B share one LUT
Gk             = K // GROUP
DEBUG_R        = 4             # print first DEBUG_R x DEBUG_R elements in debug blocks
# ──────────────────────────────────────────────────────────────────────────────


# ── Reproduce A, B with the same seed as run_experiment ───────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
A = torch.randn(M, K, device=DEV, dtype=torch.float32)
B = torch.randn(K, N, device=DEV, dtype=torch.float32)


# ── Build LUTs (from matmul_data_mx_fp6.h) ────────────────────────────────────
LUT_SIZE = 1 << LUT_INDEX_BITS  # 16


def _decode_fp6_e3m2(code: int) -> float:
    """Decode a 6-bit FP6 E3M2 raw code (1s|3e|2m, bias=3) to float32."""
    sign = (code >> 5) & 1
    exp  = (code >> 2) & 0x7
    mant = code & 0x3
    bias = 3
    if exp == 0:
        val = (mant / 4.0) * (2.0 ** (1 - bias))   # subnormal
    else:
        val = (1.0 + mant / 4.0) * (2.0 ** (exp - bias))
    return -val if sign else val


# ── Hardware-faithful FP6 nearest-LUT finder (mirrors FP6E3M2NearestFinder.scala) ──

def _raw_fp6e3m2_from_bits(val: int) -> dict:
    """Parse a 6-bit FP6 E3M2 value into raw components (mirrors rawFP6E3M2FromBits)."""
    val = val & 0x3F
    sign = (val >> 5) & 1
    exp  = (val >> 2) & 0b111  # bits [4:2]
    mant =  val       & 0b011  # bits [1:0]
    is_zero      = (exp == 0) and (mant == 0)
    is_subnormal = (exp == 0) and (mant != 0)
    s_exp        = -2 if exp == 0 else (exp - 3)
    implicit_bit = 0  if exp == 0 else 1
    sig = (implicit_bit << 2) | mant  # 3 bits: {implicit_bit, mant[1:0]}
    return {'sign': sign, 'is_zero': is_zero, 'is_subnormal': is_subnormal,
            's_exp': s_exp, 'sig': sig}


def fp6_to_fixed_point(val: int) -> int:
    """
    Convert a 6-bit FP6 E3M2 value to a signed fixed-point integer.
    Mirrors fp6ToFixedPoint inside FP6E3M2NearestFinder.scala.
    Fixed-point unit = 2^-4  (1.0 -> 16, 0.25 -> 4, 2.0 -> 32, …)
    """
    raw      = _raw_fp6e3m2_from_bits(val)
    shift_amt = (raw['s_exp'] + 2) & 0b111      # 3-bit uint, range [0, 6]
    shifted   = (raw['sig'] << shift_amt) & 0x1FF  # 9-bit
    magnitude = shifted & 0xFF                    # bits [7:0]
    signed    = -magnitude if raw['sign'] else magnitude
    return 0 if raw['is_zero'] else signed


def fp6e3m2_nearest_finder(in_fp6: int, in_lut: list) -> int:
    """
    Find the index of the nearest LUT entry to in_fp6.
    Mirrors FP6E3M2NearestFinder.scala.
    Tie-breaking: lower index wins (matches Mux(d1 <= d2, i1, i2)).
    """
    assert len(in_lut) == 16, "LUT must have exactly 16 entries"
    fixed_in  = fp6_to_fixed_point(in_fp6)
    fixed_lut = [fp6_to_fixed_point(v) for v in in_lut]
    diffs     = [abs(fixed_in - f) & 0x1FF for f in fixed_lut]  # 9-bit mask
    min_idx   = 0
    for i in range(1, 16):
        if diffs[i] < diffs[min_idx]:
            min_idx = i
    return min_idx

# Every 2^QUANT_LUT_UPDATE_GRANULARITY rows of A share one LUT; same for B cols
G = QUANT_LUT_UPDATE_GRANULARITY
print("[Step 1]: Generate the luts for fp6 projection")
A_luts = [make_lut(INPUT_SPEC, LUT_INDEX_BITS, DEV) for _ in range(M >> G)]  # M >> G LUTs
B_luts = [make_lut(INPUT_SPEC, LUT_INDEX_BITS, DEV) for _ in range(N >> G)]  # N >> G LUTs
C_luts = [make_lut(INPUT_SPEC, LUT_INDEX_BITS, DEV) for _ in range(M >> G)]  # M >> G LUTs

print("[Step 2]: Generate the projected data")
A_indices = torch.stack([quantize_lut_indices(A_luts[i >> G], A[i])      for i in range(M)])        # (M, K)
B_indices = torch.stack([quantize_lut_indices(B_luts[j >> G], B[:, j])   for j in range(N)], dim=1) # (K, N)
#C_indices = torch.stack([quantize_lut_indices(C_luts[j], C[:, j])   for j in range(N)], dim=1) # (K, N)

A_scale_exp = torch.randint(low=-4, high=4, size=(M, Gk), device=DEV)
B_scale_exp = torch.randint(low=-4, high=4, size=(Gk, N), device=DEV)

A_scales_row = torch.pow(2.0, A_scale_exp.to(torch.float32))
B_scales_col = torch.pow(2.0, B_scale_exp.to(torch.float32))

print("[Step 3]: Generate E8M0 scales")
A_scales_row_q = make_fp_quantizer(SCALE_SPEC, "nearest")(A_scales_row)
B_scales_col_q = make_fp_quantizer(SCALE_SPEC, "nearest")(B_scales_col)


print("[Step 4]: Generate HW like results C_out")
# Tile dimensions
A_TILE_M = 32   # A rows per tile
K_TILE   = 16   # K-dimension per tile (= systolic TILE)
B_TILE_N = 32   # B cols per tile

# Stack LUTs into tensors for efficient indexing
A_luts_t = torch.stack(A_luts)  # (M>>G, LUT_SIZE) — one LUT per 2^G A rows
B_luts_t = torch.stack(B_luts)  # (N>>G, LUT_SIZE) — one LUT per 2^G B cols

# ── Build full scaled A/B matrices and compute C_out via matmul_outer_quantized_hwlike ──
mi_full = (torch.arange(M, device=DEV) >> G).unsqueeze(1).expand(-1, K)
ni_full = (torch.arange(N, device=DEV) >> G).unsqueeze(0).expand(K, -1)
A_fp = A_luts_t[mi_full, A_indices]                                      # (M, K)
B_fp = B_luts_t[ni_full, B_indices]                                      # (K, N)
k_group_idx = torch.arange(K, device=DEV) // GROUP                       # (K,)
A_fp_scaled = A_fp * A_scales_row_q[:, k_group_idx]                      # (M, K)
B_fp_scaled = B_fp * B_scales_col_q[k_group_idx, :]                      # (K, N)

fp8_matmul_model.prod_quant = make_fp_quantizer(INPUT_SPEC, "nearest")
C_out = matmul_outer_quantized_hwlike(A_fp_scaled, B_fp_scaled)

for k_base in range(0, K, K_TILE):
    k_group = k_base // GROUP   # scale-group index — constant within K tile (K_TILE=16 <= GROUP=32)
    for n_base in range(0, N, B_TILE_N):
        # ── Pack B_in tile: (K_TILE × B_TILE_N) ─────────────────────────────
        b_idx = B_indices[k_base:k_base+K_TILE, n_base:n_base+B_TILE_N]  # (K_TILE, B_TILE_N)
        b_idx_combined = (b_idx[:, 1::2] << 4) | b_idx[:, 0::2]          # (K_TILE, B_TILE_N//2), msb=odd col, lsb=even col
        B_hex = codes_to_hex_rows(b_idx_combined, 8)
        #print(B_hex[15])
        b_lut = B_luts_t[n_base >> G:(n_base + B_TILE_N) >> G]            # (B_TILE_N>>G, LUT_SIZE)
        b_lut_in_codes, b_lut_in_bits = tensor_to_custom_fp_codes(b_lut, INPUT_SPEC)
        b_lut_in_codes_hex = codes_to_hex_rows(b_lut_in_codes, 8)
        #print(b_lut_in_codes_hex)
        B_lut_codes, B_lut_bits = tensor_to_custom_fp_codes(b_lut, INPUT_SPEC)
        ni    = (torch.arange(B_TILE_N, device=DEV) >> G).unsqueeze(0).expand(K_TILE, -1)  # col j -> lut j>>G
        B_in  = b_lut[ni, b_idx]                                           # (K_TILE, B_TILE_N)
        B_in_codes, B_in_bits = tensor_to_custom_fp_codes(B_in, INPUT_SPEC)
        B_in_codes_hex = codes_to_hex_rows(B_in_codes, 8)
        # print(B_in_codes_hex)
        # exit()
        # Per-element trace for k=15: before/after LUT projection
        k = 0
        #print(f"\n--- b_idx[{k}] nibble breakdown (byte_pos: hex | MSB[7:4]=odd_col | LSB[3:0]=even_col) ---")
        for c in range(B_TILE_N // 2):
            byte_hex = B_hex[k][c]
            msb_idx  = b_idx[k, 2*c+1].item()   # odd col  -> high nibble
            lsb_idx  = b_idx[k, 2*c  ].item()   # even col -> low nibble
            #print(f"  byte[{c:>2}]={byte_hex}  MSB(j={2*c+1})={msb_idx:x}  LSB(j={2*c})={lsb_idx:x}")
        #print(f"\n--- B_in[{k}] after projection: j, lut_row, idx, lut_val, fp6_hex ---")
        for j in range(B_TILE_N):
            lut_row = j >> G
            idx     = b_idx[k, j].item()
            lut_val = b_lut[lut_row, idx].item()
            #print(f"  j={j:>2}  lut_row={lut_row}  idx={idx:>2}  lut_val={lut_val:>8.4f}  fp6_hex={B_in_codes_hex[k][j]}")
        b_scale    = B_scales_col_q[k_group, n_base:n_base+B_TILE_N]      # (B_TILE_N,)
        B_in_scaled = B_in * b_scale.unsqueeze(0)                          # (K_TILE, B_TILE_N)
        for m_base in range(0, M, A_TILE_M):
            a_idx  = A_indices[m_base:m_base+A_TILE_M, k_base:k_base+K_TILE]  # (A_TILE_M, K_TILE)
            #a_idx = codes_to_hex_rows(a_idx, 8)
            a_idx_hw_layout = a_idx.reshape(A_TILE_M // 2, 2, K_TILE).permute(0, 2, 1).reshape(A_TILE_M // 2, K_TILE * 2)
            a_idx_combined = (a_idx_hw_layout[:, 1::2] << 4) | a_idx_hw_layout[:, 0::2]    # (A_TILE_M//2, K_TILE), msb=odd col, lsb=even col
            a_lut  = A_luts_t[m_base >> G:(m_base + A_TILE_M) >> G]               # (A_TILE_M>>G, LUT_SIZE)
            mi     = (torch.arange(A_TILE_M, device=DEV) >> G).unsqueeze(1).expand(-1, K_TILE)  # row m -> lut m>>G
            A_vals = a_lut[mi, a_idx]                                           # (A_TILE_M, K_TILE)
            a_scale    = A_scales_row_q[m_base:m_base+A_TILE_M, k_group]   # (A_TILE_M,)
            A_vals_scaled = A_vals * a_scale.unsqueeze(1)                   # (A_TILE_M, K_TILE)
            if m_base == 0 and n_base == 0 and k_base == 0:
                A_hex = codes_to_hex_rows(a_idx_combined, 8)
                A_vals_codes, _ = tensor_to_custom_fp_codes(A_vals, INPUT_SPEC)
                A_vals_codes_hex = codes_to_hex_rows(A_vals_codes, 8)
                print(A_vals_codes_hex[1])
                # A_vals in HW layout: interleave pairs of m-rows per k, shape (A_TILE_M//2, K_TILE*2)
                A_vals_hw = A_vals.reshape(A_TILE_M // 2, 2, K_TILE).permute(0, 2, 1).reshape(A_TILE_M // 2, K_TILE * 2)
                A_vals_hw_codes, _ = tensor_to_custom_fp_codes(A_vals_hw, INPUT_SPEC)
                A_vals_hw_hex = codes_to_hex_rows(A_vals_hw_codes, 8)
                # print(f"\n--- A_vals HW layout (A_TILE_M//2={A_TILE_M//2} rows x K_TILE*2={K_TILE*2} cols) ---")
                # print(f"  row r, col 2j   = a_val(row=2r,   k=j)  [LSB / even m-row]")
                # print(f"  row r, col 2j+1 = a_val(row=2r+1, k=j)  [MSB / odd  m-row]")
                # for r in range(min(A_TILE_M // 2, DEBUG_R)):
                #     print(f"  hw_row[{r:>2}]: {A_vals_hw_hex[r]}")
                # exit()
                r = 0  # inspect combined row 0 -> original A rows 0 (LSB) and 1 (MSB)
                print(f"\n--- A a_idx_combined[{r}] nibble breakdown (HW layout) ---")
                print(f"{'k':>3}  {'byte':>4}  {'MSB[7:4]=row{2*r+1}':>20}  {'LSB[3:0]=row{2*r}':>18}")
                for k_pos in range(K_TILE):
                    msb_idx = a_idx[2*r+1, k_pos].item()
                    lsb_idx = a_idx[2*r,   k_pos].item()
                    print(f"{k_pos:>3}  {A_hex[r][k_pos]:>4}  MSB=row{2*r+1},k{k_pos}:{msb_idx:>2x}  LSB=row{2*r},k{k_pos}:{lsb_idx:>2x}")
                print(f"\n--- A_vals after projection (row, k, lut_row, idx, fp6_hex) ---")
                for row in range(2):  # rows 0,1 covered by combined row r=0
                    lut_row = row >> G
                    for k_pos in range(K_TILE):
                        idx     = a_idx[row, k_pos].item()
                        lut_val = a_lut[lut_row, idx].item()
                        ##print(f"  row={row}  k={k_pos:>2}  lut_row={lut_row}  idx={idx:>2}  lut_val={lut_val:>8.4f}  fp6_hex={A_vals_codes_hex[row][k_pos]}")
                # Scale prints for this tile (k_base=0, n_base=0, m_base=0)
                sA = A_scales_row_q[m_base:m_base+A_TILE_M, k_group]       # (A_TILE_M,)
                sB = B_scales_col_q[k_group, n_base:n_base+B_TILE_N]       # (B_TILE_N,)
                sA_codes, sA_bits = tensor_to_custom_fp_codes(sA[:DEBUG_R].unsqueeze(1), SCALE_SPEC)
                sB_codes, sB_bits = tensor_to_custom_fp_codes(sB[:DEBUG_R].unsqueeze(1), SCALE_SPEC)
                sA_hex = codes_to_hex_rows(sA_codes, sA_bits)
                sB_hex = codes_to_hex_rows(sB_codes, sB_bits)
                print(f"\n--- A scales[:{DEBUG_R}] (separate, {SCALE_SPEC} hex) ---")
                print([row[0] for row in sA_hex])
                print(f"\n--- B scales[:{DEBUG_R}] (separate, {SCALE_SPEC} hex) ---")
                print([row[0] for row in sB_hex])
                S_joint = compute_tile_scale_matrix_fpe8m0(
                    A_scales_row_q, B_scales_col_q,
                    m0=m_base, n0=n_base, k0=k_base,
                    TM=A_TILE_M, TN=B_TILE_N,
                    M=M, N=N, K=K, group=GROUP, scale_spec=SCALE_SPEC,
                )
                S_codes, S_bits = tensor_to_custom_fp_codes(S_joint[:DEBUG_R, :DEBUG_R], SCALE_SPEC)
                S_hex = codes_to_hex_rows(S_codes, S_bits)
                print(f"\n--- Joint scale S_tile[:{DEBUG_R},:{DEBUG_R}] ({SCALE_SPEC} hex | float) ---")
<<<<<<< HEAD
                for r in range(DEBUG_R):
                    hex_row   = S_hex[r]
                    float_row = [f"{S_joint[r, c].item():>8.4f}" for c in range(DEBUG_R)]
=======
                S_joint_sub = S_joint[:DEBUG_R, :DEBUG_R]
                for r in range(min(DEBUG_R, S_joint.shape[0])):
                    hex_row   = S_hex[r]
                    float_row = [f"{S_joint_sub[r, c].item():>8.4f}" for c in range(min(DEBUG_R, S_joint.shape[1]))]
>>>>>>> 33b4521c (adding fp6 test case)
                    print(f"  row{r}: {hex_row}  |  {float_row}")
                print(f"\n--- A scales[:{DEBUG_R}] float | hex ---")
                print([f"{v.item():.4f} ({h[0]})" for v, h in zip(sA[:DEBUG_R], sA_hex)])
                print(f"\n--- B scales[:{DEBUG_R}] float | hex ---")
                print([f"{v.item():.4f} ({h[0]})" for v, h in zip(sB[:DEBUG_R], sB_hex)])

                # C_out tile before and after scale (2x2)
                # Hardware-faithful: fp6 product quant + hw_add_bf16 accumulation, no scale
                _prod_quant = make_fp_quantizer(INPUT_SPEC, "zero")
                _in_q = make_fp_quantizer(INPUT_SPEC, rounding="zero")

                C_before_hw = None
                for k_pos in range(K_TILE):
                    a_in_k =A_vals[:DEBUG_R, k_pos]         # (DEBUG_R,)
                    b_in_k = B_in[k_pos, :DEBUG_R]           # (DEBUG_R,)
                    prod = A_vals[:, k_pos:k_pos+1] * B_in[k_pos:k_pos+1, :]
                    if C_before_hw is None:
                        C_before_hw = q_bf16_rne(prod)
                    else:
                        C_before_hw = hw_add_bf16(prod, C_before_hw)
                     
                    if k_pos == 0:
                        a_codes, a_bits = tensor_to_custom_fp_codes(a_in_k.unsqueeze(1), INPUT_SPEC)
                        b_codes, b_bits = tensor_to_custom_fp_codes(b_in_k.unsqueeze(1), INPUT_SPEC)
                        c_codes, c_bits = tensor_to_custom_fp_codes(C_before_hw[:DEBUG_R, :DEBUG_R], "bf16")
                        a_hex = [r[0] for r in codes_to_hex_rows(a_codes, a_bits)]
                        b_hex = [r[0] for r in codes_to_hex_rows(b_codes, b_bits)]
                        c_hex = codes_to_hex_rows(c_codes, c_bits)
                        print(f"\n--- k=0 ---")
                        print(f"  A_in[:{DEBUG_R}], hex={a_hex}")
                        print(f"  B_in[:{DEBUG_R}], hex={b_hex}")
                        print(f"  C[:{DEBUG_R},:{DEBUG_R}], bf16")
                        for row in c_hex:
                            print(f"    {row}")
                        #print(f"  C[:{DEBUG_R},:{DEBUG_R}]  bf16={c_hex}")
                        c_rec = bf16_tensor_to_recfn_hex(C_before_hw[:DEBUG_R, :DEBUG_R])
                        print(f"  C[:{DEBUG_R},:{DEBUG_R}]  recfn:")
                        for row in c_rec:
                            print(f"    {row}")

                # ── Every K_TILE: per-tile + cumulative accumulation ──────────
                _pq_all = make_fp_quantizer(INPUT_SPEC, "zero")
                C_cumulative = None
                C_cumulative_wide = None  # tracks [:DEBUG_R, :N] across K tiles
                for t, kb in enumerate(range(0, K, K_TILE)):
                    # single tile accumulation (no scale)
                    c_tile = None
                    for k_pos in range(kb, kb + K_TILE):
                        prod = A_fp[:, k_pos:k_pos+1] * B_fp[k_pos:k_pos+1, :]
                        if c_tile is None:
                            c_tile = q_bf16_rne(prod)
                        else:
                            c_tile = hw_add_bf16(prod, c_tile)
                    
                    c_tile_r = c_tile[:DEBUG_R, :DEBUG_R]
                    # wide cumulative (no scale) across all N cols
                    c_tile_wide = c_tile[:DEBUG_R, :]
                    C_cumulative_wide = c_tile_wide if C_cumulative_wide is None else hw_add_bf16(C_cumulative_wide, c_tile_wide)
                    # scale and accumulate
                    St = compute_tile_scale_matrix(
                        A_scales_row_q, B_scales_col_q,
                        m0=m_base, n0=n_base, k0=kb, TM=DEBUG_R, TN=DEBUG_R
                    )
                    c_tile_scaled = q_bf16_rne(c_tile_r * St)
                    C_cumulative = c_tile_scaled if C_cumulative is None else hw_add_bf16(C_cumulative, c_tile_scaled)
                    # print
                    tile_codes, tile_bits = tensor_to_custom_fp_codes(c_tile_r, "bf16")
                    scaled_codes, scaled_bits = tensor_to_custom_fp_codes(c_tile_scaled, "bf16")
                    cum_codes, cum_bits = tensor_to_custom_fp_codes(C_cumulative, "bf16")
                   
                    def fmt_row(hex_list, sep=32):
                        groups = [" ".join(hex_list[g:g+sep]) for g in range(0, len(hex_list), sep)]
                        return " | ".join(groups)

                    print(f"\n=== K_TILE {t+1} (k={kb}..{kb+K_TILE-1}) ===")
                    print(f"  tile (no scale):")
                    for row in codes_to_hex_rows(tile_codes, tile_bits):
                        print(f"    {fmt_row(row)}")
                    print(f"  tile (scaled):")
                    for row in codes_to_hex_rows(scaled_codes, scaled_bits):
                        print(f"    {fmt_row(row)}")
                    print(f"  cumulative tiles 1..{t+1}:")
                    for row in codes_to_hex_rows(cum_codes, cum_bits):
                        print(f"    {fmt_row(row)}")
                    wide_codes, wide_bits = tensor_to_custom_fp_codes(C_cumulative_wide, "bf16")
                    print(f"  cumulative (all N cols, no scale) tiles 1..{t+1}:")
                    for i, row in enumerate(codes_to_hex_rows(wide_codes, wide_bits)):
                        print(f"    row{i:>2}: {fmt_row(row)}")
                # exit()


print("[Step 5]: Compare C_out with tiled_matmul_hwlike golden")
# print(fp8_matmul_model.TILE)  # should be 16 = K_TILE
# print(fp8_matmul_model.GROUP) # should be 32 = GROUP
# exit()
fp8_matmul_model.prod_quant = lambda x: x   # no fp6 prod quant — matches hardware
C_golden = tiled_matmul_hwlike(
    A_fp, B_fp,
    A_scales_row_q, B_scales_col_q,
    verbose=False,
)
C_out_bf16   = q_bf16_rne(C_out)
C_golden_bf16 = q_bf16_rne(C_golden)
C_golden_r = C_golden_bf16[:DEBUG_R, :DEBUG_R]
C_golden_r_codes, C_golden_r_bits = tensor_to_custom_fp_codes(C_golden_r, "bf16")
# print(f"\n--- C_golden[:{DEBUG_R},:{DEBUG_R}] (bf16 hex) ---")
# for row in codes_to_hex_rows(C_golden_r_codes, C_golden_r_bits):
#     print(row)
# diff = (C_out_bf16 - C_golden_bf16).abs()
# print(f"Max abs diff: {diff.max().item()}")
# print(f"Num mismatched elements: {(diff > 0).sum().item()} / {diff.numel()}")
# if diff.max().item() > 0:
#     idx = diff.argmax()
#     m, n = idx // N, idx % N
#     print(f"Largest diff at ({m},{n}): demo={C_out_bf16[m,n].item():.6f}  golden={C_golden_bf16[m,n].item():.6f}")

print("[Step 6]: Write C header with HW data layout")
C_out_bf16 = q_bf16_rne(C_golden)
# write_c_header_tiled_hw(
#     path           = "./include/matmul_data_mx_lut_hw.h",
#     M=M, K=K, N=N,
#     group          = GROUP,
#     input_spec     = INPUT_SPEC,
#     acc_spec       = "bf16",
#     scale_spec     = SCALE_SPEC,
#     lut_index_bits = LUT_INDEX_BITS,
#     A_in           = A_indices,       # [M, K]  — raw LUT indices
#     B_in           = B_indices,       # [K, N]  — raw LUT indices
#     A_scales_row_q = A_scales_row_q,  # [M, Gk]
#     B_scales_col_q = B_scales_col_q,  # [Gk, N]
#     C_out_bf16     = C_out_bf16,      # [M, N]
#     A_lut          = A_luts_t,        # [M, LUT_SIZE] per-row LUTs
#     B_lut          = B_luts_t,        # [N, LUT_SIZE] per-col LUTs
#     a_tile_m       = A_TILE_M,
#     k_tile         = K_TILE,
# )
print("Header written to matmul_data_mx_lut_hw.h")

print("\n[Step 7]: Requantize C_golden_bf16 via matrix_mx_requantize")
from fp8_matmul_model import matrix_mx_requantize
C_requantized, C_req_scales = matrix_mx_requantize(
        C_golden_bf16,
<<<<<<< HEAD
        quant_spec=INPUT_SPEC,
        group_size=GROUP,
        Gk=Gk)
=======
        quant_spec=INPUT_SPEC)
>>>>>>> 33b4521c (adding fp6 test case)
print(f"  quant_spec: {INPUT_SPEC}  scale_spec: {SCALE_SPEC}")
# print(f"  C_requantized shape: {list(C_requantized.shape)}  C_req_scales shape: {list(C_req_scales.shape)}")
# req_codes, req_bits = tensor_to_custom_fp_codes(C_requantized, INPUT_SPEC)
# print(f"\n--- C_requantized (all {M}x{N}, {INPUT_SPEC} hex) ---")
# for i, row in enumerate(codes_to_hex_rows(req_codes, req_bits)):
#     print(f"  row {i:3d}: {row}")
# print(f"\n--- C_req_scales ({SCALE_SPEC} hex) ---")
scale_codes, scale_bits = tensor_to_custom_fp_codes(C_req_scales, SCALE_SPEC)
# for i, row in enumerate(codes_to_hex_rows(scale_codes, scale_bits)):
#     print(f"  row {i:3d}: {row}")

# Per-row, per-group breakdown (first 2 rows only)
Gk_out = C_golden_bf16.shape[1] // GROUP
print(f"\n--- C per-row group breakdown (group_size={GROUP}, Gk={Gk_out}, showing first 2 rows) ---")
for m in range(min(2, C_golden_bf16.shape[0])):
    print(f"  row {m:3d}:")
    bf16_row = C_golden_bf16[m]
    for g in range(1):
        col_start = g * GROUP
        col_end   = col_start + GROUP
        golden_group = bf16_row[col_start:col_end]
        quant_group  = C_requantized[m, col_start:col_end]
        scale_val    = C_req_scales[m, g].item()
        scale_e8m0_hex = f"{scale_codes[m][g]:02x}"   # e8m0 = 8-bit biased exponent
        # maximal magnitude in the group
        abs_group = golden_group.abs()
        max_mag_val = abs_group.max().item()
        max_mag_bf16 = abs_group.max().to(torch.bfloat16).view(torch.int16).item() & 0xFFFF
        q_codes, q_bits = tensor_to_custom_fp_codes(quant_group.unsqueeze(0), INPUT_SPEC)
        q_hex = codes_to_hex_rows(q_codes, q_bits)[0]
        golden_bits = golden_group.to(torch.bfloat16).view(torch.int16)
        golden_hex = " ".join(f"{b.item() & 0xFFFF:04x}" for b in golden_bits)
        print(f"    group {g} (cols {col_start:3d}-{col_end-1:3d}) \n"
              f"  max_magnitude={max_mag_val:.6g} (bf16=0x{max_mag_bf16:04x}) \n"
              f"  scale={scale_val:.6g} (e8m0=0x{scale_e8m0_hex}) \n"
              f"  golden_bf16=[{golden_hex}] \n"
              f"  quantized={q_hex}")


print("\n[Step 8]: project the quantized fp6 down to INT4 using C_luts")

# Pre-compute: convert every C_lut (float values) to 6-bit FP6 codes once
C_luts_t = torch.stack(C_luts)                                        # (M>>G, LUT_SIZE)
C_luts_codes_raw, _ = tensor_to_custom_fp_codes(C_luts_t, INPUT_SPEC) # list[list[int]], 6-bit

# Convert quantized C float values to 6-bit FP6 codes
C_req_codes_raw, C_req_bits = tensor_to_custom_fp_codes(
    C_requantized.float().view(M, N), INPUT_SPEC)                      # list[list[int]]

# For element (m, n): LUT = C_luts[m >> G]  (M-dim row grouping, same as A_luts)
# Find nearest LUT entry via fp6e3m2_nearest_finder and store 4-bit index
C_proj = torch.zeros(M, N, dtype=torch.int32)
for m in range(M):
    lut_idx      = m >> G
    lut_codes    = C_luts_codes_raw[lut_idx]
    for n in range(N):
        fp6_code     = C_req_codes_raw[m][n]
        C_proj[m, n] = fp6e3m2_nearest_finder(fp6_code, lut_codes)

# ── Debug print for first NUM_PRINT_MGRP row-groups, showing first NUM_PRINT_COLS cols ──
NUM_PRINT_MGRP = 2   # how many M-groups to show
NUM_PRINT_COLS = 4   # how many cols to show per group
m_group_size   = 1 << G

print(f"  INPUT_SPEC={INPUT_SPEC}  LUT_SIZE={LUT_SIZE}  M-group-size={m_group_size}")

for mg in range(min(NUM_PRINT_MGRP, M >> G)):
    m_start   = mg * m_group_size
    m_end     = m_start + m_group_size
    lut_hex   = [f"{c:02x}" for c in C_luts_codes_raw[mg]]
    lut_vals  = [f"{C_luts_t[mg, e].item():.4f}" for e in range(LUT_SIZE)]
    print(f"\n  === M-group {mg} (rows {m_start}-{m_end-1}), LUT[{mg}] ===")
    print(f"    LUT entries (fp6 hex): {lut_hex}")
    print(f"    LUT entries (float):   {lut_vals}")
    for m in range(m_start, m_end):
        fp6_hex      = [f"{C_req_codes_raw[m][n]:02x}" for n in range(NUM_PRINT_COLS)]
        indices      = [C_proj[m, n].item()            for n in range(NUM_PRINT_COLS)]
        lut_looked_up= [f"{C_luts_t[mg, indices[i]].item():.4f}" for i in range(NUM_PRINT_COLS)]
        print(f"    row {m:3d}: fp6={fp6_hex}  ->  idx={indices}  ->  lut_val={lut_looked_up}")

<<<<<<< HEAD
print("\n--- C_proj flat [M, N] (4-bit indices) ---")
for m in range(M):
    flat_hex = " ".join(f"{C_proj[m, n].item():x}" for n in range(N))
    print(f"  row {m:3d}: {flat_hex}")

print("\n--- C_proj A_in HW layout [M//2, N] (bits[3:0]=even row, bits[7:4]=odd row per col) ---")
C_proj_hw = _a_indices_to_hw_layout(C_proj, a_tile_m=A_TILE_M, k_tile=K_TILE)  # [M//2, N]
for r in range(M // 2):
    hw_hex = " ".join(f"{C_proj_hw[r][n]:02x}" for n in range(N))
    print(f"  hw row {r:3d}: {hw_hex}")
    
=======
print("\n--- C_req_codes_raw first 32 rows (fp6 codes, hex) ---")
for m in range(min(128, M)):
    groups = [" ".join(f"{C_req_codes_raw[m][n]:02x}" for n in range(g, min(g + 32, N)))
              for g in range(0, N, 32)]
    print(f"  row {m:3d}: " + " | ".join(groups))

print("\n--- C_golden_bf16 first 32 rows (bf16 hex) ---")
for m in range(min(128, M)):
    groups = [" ".join(f"{C_golden_bf16[m, n].to(torch.bfloat16).view(torch.int16).item() & 0xFFFF:04x}"
                       for n in range(g, min(g + 32, N)))
              for g in range(0, N, 32)]
    print(f"  row {m:3d}: " + " | ".join(groups))

print("\n--- C_proj A_in HW layout [M//2, N] (bits[3:0]=even row, bits[7:4]=odd row per col) ---")
C_proj_hw = _a_indices_to_hw_layout(C_proj, a_tile_m=A_TILE_M, k_tile=K_TILE)  # [M//2, N]
# for r in range(M // 2):
#     hw_hex = " ".join(f"{C_proj_hw[r][n]:02x}" for n in range(N))
#     print(f"  hw row {r:3d}: {hw_hex}")
for m in range(min(64, M)):
    groups = [" ".join(f"{C_proj_hw[m][n]:02x}" for n in range(g, min(g + 32, N)))
              for g in range(0, N, 32)]
    print(f"  row {m:3d}: " + " | ".join(groups))

>>>>>>> 33b4521c (adding fp6 test case)
print("\n[Step 9]: Write header with C_lut, C_proj indices, and C scales")
write_c_header_tiled_hw(
    path           = "./include/matmul_data_mx_lut_hw.h",
    M=M, K=K, N=N,
    group          = GROUP,
    input_spec     = INPUT_SPEC,
    acc_spec       = "bf16",
    scale_spec     = SCALE_SPEC,
    lut_index_bits = LUT_INDEX_BITS,
    A_in           = A_indices,
    B_in           = B_indices,
    A_scales_row_q = A_scales_row_q,
    B_scales_col_q = B_scales_col_q,
    C_out_bf16     = C_out_bf16,
    C_out_quantized= C_proj,
    C_out_scales   = C_req_scales,
    A_lut          = A_luts_t,
    B_lut          = B_luts_t,
    C_lut          = C_luts_t,
    a_tile_m       = A_TILE_M,
    k_tile         = K_TILE,
)
print("Header written to ./include/matmul_data_mx_lut_hw.h")

# Append C_proj in A_in HW layout [M//2, N] to the header
HEADER_PATH = "./include/matmul_data_mx_lut_hw.h"
with open(HEADER_PATH, "r") as f:
    content = f.read()
# Remove existing #endif, append C_proj_hw array, then re-close with #endif
guard = HEADER_PATH.upper()
for ch in [".", "/", "\\", "-"]:
    guard = guard.replace(ch, "_")
rows_hw = M // 2
c_proj_hw_lines = ",\n".join(
    "    { " + ", ".join(f"0x{C_proj_hw[r][n]:02x}" for n in range(N)) + " }"
    for r in range(rows_hw)
)
c_proj_hw_section = (
    f"// C_proj A_in HW layout [{rows_hw}][{N}]: bits[3:0]=even row, bits[7:4]=odd row\n"
    f"static const uint8_t C_proj_hw[{rows_hw}][{N}] = {{\n{c_proj_hw_lines}\n}};\n\n"
    f"#endif // {guard}\n"
)
content = content.replace(f"#endif // {guard}\n", c_proj_hw_section)
with open(HEADER_PATH, "w") as f:
    f.write(content)
print("C_proj_hw appended to header.")