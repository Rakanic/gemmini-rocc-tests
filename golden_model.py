#!/usr/bin/env python3
import math
import re
import time
from typing import Callable, Optional, Dict, Tuple, List

import torch

Tensor = torch.Tensor
QuantFn = Optional[Callable[[Tensor], Tensor]]

# ----------------------------------------------------------------------
# FP format presets
# ----------------------------------------------------------------------

_FP_PRESETS = {
    "fp16":      dict(exp=5, man=10),
    "bf16":      dict(exp=8, man=7),
    "fp8:e4m3":  dict(exp=4, man=3),
    "fp8:e5m2":  dict(exp=5, man=2),

    # Experimental / custom
    "fp26":      dict(exp=7, man=18),
    "fp20":      dict(exp=6, man=13),
    "fp14":      dict(exp=5, man=8),

    # Scaling format: fpE8M0 (1 sign, 8 exp, 0 mant)
    "fpe8m0":    dict(exp=8, man=0),
}

# ----------------------------------------------------------------------
# Spec parsing / quantization helpers
# ----------------------------------------------------------------------

def trunc_product_mantissa(x: torch.Tensor, frac_bits: int) -> torch.Tensor:
    """
    Truncate mantissa (fraction) bits of x to frac_bits, with RTZ/chop behavior.

    Models: x = sign * (1.fraction) * 2^e  (normalized)
      -> fraction chopped to frac_bits

    Notes:
    - Keeps exponent basically FP32-wide (no exponent quantization).
    - Flushes subnormals to 0 for simplicity (often matches hardware if you don't support denorms).
    - Preserves inf/nan.
    """
    x = x.to(torch.float32)

    # Handle specials
    is_finite = torch.isfinite(x)
    ax = x.abs()
    out = x.clone()

    # zero stays zero
    nz = (ax != 0) & is_finite
    if not torch.any(nz):
        return out

    xn = x[nz]

    # Decompose: xn = m * 2^e with m in [0.5, 1)
    m, e = torch.frexp(xn)  # m in (-1, -0.5] U [0.5, 1)
    sign = torch.sign(m)
    m = m.abs()

    # Convert to [1,2): m2 = m*2, e2 = e-1
    m2 = m * 2.0
    e2 = e - 1

    # Chop fraction bits: m2 = 1 + frac, frac in [0,1)
    frac = m2 - 1.0
    scale = float(1 << frac_bits)
    frac_q = torch.floor(frac * scale) / scale   # RTZ/chop
    m2_q = 1.0 + frac_q

    # Recompose
    out_nz = (sign * m2_q) * torch.ldexp(torch.ones_like(m2_q), e2)

    # Flush subnormals to zero (optional, but often what hardware does)
    # Smallest normal float32 is ~1.175e-38; anything smaller becomes subnormal.
    # If your hardware *does* keep subnormals, remove this block.
    out_nz = torch.where(out_nz.abs() < torch.finfo(torch.float32).tiny, torch.zeros_like(out_nz), out_nz)

    out[nz] = out_nz
    return out

def q_bf16_rne(x: torch.Tensor) -> torch.Tensor:
    # PyTorch bf16 cast is round-to-nearest-even; converting back keeps values as fp32
    return x.to(torch.bfloat16).to(torch.float32)

def hw_add_bf16(raw_in: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    raw_q = q_bf16_rne(raw_in)
    c_q   = q_bf16_rne(c)
    return q_bf16_rne(raw_q + c_q)

def zext_codes(codes: List[List[int]], target_bits: int) -> List[List[int]]:
    mask = (1 << target_bits) - 1
    return [[(code & mask) for code in row] for row in codes]

def parse_fp_spec(spec: str) -> Tuple[int, int]:
    """
    Parse something like:
      - 'bf16'       -> (8, 7)
      - 'fp8:e4m3'   -> (4, 3)
      - 'fp32'       -> (8, 23)
      - 'fpe8m0'     -> (8, 0)
    using _FP_PRESETS or the regex 'fp\\d+:(e(\\d+)m(\\d+))'.
    """
    s = spec.strip().lower()
    if s in _FP_PRESETS:
        return _FP_PRESETS[s]["exp"], _FP_PRESETS[s]["man"]
    if s == "fp32":
        return 8, 23
    m = re.fullmatch(r"fp\d+:(e(\d+)m(\d+))", s)
    if m:
        return int(m.group(2)), int(m.group(3))
    raise ValueError(f"Unrecognized floating-point spec: {spec}")


def float_quantize_trunc(x: Tensor, exp: int, man: int) -> Tensor:
    """
    Quantize to a floating-point format with `exp` exponent bits and `man` mantissa bits,
    using round-toward-zero (truncation).
    - Normalized numbers only (subnormals flushed to zero).
    - Overflow saturates to max finite.
    """
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)

    out = torch.zeros_like(x)
    is_zero = (x == 0)
    if torch.all(is_zero):
        return out

    sign = torch.sign(x)
    ax = x.abs()

    nz_mask = ax > 0
    ax_nz = ax[nz_mask]
    if ax_nz.numel() == 0:
        return out

    log2_ax = torch.log2(ax_nz)
    E = torch.floor(log2_ax)

    bias = (1 << (exp - 1)) - 1   # 2^(exp-1) - 1
    emin = 1 - bias
    emax = bias

    underflow_mask = E < emin
    overflow_mask = E > emax
    normal_mask = (~underflow_mask) & (~overflow_mask)

    ax_q = torch.zeros_like(ax_nz)

    # Underflow -> 0
    ax_q[underflow_mask] = 0.0

    # Overflow -> max finite
    if overflow_mask.any():
        E_max = float(emax)
        base_max = torch.pow(
            torch.tensor(2.0, dtype=ax_nz.dtype, device=ax_nz.device),
            E_max,
        )
        delta_max = base_max / (2 ** man)
        max_val = base_max + (2 ** man - 1) * delta_max
        ax_q[overflow_mask] = max_val

    # Normal
    if normal_mask.any():
        E_norm = E[normal_mask]
        x_norm = ax_nz[normal_mask]

        base = torch.pow(
            torch.tensor(2.0, dtype=ax_nz.dtype, device=ax_nz.device),
            E_norm,
        )
        delta = base / (2 ** man)
        t = (x_norm - base) / delta
        # Round-toward-zero in magnitude: floor
        k = torch.floor(torch.clamp(t, 0, 2 ** man - 1 - 1e-7))

        q_norm = base + k * delta
        ax_q[normal_mask] = q_norm

    out_nz = sign[nz_mask] * ax_q
    out[nz_mask] = out_nz
    return out


def make_fp_quantizer(spec: str, rounding: str = "nearest") -> QuantFn:
    """
    Create a quantizer for a given FP format.
    rounding:
      - 'nearest' or 'nearest_even' or 'stochastic' -> use qtorch.float_quantize
      - 'zero' / 'toward_zero' / 'trunc' / 'rtz'    -> custom RTZ
    """
    s = spec.strip().lower()
    if s in ("", "none", "identity"):
        return None
    if s == "fp32":
        return None

    e, m = parse_fp_spec(spec)
    rounding = rounding.lower()

    if rounding in ("nearest", "nearest_even", "stochastic"):
        from qtorch.quant import float_quantize
        mode = "nearest" if rounding in ("nearest", "nearest_even") else "stochastic"
        return lambda x: float_quantize(x, exp=e, man=m, rounding=mode)

    if rounding in ("zero", "toward_zero", "trunc", "rtz"):
        return lambda x: float_quantize_trunc(x, exp=e, man=m)

    raise ValueError(f"Unsupported rounding mode: {rounding}")


def tensor_to_custom_fp_codes(t: Tensor, spec: str) -> Tuple[List[List[int]], int]:
    """
    Encode a *quantized* tensor (values assumed representable in the given FP format)
    into sign/exponent/mantissa bits.

    Format:
      - 1 sign bit
      - e_bits exponent bits with bias
      - m_bits mantissa bits
      - no subnormals (all denormals flushed to zero)
      - overflow assumed already saturated

    Returns:
      codes: List of rows, each row is a list of integer codes.
      total_bits: 1 + e_bits + m_bits
    """
    e_bits, m_bits = parse_fp_spec(spec)
    total_bits = 1 + e_bits + m_bits

    bias = (1 << (e_bits - 1)) - 1
    emin = 1 - bias
    emax = bias

    arr = t.detach().cpu()
    if arr.ndim != 2:
        raise ValueError("Expected 2D tensor for hex dump.")
    rows, cols = arr.shape

    out_codes: List[List[int]] = []
    for r in range(rows):
        row_codes: List[int] = []
        for c in range(cols):
            v = float(arr[r, c].item())
            if v == 0.0 or not math.isfinite(v):
                code = 0
            else:
                sign = 1 if v < 0 else 0
                av = abs(v)
                E = math.floor(math.log2(av))

                if E < emin:
                    code = 0
                else:
                    if E > emax:
                        E_used = emax
                        base = 2.0 ** E_used
                        delta = base / (2 ** m_bits)
                        mant = (2 ** m_bits) - 1
                    else:
                        E_used = E
                        base = 2.0 ** E_used
                        delta = base / (2 ** m_bits)
                        tpos = (av - base) / delta
                        mant = int(round(tpos))
                        mant = max(0, min(mant, 2 ** m_bits - 1))

                    exp_bits_val = int(E_used + bias)
                    code = ((sign & 0x1) << (e_bits + m_bits)) | \
                           ((exp_bits_val & ((1 << e_bits) - 1)) << m_bits) | \
                           (mant & ((1 << m_bits) - 1))

            row_codes.append(code)
        out_codes.append(row_codes)
    return out_codes, total_bits


def codes_to_hex_rows(codes: List[List[int]], total_bits: int) -> List[List[str]]:
    width = (total_bits + 3) // 4  # hex digits
    return [[f"{code:0{width}x}" for code in row] for row in codes]


def c_type_for_bits(total_bits: int) -> str:
    if total_bits <= 8:
        return "uint8_t"
    elif total_bits <= 16:
        return "uint16_t"
    elif total_bits <= 32:
        return "uint32_t"
    else:
        return "uint64_t"

# ----------------------------------------------------------------------
# Matmul and error metrics
# ----------------------------------------------------------------------

def matmul_outer(A: Tensor, B: Tensor) -> Tensor:
    """
    Reference outer-product matmul (full precision).
    C = sum_k A[:,k] * B[k,:]
    """
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = torch.zeros((M, N), dtype=A.dtype, device=A.device)
    for k in range(K):
        C += torch.outer(A[:, k], B[k, :])
    return C


def matmul_outer_quantized_hwlike(
    A_in: Tensor,
    B_in: Tensor,
    prod_quant: QuantFn = None,   # whatever you already use for mul output rounding/trunc
    acc_each_add: bool = True,    # true matches “round after add each cycle”
) -> Tensor:
    M, K = A_in.shape
    K2, N = B_in.shape
    assert K == K2

    # Accumulator stored as fp32 values, but forced onto BF16 grid after each add
    C = torch.zeros((M, N), dtype=torch.float32, device=A_in.device)

    for k in range(K):
        outer = torch.outer(A_in[:, k], B_in[k, :])        # mul
        if prod_quant:
            outer = prod_quant(outer)                      # round/trunc after mul (your rule)

        # IMPORTANT: match your resize(... -> cType) before the add
        outer = q_bf16_rne(outer)

        # Add in BF16 and round result to BF16
        C = q_bf16_rne(q_bf16_rne(C) + q_bf16_rne(outer))   # BF16 add + RNE

        if not acc_each_add:
            pass  # keep as-is; if you want “only round at end”, move rounding outside loop

    return C


def matmul_loss(C_ref: Tensor, C_quant: Tensor) -> Dict[str, float]:
    diff = (C_quant - C_ref).detach()
    mse = float(torch.mean(diff.pow(2)).item())
    mae = float(torch.mean(diff.abs()).item())
    max_abs = float(torch.max(diff.abs()).item())
    fro_ref = float(torch.linalg.norm(C_ref).item())
    fro_diff = float(torch.linalg.norm(diff).item())
    rel_fro = (fro_diff / (fro_ref + 1e-12)) if fro_ref != 0.0 else float("inf")
    return {
        "mse": mse,
        "mae": mae,
        "max_abs": max_abs,
        "rel_fro": rel_fro,
    }

# ----------------------------------------------------------------------
# Header file writer
# ----------------------------------------------------------------------

def write_c_header(
    path: str,
    M: int,
    K: int,
    N: int,
    input_spec: str,
    acc_spec: str,
    scale_spec: str,
    scaled_spec: str,
    A_codes: List[List[int]],
    A_bits: int,
    B_codes: List[List[int]],
    B_bits: int,
    C_codes: List[List[int]],
    C_bits: int,
    S_codes: List[List[int]],
    S_bits: int,
    C_scaled_codes: List[List[int]],
    C_scaled_bits: int,
):
    # Derive a crude include guard from file name
    guard = path.upper()
    for ch in [".", "/", "\\", "-"]:
        guard = guard.replace(ch, "_")

    # Force storage types
    in_type_A = "uint16_t"
    in_type_B = "uint32_t"

    # Zero-extend element codes into those storage widths
    A_store_bits = 16
    B_store_bits = 32
    A_codes_store = zext_codes(A_codes, A_store_bits)
    B_codes_store = zext_codes(B_codes, B_store_bits)

    A_hex = codes_to_hex_rows(A_codes_store, A_store_bits)
    B_hex = codes_to_hex_rows(B_codes_store, B_store_bits)

    out_type  = c_type_for_bits(C_bits)
    scale_type = c_type_for_bits(S_bits)
    scaled_type = c_type_for_bits(C_scaled_bits)

    C_hex = codes_to_hex_rows(C_codes, C_bits)
    S_hex = codes_to_hex_rows(S_codes, S_bits)
    C_scaled_hex = codes_to_hex_rows(C_scaled_codes, C_scaled_bits)

    def format_2d_array(hex_rows: List[List[str]]) -> str:
        lines = []
        for row in hex_rows:
            line = ", ".join(f"0x{h}" for h in row)
            lines.append("    { " + line + " }")
        return ",\n".join(lines)

    with open(path, "w") as f:
        f.write(f"#ifndef {guard}\n")
        f.write(f"#define {guard}\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"#define MATMUL_M {M}\n")
        f.write(f"#define MATMUL_K {K}\n")
        f.write(f"#define MATMUL_N {N}\n\n")

        f.write(f"// Input precision: {input_spec}\n")
        f.write(f"static const {in_type_A} A_in[MATMUL_M][MATMUL_K] = {{\n")
        f.write(format_2d_array(A_hex))
        f.write("\n};\n\n")

        f.write(f"static const {in_type_B} B_in[MATMUL_K][MATMUL_N] = {{\n")
        f.write(format_2d_array(B_hex))
        f.write("\n};\n\n")

        f.write(f"// Output precision (pre-scale): {acc_spec}\n")
        f.write(f"static const {out_type} C_out[MATMUL_M][MATMUL_N] = {{\n")
        f.write(format_2d_array(C_hex))
        f.write("\n};\n\n")

        f.write(f"// Scaling factors (elementwise) in {scale_spec}\n")
        f.write(f"static const {scale_type} C_scale[MATMUL_M][MATMUL_N] = {{\n")
        f.write(format_2d_array(S_hex))
        f.write("\n};\n\n")

        f.write(f"// Scaled outputs, quantized to {scaled_spec}\n")
        f.write(f"static const {scaled_type} C_scaled[MATMUL_M][MATMUL_N] = {{\n")
        f.write(format_2d_array(C_scaled_hex))
        f.write("\n};\n\n")

        f.write(f"#endif // {guard}\n")

# ----------------------------------------------------------------------
# Single experiment: generate A,B, quantize, run matmul, print everything
# ----------------------------------------------------------------------

def run_experiment(
    M: int,
    K: int,
    N: int,
    input_spec: str,
    prod_spec: str,
    acc_spec: str,
    scaled_spec: Optional[str] = None,
    input_rounding: str = "nearest",
    prod_rounding: str = "nearest",
    acc_rounding: str = "nearest",
    scale_spec: str = "fpe8m0",
    scale_exp: int = 0,
    header_path: str = "matmul_data.h",
    seed: int = 0,
    device: str = "cpu",
):
    torch.manual_seed(seed)
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")

    if scaled_spec is None:
        scaled_spec = prod_spec  # default: shrink back to same FP8 as product

    # Generate random inputs
    A = torch.randn(M, K, device=dev, dtype=torch.float32)
    B = torch.randn(K, N, device=dev, dtype=torch.float32)

    print("=== Configuration ===")
    print(f"M={M}, K={K}, N={N}")
    print(f"input_spec={input_spec}, input_rounding={input_rounding}")
    print(f"prod_spec={prod_spec}, prod_rounding={prod_rounding}")
    print(f"acc_spec={acc_spec}, acc_rounding={acc_rounding}")
    print(f"scaled_spec={scaled_spec} (post-scale output FP8)")
    print(f"scale_spec={scale_spec}, scale_exp={scale_exp}")
    print(f"header_path={header_path}")
    print(f"device={dev}, seed={seed}")
    print()

    print("=== Full-precision Inputs (FP32) ===")
    print("A_fp32:")
    print(A.detach().cpu().numpy())
    print("\nB_fp32:")
    print(B.detach().cpu().numpy())
    print()

    # Build quantizers
    in_q   = make_fp_quantizer(input_spec, rounding=input_rounding)
    if args.prod_mant_bits is not None:
        prod_q = lambda t: trunc_product_mantissa(t, frac_bits=args.prod_mant_bits)
        print(f"Product quantization: mantissa chopped to {args.prod_mant_bits} fraction bits")
    else:
        prod_q = make_fp_quantizer(prod_spec, rounding=prod_rounding)
    acc_q  = q_bf16_rne if acc_rounding == "q_bf16_rne" else make_fp_quantizer(acc_spec,   rounding=acc_rounding)
    # Post-scale FP8 quantizer uses nearest-even
    scaled_q = make_fp_quantizer(scaled_spec, rounding="nearest_even")

    # Quantize inputs to input precision (for storage / feeding MAC)
    A_in = in_q(A) if in_q is not None else A
    B_in = in_q(B) if in_q is not None else B

    print("\n=== TRACE C[0,0] with product mantissa chopped to 8 bits ===")
    _ = trace_dot(
        A_in, B_in, 0, 0,
        prod_quant=lambda t: trunc_product_mantissa(t, frac_bits=8),
        acc_quant=acc_q,
        cast_prod=True,
        cast_each_add=True,   # if you quantize accumulator each cycle
        verbose=True,
    )

    print("=== Inputs quantized to input precision ===")
    print(f"A_in (float, {input_spec}, rounding={input_rounding}):")
    print(A_in.detach().cpu().numpy())
    print()
    print(f"B_in (float, {input_spec}, rounding={input_rounding}):")
    print(B_in.detach().cpu().numpy())
    print()

    # Hex dump of quantized inputs (input format, e.g. fp8)
    try:
        A_codes, A_bits = tensor_to_custom_fp_codes(A_in, input_spec)
        B_codes, B_bits = tensor_to_custom_fp_codes(B_in, input_spec)
        A_hex = codes_to_hex_rows(A_codes, A_bits)
        B_hex = codes_to_hex_rows(B_codes, B_bits)

        print(f"=== A_in hex encoding ({input_spec}) ===")
        for row in A_hex:
            print(" ".join(row))
        print()
        print(f"=== B_in hex encoding ({input_spec}) ===")
        for row in B_hex:
            print(" ".join(row))
        print()
    except ValueError as e:
        print(f"[WARN] Could not hex-encode inputs for {input_spec}: {e}")
        A_codes = B_codes = []
        A_bits = B_bits = 0
        print()

    # ------------------------------------------------------------------
    # Reference 1: Fully ideal FP32 (unquantized inputs)
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    C_ref_full = matmul_outer(A, B)
    t1 = time.perf_counter()

    print("=== Reference Output 1: FP32 matmul, FP32 inputs ===")
    print("C_ref_full:")
    print(C_ref_full.detach().cpu().numpy())
    print(f"(ref_full matmul time: {t1 - t0:.6f} s)")
    print()

    # ------------------------------------------------------------------
    # Reference 2: FP32 matmul, but with input-quantized inputs
    # (ideal product+accumulator precision, but quantized inputs)
    # ------------------------------------------------------------------
    t2 = time.perf_counter()
    C_ref_in = matmul_outer(A_in, B_in)
    t3 = time.perf_counter()

    print("=== Reference Output 2: FP32 matmul, input-quantized inputs ===")
    print(f"C_ref_in (inputs in {input_spec}, ideal MAC):")
    print(C_ref_in.detach().cpu().numpy())
    print(f"(ref_in matmul time: {t3 - t2:.6f} s)")
    print()

    # ------------------------------------------------------------------
    # Quantized MAC matmul: product+acc quantization (hardware-like)
    # ------------------------------------------------------------------
    t4 = time.perf_counter()
    C_quant = matmul_outer_quantized_hwlike(
        A_in, B_in,
        prod_quant=prod_q,     # keep your existing mul model
        acc_each_add=True
    )
    t5 = time.perf_counter()

    print("=== Quantized Output: product+acc quantization ===")
    print(f"C_quant (acc_spec={acc_spec}, rounding={acc_rounding}):")
    print(C_quant.detach().cpu().numpy())
    print(f"(quantized matmul time: {t5 - t4:.6f} s)")
    print()

    # Hex dump of output in accumulator format (e.g. bf16)
    try:
        C_codes, C_bits = tensor_to_custom_fp_codes(C_quant, acc_spec)
        C_hex = codes_to_hex_rows(C_codes, C_bits)
        print(f"=== C_quant hex encoding ({acc_spec}) ===")
        for row in C_hex:
            print(" ".join(row))
        print()
    except ValueError as e:
        print(f"[WARN] Could not hex-encode output for {acc_spec}: {e}")
        C_codes = []
        C_bits = 0
        print()

    # ------------------------------------------------------------------
    # Scaling factors (elementwise) in fpE8M0 (scale_spec)
    # ------------------------------------------------------------------
    scale_val = float(2.0 ** scale_exp)
    S = torch.full_like(C_quant, scale_val)

    print(f"=== Scaling factors (float, {scale_spec}, before quantization) ===")
    print(S.detach().cpu().numpy())
    print()

    # Quantize scaling factors to scale_spec
    S_q = make_fp_quantizer(scale_spec, "nearest")(S)  # scale usually nearest-even/nearest is fine

    try:
        S_codes, S_bits = tensor_to_custom_fp_codes(S_q, scale_spec)
        S_hex = codes_to_hex_rows(S_codes, S_bits)
        print(f"=== Scaling factors hex encoding ({scale_spec}) ===")
        for row in S_hex:
            print(" ".join(row))
        print()
    except ValueError as e:
        print(f"[WARN] Could not hex-encode scaling factors for {scale_spec}: {e}")
        S_codes = []
        S_bits = 0
        print()

    # ------------------------------------------------------------------
    # Apply scaling (elementwise): C_scaled_pre_q = C_quant * S_q
    # Then quantize scaled outputs to FP8 (scaled_spec) with nearest-even
    # ------------------------------------------------------------------
    C_scaled_pre_q = C_quant * S_q

    print("=== Output before scaling (C_quant) ===")
    print(C_quant.detach().cpu().numpy())
    print()

    print("=== Output after scaling (float, before re-quantization to FP8) ===")
    print(C_scaled_pre_q.detach().cpu().numpy())
    print()

    # Quantize scaled outputs to FP8 (scaled_spec), nearest-even
    C_scaled = scaled_q(C_scaled_pre_q) if scaled_q is not None else C_scaled_pre_q

    print(f"=== Output after scaling, quantized to {scaled_spec} (FP8, nearest-even) ===")
    print(C_scaled.detach().cpu().numpy())
    print()

    # Hex dump of scaled outputs in scaled_spec (FP8)
    try:
        C_scaled_codes, C_scaled_bits = tensor_to_custom_fp_codes(C_scaled, scaled_spec)
        C_scaled_hex = codes_to_hex_rows(C_scaled_codes, C_scaled_bits)
        print(f"=== C_scaled hex encoding ({scaled_spec}) ===")
        for row in C_scaled_hex:
            print(" ".join(row))
        print()
    except ValueError as e:
        print(f"[WARN] Could not hex-encode scaled outputs for {scaled_spec}: {e}")
        C_scaled_codes = []
        C_scaled_bits = 0
        print()

    # ------------------------------------------------------------------
    # Error metrics (on C_quant; you can add for C_scaled if needed)
    # ------------------------------------------------------------------
    metrics_vs_full   = matmul_loss(C_ref_full,  C_quant)
    metrics_vs_in     = matmul_loss(C_ref_in,    C_quant)
    metrics_in_vs_full = matmul_loss(C_ref_full, C_ref_in)

    print("=== Error metrics: C_quant vs C_ref_full (FP32 inputs) ===")
    for k, v in metrics_vs_full.items():
        print(f"{k}: {v:.6e}")
    print()

    print("=== Error metrics: C_quant vs C_ref_in (input-quantized, ideal MAC) ===")
    for k, v in metrics_vs_in.items():
        print(f"{k}: {v:.6e}")
    print()

    print("=== Error metrics: C_ref_in vs C_ref_full (pure input quantization error) ===")
    for k, v in metrics_in_vs_full.items():
        print(f"{k}: {v:.6e}")
    print()

    # ------------------------------------------------------------------
    # Write C header with hex-encoded A_in, B_in, C_quant, S_q, C_scaled
    # ------------------------------------------------------------------
    if A_codes and B_codes and C_codes and S_codes and C_scaled_codes:
        write_c_header(
            path=header_path,
            M=M,
            K=K,
            N=N,
            input_spec=input_spec,
            acc_spec=acc_spec,
            scale_spec=scale_spec,
            scaled_spec=scaled_spec,
            A_codes=A_codes,
            A_bits=A_bits,
            B_codes=B_codes,
            B_bits=B_bits,
            C_codes=C_codes,
            C_bits=C_bits,
            S_codes=S_codes,
            S_bits=S_bits,
            C_scaled_codes=C_scaled_codes,
            C_scaled_bits=C_scaled_bits,
        )
        print(f"Header written to: {header_path}")
    else:
        print("[WARN] Header not written because some codes/bits are missing (see warnings above).")

def trace_dot(
    A_in: Tensor,
    B_in: Tensor,
    i: int,
    j: int,
    *,
    prod_quant: QuantFn = None,
    acc_quant: QuantFn = None,
    cast_each_add: bool = True,   # if True: acc_quant after every add; else only at end
    cast_prod: bool = True,       # if True: apply prod_quant to each product
    verbose: bool = True,
) -> Tensor:
    """
    Trace C[i,j] = sum_k A[i,k]*B[k,j] with optional quant at product and/or accumulator.

    Returns final scalar tensor.
    """
    K = A_in.shape[1]
    acc = torch.zeros((), dtype=torch.float32, device=A_in.device)

    if verbose:
        print(f"Tracing C[{i},{j}] over K={K}")
        print(f"  cast_prod={cast_prod}, cast_each_add={cast_each_add}")
        print(f"  prod_quant={'yes' if prod_quant else 'no'}, acc_quant={'yes' if acc_quant else 'no'}")
        print()

    for k in range(K):
        a = A_in[i, k]
        b = B_in[k, j]
        p = a * b

        p_q = prod_quant(p) if (cast_prod and prod_quant) else p
        acc_next = acc + p_q
        acc_q = acc_quant(acc_next) if (cast_each_add and acc_quant) else acc_next

        if verbose:
            # .item() is fine for scalars
            print(
                f"k={k:2d}  a={a.item(): .10g}  b={b.item(): .10g}  "
                f"p={p.item(): .10g}  p_q={p_q.item(): .10g}  "
                f"acc-> {acc_next.item(): .10g}  acc_q={acc_q.item(): .10g}"
            )

        acc = acc_q

    if (not cast_each_add) and acc_quant:
        acc = acc_quant(acc)
        if verbose:
            print("\nfinal acc quant:", acc.item())

    return acc

# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="fp8:e4m3", help="Input storage precision spec")
    parser.add_argument("--prod",  type=str, default="fp8:e4m3", help="Product precision spec")
    parser.add_argument("--acc",   type=str, default="bf16",     help="Accumulator precision spec")

    parser.add_argument(
        "--scaled-spec",
        type=str,
        default=None,
        help="Post-scale FP8 spec (defaults to --prod if not set)."
    )

    parser.add_argument(
        "--input-rounding",
        type=str,
        default="zero",
        choices=["nearest", "nearest_even", "stochastic", "zero", "toward_zero", "trunc", "rtz"],
        help="Rounding mode for input quantization."
    )
    parser.add_argument(
        "--prod-rounding",
        type=str,
        default="zero",
        choices=["nearest", "nearest_even", "stochastic", "zero", "toward_zero", "trunc", "rtz"],
        help="Rounding mode for product."
    )
    parser.add_argument(
        "--acc-rounding",
        type=str,
        default="nearest",
        choices=["nearest", "nearest_even", "stochastic", "zero", "toward_zero", "trunc", "rtz", "q_bf16_rne"],
        help="Rounding mode for accumulator."
    )

    parser.add_argument(
        "--scale-spec",
        type=str,
        default="fpe8m0",
        help="Scaling factor FP spec (e.g., fpe8m0 for fpE8M0)."
    )
    parser.add_argument(
        "--scale-exp",
        type=int,
        default=0,
        help="Exponent for scaling factor: scale = 2**scale_exp."
    )

    parser.add_argument("--M", type=int, default=4)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--N", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument(
        "--header-path",
        type=str,
        default="matmul_data.h",
        help="Output C header file path."
    )

    parser.add_argument(
        "--prod-mant-bits",
        type=int,
        default=None,
        help="If set, truncate product mantissa to this many fraction bits (RTZ)."
    )

    args = parser.parse_args()

    run_experiment(
        M=args.M,
        K=args.K,
        N=args.N,
        input_spec=args.input,
        prod_spec=args.prod,
        acc_spec=args.acc,
        scaled_spec=args.scaled_spec,
        input_rounding=args.input_rounding,
        prod_rounding=args.prod_rounding,
        acc_rounding=args.acc_rounding,
        scale_spec=args.scale_spec,
        scale_exp=args.scale_exp,
        header_path=args.header_path,
        seed=args.seed,
        device=args.device,
    )
