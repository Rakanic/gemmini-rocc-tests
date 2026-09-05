"""Real TinyLlama operands + MX quantization + data-derived FP6 LUTs.

Used by `lut_mapping_demo.py` (the FP6 generator) when MXGEMMINI_LLAMA=1, so that its operands,
block scales and lookup tables all come from a real model instead of `torch.randn`, random
power-of-two scale exponents and `make_lut`'s random 16-entry subsets. Everything downstream of
this module -- index assignment, HW packing, the mesh model, the requant post-pass -- is untouched.

The LUT scheme follows MXQuant's `prodacc_bundle/lut_quantization.py` (branch chloe-branch-all),
which is itself a copy of `microxcaling/mx/level2_scratch.py`: MX-quantize first, so every value is
already a valid FP6 codebook entry, then reduce that codebook to `num_signposts=16` per channel by
1-D k-means. The hardware's granularity is per group of 2^G rows of A / columns of B
(`gemmini.cc:1321` `lut_idx = (i * TM + m) >> G`), so that is the grouping used here.

The k-means below is deterministic -- quantile init, Lloyd to convergence over the *distinct*
values weighted by count -- rather than MXQuant's random init, because a generated header must be
reproducible.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
NPU = HERE.parent.parent / "npu-exploration"
if str(NPU) not in sys.path:
    sys.path.insert(0, str(NPU))

from app.mxq_golden import MXQ_ROOT, e8m0_encode_exact          # noqa: E402
from end_to_end_linear.mx_block_quant import quantize_mx_block32  # noqa: E402

DATA = MXQ_ROOT / "end_to_end_linear" / "systolic_simulation" / "data_evalrun_512"


def load_pair(layer: str, proj: str, M: int, K: int, N: int):
    """Real ``(A[M][K], B[K][N])`` from one contiguous logged (A_square, W_square) pair.

    ``A_square`` is [tokens][in_features], ``W_square`` is [out_features][in_features], both at the
    same in-feature offset, so ``A @ B`` is a real sub-block of that projection's own output.
    """
    d = DATA / layer / proj
    if not d.is_dir():
        raise SystemExit(f"missing {d}\nRun: cd {NPU} && .venv/bin/python3 -m app.capture_llama_tiles")

    def tile(which):
        with np.load(d / f"{which}_square.npz") as z:
            return z["data"].astype(np.float32)

    A_full, W_full = tile("A"), tile("W")
    if A_full.shape[0] < M or A_full.shape[1] < K:
        raise SystemExit(f"{d}/A_square.npz is {A_full.shape}, need {(M, K)}")
    if W_full.shape[0] < N or W_full.shape[1] < K:
        raise SystemExit(f"{d}/W_square.npz is {W_full.shape}, need {(N, K)}")
    A = torch.from_numpy(np.ascontiguousarray(A_full[:M, :K]))
    B = torch.from_numpy(np.ascontiguousarray(W_full[:N, :K].T))
    return A, B


def mx_quantize(V: torch.Tensor, axis: str, fmt: str = "MXFP6_E3M2"):
    """``(P, scale_codes)``: MXQuant's normalized values and their E8M0 block-scale bytes."""
    out = quantize_mx_block32(V, fmt=fmt, axis=axis)
    return out.P.to(torch.float32), e8m0_encode_exact(out.X.numpy().astype(np.float32))


def _kmeans_1d(values: np.ndarray, k: int) -> np.ndarray:
    """Deterministic weighted 1-D k-means over the distinct values of ``values``.

    Returns up to ``k`` centroids. The support is tiny (FP6 has 64 codes), so this collapses to a
    weighted Lloyd over distinct values and converges in a handful of passes.
    """
    uniq, counts = np.unique(values, return_counts=True)
    if uniq.size <= k:
        return uniq
    # Quantile init over the weighted distribution: spreads seeds by mass, not by range, so a
    # cluster of near-zero values (which is most of a real activation block) is not under-served.
    cdf = np.cumsum(counts) / counts.sum()
    probes = (np.arange(k) + 0.5) / k
    centers = np.unique(uniq[np.searchsorted(cdf, probes).clip(0, uniq.size - 1)])
    for _ in range(50):
        lab = np.abs(uniq[:, None] - centers[None, :]).argmin(axis=1)
        new = np.array([
            (uniq[lab == c] * counts[lab == c]).sum() / counts[lab == c].sum()
            if np.any(lab == c) else centers[c]
            for c in range(centers.size)])
        new = np.unique(new)
        if new.size == centers.size and np.allclose(new, centers):
            break
        centers = new
    return centers


def build_luts(P: torch.Tensor, *, axis: str, G: int, lut_size: int,
               codebook: torch.Tensor) -> list[torch.Tensor]:
    """One ``lut_size``-entry LUT per group of ``2**G`` rows (axis="row") or columns (axis="col").

    ``P`` is already MX-quantized, so its values are exact FP6 codebook entries; the k-means
    centroids are snapped back onto ``codebook`` so every LUT entry stays a representable code.
    Entries are padded to exactly ``lut_size`` with unused codes nearest the group's range, so no
    slot is a duplicate that could make the index assignment ambiguous.
    """
    p = P.numpy()
    n_groups = (p.shape[0] if axis == "row" else p.shape[1]) >> G
    cb = codebook.numpy()
    luts = []
    for g in range(n_groups):
        sl = slice(g << G, (g + 1) << G)
        vals = (p[sl, :] if axis == "row" else p[:, sl]).ravel()
        centers = _kmeans_1d(vals, lut_size)
        snapped = cb[np.abs(centers[:, None] - cb[None, :]).argmin(axis=1)]
        entries = list(dict.fromkeys(snapped.tolist()))          # order-preserving dedupe
        if len(entries) < lut_size:                              # pad with unused codes
            for c in sorted(cb.tolist(), key=lambda v: abs(v)):
                if c not in entries:
                    entries.append(c)
                if len(entries) == lut_size:
                    break
        luts.append(torch.tensor(sorted(entries[:lut_size]), dtype=torch.float32))
    return luts


def fp6_codebook() -> torch.Tensor:
    """Every finite FP6 E3M2 value, decoded from its 6-bit code (mx_fp_math.h:274)."""
    vals = []
    for code in range(64):
        s, e, m = (code >> 5) & 1, (code >> 2) & 0x7, code & 0x3
        v = (m * 0.0625) if e == 0 else (1.0 + m * 0.25) * 2.0 ** (e - 3)
        vals.append(-v if s else v)
    return torch.tensor(sorted(set(vals)), dtype=torch.float32)
