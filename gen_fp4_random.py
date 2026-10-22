#!/usr/bin/env python3
"""FP4 goldens without the TinyLlama capture (npu-exploration/MXQuant/end_to_end_linear absent): runs gen_matmul_llama
unchanged except for two hooks -- seeded random operands instead of load_pair, and a local MX block quantizer (block
scale 2^floor(log2 amax) over 32 values, FP4 E2M1 codes) instead of MXQuant's quantize_mx_block32. The mesh model, the
output requant (fp4_matmul_model.matrix_mx_requantize) and the header writer are gen_matmul_llama's, so the header
layout matches the existing fp4 tests. Self-consistent: the codes/scales written are the ones the golden C uses.

  python3 gen_fp4_random.py fp4_64x64_dim4
"""
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
os.chdir(here)
sys.path.insert(0, here)

import numpy as np
import torch
import gen_matmul_llama as g


def load_pair(shape):
    rng = np.random.default_rng(0)
    return (rng.standard_normal((shape.M, shape.K)).astype(np.float32),
            rng.standard_normal((shape.K, shape.N)).astype(np.float32))


def quantize(V, *, axis, f, pmax_shift=0):
    assert f.bits == 4 and pmax_shift == 0, "gen_fp4_random: FP4 operands only"
    mm = __import__(f.model)
    Vb = V if axis == "row" else V.T                     # 32-value blocks along the last axis
    R, C = Vb.shape
    blk = Vb.reshape(R, C // 32, 32)
    amax = np.maximum(np.abs(blk).max(axis=2), np.finfo(np.float32).eps)
    X = np.exp2(np.floor(np.log2(amax))).astype(np.float32)          # [R][C/32]
    q = mm.make_fp_quantizer(f.name, rounding="nearest")
    P = q(torch.from_numpy((blk / X[:, :, None]).reshape(R, C).astype(np.float32))).numpy().astype(np.float32)
    if axis == "col":
        P, X = P.T, X.T
    P = np.ascontiguousarray(P)
    raw, _ = mm.tensor_to_custom_fp_codes(torch.from_numpy(P), f.name)
    return np.array(raw, dtype=np.uint8), g.e8m0_encode_exact(np.ascontiguousarray(X)), P


g.load_pair = load_pair
g.quantize = quantize
sys.exit(g.main())
