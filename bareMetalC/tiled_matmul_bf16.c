// See LICENSE for license details.

// BF16 tiled matmul for the plain BF16 Gemmini (BF16Dim*BaselineConfig, gemmini_params_bf16_dim*_vpu.h):
// C = A * B + D with BF16 A, B, C and an FP32 bias D, checked bit-exactly against a CPU model of the hardware:
//   - each mesh column sums one DIM-long K chunk through BF16 FMA PEs (hardfloat MulAddRecFN, one RNE rounding to BF16
//     per PE, k ascending, starting from +0),
//   - the FP32 accumulator adds the chunks in K order on top of D,
//   - mvout rounds the FP32 accumulator to BF16 (RNE).
// elem_t is the raw BF16 bit pattern (uint16_t) in the generated header; acc_t is float.

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"

#if !defined(ELEM_T_IS_FLOAT) || ELEM_T_EXP_BITS != 8 || ELEM_T_SIG_BITS != 8
#error "tiled_matmul_bf16 needs a BF16 Gemmini header (gemmini_params_bf16_dim*_vpu.h)"
#endif

// RTL sim: the CPU reference (software BF16 FMAs, I*J*K of them) dominates the run, so keep it small while still
// covering two K chunks into the accumulator and, at DIM 16 / 32, a partial row tile. Spike runs the larger problem.
#ifdef SPIKE_SIM
#define MAT_DIM_I 64
#define MAT_DIM_K 128
#define MAT_DIM_J 64
#else
#define MAT_DIM_I 8
#define MAT_DIM_K (2 * DIM)
#define MAT_DIM_J DIM
#endif

#define MAX_PRINTED_ERRORS 10

static uint32_t f32_bits(float f) { uint32_t u; memcpy(&u, &f, sizeof(u)); return u; }
static float f32_from_bits(uint32_t u) { float f; memcpy(&f, &u, sizeof(f)); return f; }
static float bf16_to_f32(uint16_t b) { return f32_from_bits((uint32_t)b << 16); }

// RNE FP32 -> BF16, keeping -0 (as hardfloat's RecFNToRecFN)
static uint16_t f32_to_bf16(float f) {
  const uint32_t u = f32_bits(f);
  if ((u & 0x7F800000u) == 0x7F800000u && (u & 0x7FFFFFu)) return (uint16_t)((u >> 16) | 0x40);
  return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1)) >> 16);
}

static int msb64(uint64_t x) { int m = -1; while (x) { x >>= 1; m++; } return m; }

// Exact a*b + c for BF16 a, b, c, rounded once to BF16 (RNE). Inputs are finite here (the test data has no Inf/NaN).
static uint16_t bf16_fma(uint16_t a, uint16_t b, uint16_t c) {
  // value = (-1)^s * m * 2^e, m an integer significand (hidden bit included for normals)
  const int sa = a >> 15, sb = b >> 15, sc = c >> 15;
  const int ea = (a >> 7) & 0xFF, eb = (b >> 7) & 0xFF, ec = (c >> 7) & 0xFF;
  const uint64_t ma = ea ? (0x80 | (a & 0x7F)) : (a & 0x7F);
  const uint64_t mb = eb ? (0x80 | (b & 0x7F)) : (b & 0x7F);
  const uint64_t mc = ec ? (0x80 | (c & 0x7F)) : (c & 0x7F);
  const int sp = sa ^ sb;
  uint64_t mp = ma * mb;
  int ep = (ea ? ea : 1) + (eb ? eb : 1) - 2 * (127 + 7);
  const int e_c = (ec ? ec : 1) - (127 + 7);

  if (mp == 0 && mc == 0) return (uint16_t)((sp & sc) << 15);   // +0 unless both are -0 (RNE)
  if (mp == 0) return c;

  // align: the operand with the higher LSB exponent is shifted left (exact up to 40 bits); a farther-away operand only
  // matters as a sticky bit, so it is replaced by 1 unit below the shifted operand's LSB
  uint64_t mx, my; int sx, sy, ex, ey;
  if (mc == 0 || (mp != 0 && ep >= e_c)) { mx = mp; sx = sp; ex = ep; my = mc; sy = sc; ey = e_c; }
  else { mx = mc; sx = sc; ex = e_c; my = mp; sy = sp; ey = ep; }
  int e;
  if (my == 0) { e = ex; my = 0; }
  else if (ex - ey <= 40) { mx <<= (ex - ey); e = ey; }
  else { mx <<= 40; my = 1; e = ex - 40; }

  int64_t s = (sx ? -(int64_t)mx : (int64_t)mx) + (sy ? -(int64_t)my : (int64_t)my);
  if (s == 0) return 0;   // exact cancellation is +0 under RNE
  const int sign = s < 0;
  uint64_t m = sign ? (uint64_t)(-s) : (uint64_t)s;

  // keep 8 significant bits (fewer for subnormals: the LSB exponent may not go below 2^-133)
  const int top = msb64(m);                 // value = m * 2^e, leading bit at 2^(top + e)
  int lsb_exp = top + e - 7;
  if (lsb_exp < -133) lsb_exp = -133;
  const int shift = lsb_exp - e;
  uint64_t q;
  if (shift <= 0) {
    q = m << (-shift);
  } else if (shift > 63) {
    q = 0;   // far below the smallest subnormal: rounds to zero (half-way impossible here)
  } else {
    q = m >> shift;
    const uint64_t rem = m & ((1ull << shift) - 1);
    const uint64_t half = 1ull << (shift - 1);
    if (rem > half || (rem == half && (q & 1))) q++;
  }
  // q * 2^lsb_exp, q <= 0x100 (a carry out of 8 bits moves to the next binade)
  int exp_field;
  uint32_t frac;
  if (q == 0) return (uint16_t)(sign << 15);
  if (q >= 0x100) { q >>= 1; lsb_exp++; }
  if (q & 0x80) {
    exp_field = lsb_exp + 7 + 127;
    frac = q & 0x7F;
    if (exp_field >= 0xFF) return (uint16_t)((sign << 15) | 0x7F80);   // overflow to Inf
  } else {
    exp_field = 0;   // subnormal (lsb_exp == -133)
    frac = q & 0x7F;
  }
  return (uint16_t)((sign << 15) | (exp_field << 7) | frac);
}

static uint32_t lcg_state = 12345;
static uint32_t lcg(void) { lcg_state = lcg_state * 1664525u + 1013904223u; return lcg_state >> 8; }

// random BF16 in roughly [-16, 16]: random sign, exponent 2^-4..2^3, random 7-bit fraction
static uint16_t rand_bf16(void) {
  const uint32_t r = lcg();
  const uint32_t sign = r & 1;
  const uint32_t exp = 127 - 4 + ((r >> 1) % 8);
  const uint32_t frac = (r >> 4) & 0x7F;
  return (uint16_t)((sign << 15) | (exp << 7) | frac);
}

int main() {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    perror("mlockall failed");
    exit(1);
  }
#endif
  printf("BF16 tiled matmul, DIM %d: I %d, J %d, K %d\n", DIM, MAT_DIM_I, MAT_DIM_J, MAT_DIM_K);
  gemmini_flush(0);

  static elem_t A[MAT_DIM_I][MAT_DIM_K] row_align(1);
  static elem_t B[MAT_DIM_K][MAT_DIM_J] row_align(1);
  static elem_t C[MAT_DIM_I][MAT_DIM_J] row_align(1);
  static acc_t D[MAT_DIM_I][MAT_DIM_J] row_align_acc(1);
  static elem_t gold[MAT_DIM_I][MAT_DIM_J];

  for (size_t i = 0; i < MAT_DIM_I; i++)
    for (size_t k = 0; k < MAT_DIM_K; k++)
      A[i][k] = rand_bf16();
  for (size_t k = 0; k < MAT_DIM_K; k++)
    for (size_t j = 0; j < MAT_DIM_J; j++)
      B[k][j] = rand_bf16();
  for (size_t i = 0; i < MAT_DIM_I; i++)
    for (size_t j = 0; j < MAT_DIM_J; j++)
      D[i][j] = bf16_to_f32(rand_bf16());

  printf("Starting gemmini matmul\n");
  const unsigned long start = read_cycles();
  tiled_matmul_auto(MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
          (elem_t*)A, (elem_t*)B, &D[0][0], (elem_t*)C,
          MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
          MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
          NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
          false, false,
          false, false,
          0,
          WS);
  gemmini_fence();
  const unsigned long end = read_cycles();
  printf("Cycles taken: %lu\n", end - start);

  printf("Starting CPU reference\n");
  for (size_t i = 0; i < MAT_DIM_I; i++)
    for (size_t j = 0; j < MAT_DIM_J; j++) {
      float acc = D[i][j];
      for (size_t k0 = 0; k0 < MAT_DIM_K; k0 += DIM) {
        uint16_t s = 0;
        for (size_t k = k0; k < k0 + DIM && k < MAT_DIM_K; k++)
          s = bf16_fma(A[i][k], B[k][j], s);
        acc = acc + bf16_to_f32(s);
      }
      gold[i][j] = f32_to_bf16(acc);
    }

  int errors = 0;
  for (size_t i = 0; i < MAT_DIM_I; i++)
    for (size_t j = 0; j < MAT_DIM_J; j++)
      if (C[i][j] != gold[i][j]) {
        if (errors < MAX_PRINTED_ERRORS)
          printf("MISMATCH [%d][%d]: got 0x%04x, expected 0x%04x\n", (int)i, (int)j, C[i][j], gold[i][j]);
        errors++;
      }
  if (errors) {
    printf("FAIL: %d of %d outputs differ\n", errors, MAT_DIM_I * MAT_DIM_J);
    exit(1);
  }

  printf("PASS\n");
  exit(0);
}
