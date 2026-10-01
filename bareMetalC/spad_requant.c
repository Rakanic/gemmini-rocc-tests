// SPAD_REQUANT test (MxE4M3VpuGemminiRocketConfig / Spike): a row-major BF16 tile mvin'd into the scratchpad is
// requantized in place by the existing requantizer into E4M3 codes (flat, then operand-A tiled) + E8M0 scales in
// DRAM; codes and scales are checked bit-exact against include/mx_e4m3_ref.h. No fences between mvin, requant and
// mvout (ordering is the hardware's).
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "include/gemmini_testutils.h"
#include "include/mx_e4m3_ref.h"

#if !defined(MX_ROCKET) && !defined(SPIKE_SIM)
int main() { printf("skipped: SPAD_REQUANT config / Spike-only test\n"); return 0; }
#else

#define M 32
#define N 64
#define GN (N / 32)
#define SP_SRC   0x0000   // bank 0: M*N/8 = 256 rows
#define SP_FLAT  0x1000   // bank 1: M*N/16 = 128 rows
#define SP_TILED 0x2000   // bank 2: (M/16)*(N/16)*16 = 128 rows

static uint16_t X[M][N] __attribute__((aligned(64)));
static uint8_t codes_hw[M * N] __attribute__((aligned(64)));
static uint8_t scales_hw[M * GN] __attribute__((aligned(64)));
static uint8_t scales_hw2[M * GN] __attribute__((aligned(64)));
static uint8_t codes_ref[M][N], scales_ref[M][GN];

static uint32_t lcg = 4242;
static uint32_t rnd(void) { lcg = lcg * 1103515245u + 12345u; return lcg >> 8; }

static int check_scales(const char *name, const uint8_t *hw) {
  int bad = 0;
  for (int m = 0; m < M; m++)
    for (int b = 0; b < GN; b++)
      if (hw[m * GN + b] != scales_ref[m][b] && bad++ < 4)
        printf("  %s scale[%d][%d]: hw %02x ref %02x\n", name, m, b, hw[m * GN + b], scales_ref[m][b]);
  return bad;
}

int main() {
  // every 32-value block gets its own magnitude range; block (0,0) is all zero, (1,1) mixes in tiny values
  for (int m = 0; m < M; m++)
    for (int b = 0; b < GN; b++) {
      int ebase = 100 + (int)(rnd() % 50);
      for (int k = 0; k < 32; k++) {
        int e = ebase - (int)(rnd() % ((m == 1 && b == 1) ? 20 : 6));
        X[m][32 * b + k] = (uint16_t)(((rnd() & 1) << 15) | (e << 7) | (rnd() & 0x7f));
        if (m == 0 && b == 0) X[m][32 * b + k] = 0;
      }
    }
  for (int m = 0; m < M; m++)
    for (int b = 0; b < GN; b++) mxr_quant_block(&X[m][32 * b], &codes_ref[m][32 * b], &scales_ref[m][b]);

  gemmini_flush(0);
  gemmini_config_ld(DIM);
  gemmini_config_st(DIM);
  memset(scales_hw, 0xa5, sizeof(scales_hw));
  memset(scales_hw2, 0xa5, sizeof(scales_hw2));

  // ---- flat ----
  for (int r = 0; r < M * N / 8; r += DIM) gemmini_extended_mvin((uint8_t *)X + r * DIM, SP_SRC + r, DIM, DIM);
  uint64_t t0 = read_cycles();
  gemmini_spad_requant(SP_FLAT, SP_SRC, M, N, 0, (uint64_t)scales_hw, 0);
  memset(codes_hw, 0xa5, sizeof(codes_hw));
  for (int r = 0; r < M * N / 16; r += DIM) gemmini_extended_mvout(codes_hw + r * DIM, SP_FLAT + r, DIM, DIM);
  gemmini_fence();
  uint64_t t1 = read_cycles();
  int bad_flat = 0;
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++)
      if (codes_hw[m * N + n] != codes_ref[m][n] && bad_flat++ < 4)
        printf("  flat code[%d][%d]: hw %02x ref %02x\n", m, n, codes_hw[m * N + n], codes_ref[m][n]);
  int bad_sc = check_scales("flat", scales_hw);

  // ---- operand-A tiled: tile (i, kt) row r at SP_TILED + (i*N/16 + kt)*16 + r ----
  gemmini_spad_requant(SP_TILED, SP_SRC, M, N, 1, (uint64_t)scales_hw2, 0);
  memset(codes_hw, 0xa5, sizeof(codes_hw));
  for (int r = 0; r < M * N / 16; r += DIM) gemmini_extended_mvout(codes_hw + r * DIM, SP_TILED + r, DIM, DIM);
  gemmini_fence();
  int bad_tiled = 0;
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++) {
      int row = ((m / 16) * (N / 16) + n / 16) * 16 + m % 16;
      uint8_t got = codes_hw[row * DIM + n % 16];
      if (got != codes_ref[m][n] && bad_tiled++ < 4)
        printf("  tiled code[%d][%d]: hw %02x ref %02x\n", m, n, got, codes_ref[m][n]);
    }
  int bad_sc2 = check_scales("tiled", scales_hw2);

  printf("spad_requant %dx%d: flat %d, tiled %d code mismatches; scales %d, %d mismatches; %llu cycles (requant+mvout)\n",
         M, N, bad_flat, bad_tiled, bad_sc, bad_sc2, (unsigned long long)(t1 - t0));
  int fail = bad_flat || bad_tiled || bad_sc || bad_sc2;
  printf("spad_requant %s\n", fail ? "FAILED" : "PASSED");
  return fail;
}
#endif
