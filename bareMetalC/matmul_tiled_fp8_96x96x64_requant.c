#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifndef BAREMETAL
#include <sys/mman.h>
#include <stdlib.h>
#endif

#include "include/gemmini_testutils.h"
#include "include/matmul_fp8_96x96x64.h"

#define GEMMINI_SF_MEM 0x40088000
#define GEMMINI_SF_MEM_A (GEMMINI_SF_MEM + 0x2000)
#define GEMMINI_SF_MEM_B GEMMINI_SF_MEM
#define SMEM 0x40000000

#define DIM 16

#define GEMMINI_CTRL 0x40084000
#define GEMMINI_RS1_ADDR (GEMMINI_CTRL + 0x10)
#define GEMMINI_RS2_ADDR (GEMMINI_CTRL + 0x18)
#define GEMMINI_INST_ADDR (GEMMINI_CTRL + 0x0)

// MX_ROCKET is a real RoCC; keep gemmini.h's direct macro (0x40084xxx unbacked on standalone).
#if !defined(SPIKE_SIM) && !defined(MX_ROCKET)
#undef ROCC_INSTRUCTION_RS1_RS2
#define ROCC_INSTRUCTION_RS1_RS2(x, rs1, rs2, funct) { \
    *((volatile uint64_t *) GEMMINI_RS1_ADDR) = (rs1); \
    *((volatile uint64_t *) GEMMINI_RS2_ADDR) = (rs2); \
    *((volatile uint32_t*) GEMMINI_INST_ADDR) = (0x7B) | (0 << 7) | (3 << 12) | (1 << 15) | (2 << 20) | ((funct) << 25); \
}
#endif

#define ADDR_LEN 32

typedef uint8_t  elem_t;   // fp8:e4m3 input elements
typedef uint8_t  welem_t;  // fp8:e4m3 weight elements
typedef uint64_t out_t;    // 8x fp8 packed per word

// ---- Scale factor loader ----
void load_scale_factors(volatile uint64_t *sf_mem, uint8_t *scale_factors, int INDIM, int K) {
  for (size_t k = 0; k < K/32; k++) {
    for (size_t i = 0; i < INDIM / 8; i++) {
        sf_mem[k*INDIM/8 + i] = ((uint64_t*) scale_factors)[k * INDIM/8 + i];
    }
  }
}

static inline int popcount8(uint8_t x) {
  int count = 0;
  while (x) {
    count += x & 1;
    x >>= 1;
  }
  return count;
}

int main() {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    perror("mlockall");
    return 1;
  }
#endif

  // ---- Output buffer ----
  static out_t C_hw[MATMUL_M][MATMUL_N / 8];
  uint32_t scale_factors[512] __attribute__((aligned(32))) = {0};
  memset(C_hw, 0, sizeof(C_hw));

  // ---- Tile dimensions ----
  int tiles_I = MATMUL_M / DIM;
  int tiles_J = MATMUL_N / DIM;
  int tiles_K = MATMUL_K / DIM;

  uint32_t a_base = 0;
  uint32_t b_base = BANK_NUM * BANK_ROWS - tiles_K * tiles_J * DIM;
  uint32_t acc_addr = (1u << (ADDR_LEN - 1));

  // ---- Gemmini setup ----
  gemmini_flush(0);
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0, 0, 0);

#if defined(SPIKE_SIM) || defined(MX_ROCKET)
  // Unified real-RoCC path (Spike AND RTL): funct-27 MX_LOAD_SCALES. The fence orders the async
  // scale DMA on the RTL and is a no-op on Spike, so both emit the identical instruction stream.
  gemmini_mx_load_scales((uint64_t)&A_scales_row, sizeof(A_scales_row), 0);
  gemmini_mx_load_scales((uint64_t)&B_scales_col, sizeof(B_scales_col), 1);
  gemmini_fence();
#else
  load_scale_factors((volatile uint64_t *) GEMMINI_SF_MEM_A, (uint8_t *) &A_scales_row, MATMUL_M, MATMUL_K);
  load_scale_factors((volatile uint64_t *) GEMMINI_SF_MEM_B, (uint8_t *) &B_scales_col, MATMUL_N, MATMUL_K);
#endif

  // ---- MVIN A: tile (i,k) -> a_base + (i*tiles_K + k)*DIM ----
  gemmini_config_ld(MATMUL_K * sizeof(elem_t));

  for (int i = 0; i < tiles_I; i++) {
    for (int k = 0; k < tiles_K; k++) {
      elem_t *dram_ptr = ((elem_t*)A_in) + i * DIM * MATMUL_K + k * DIM;
      uint32_t sp_addr = a_base + (i * tiles_K + k) * DIM;
      gemmini_extended_mvin((void *) dram_ptr, sp_addr, DIM, DIM);
    }
  }

  // ---- MVIN B: tile (k,j) -> b_base + (k*tiles_J + j)*DIM ----
  gemmini_config_ld(MATMUL_N * sizeof(elem_t));

  for (int k = 0; k < tiles_K; k++) {
    for (int j = 0; j < tiles_J; j++) {
      elem_t *dram_ptr = ((elem_t*)B_in) + k * DIM * MATMUL_N + j * DIM;
      uint32_t sp_addr = b_base + (k * tiles_J + j) * DIM;
      gemmini_extended_mvin((void *) dram_ptr, sp_addr, DIM, DIM);
    }
  }

  int SPAD_DEST = 128;

  gemmini_config_st(1 * sizeof(out_t));
  gemmini_mxquant_config_mvout((uint64_t)scale_factors, tiles_I, tiles_J, tiles_K, 0, 0, 1);

  // ---- Compute ----
  gemmini_loop_ws_spad(
      tiles_I, tiles_J, tiles_K,
      0, 0, 0,
      a_base,
      BANK_NUM * BANK_ROWS,
      0,
      SPAD_DEST,
      false, false,
      false, false, false,
      NO_ACTIVATION,
      0, 0,
      false,
      0x38);

#if defined(SPIKE_SIM) || defined(MX_ROCKET)
  // V1: requant FP8 output lives in the INTERNAL scratchpad; read it back with a plain full-width
  // spad->DRAM mvout (FP8 byte-aligned, NOT chunked). Flat row-major-contiguous from SPAD_DEST as
  // proven at 64x64/128x128 requant. N=96 -> 576 spad rows. If the partial 64-block breaks the flat
  // contiguity, switch to the loop-store spad dst geometry (LoopMatmul dst_offset) -- verify in sim.
  gemmini_fence();
  gemmini_config_st(DIM * sizeof(uint8_t));               // one spad row = DIM (16) FP8 bytes
  uint8_t *c_base = (uint8_t *) C_hw;
  int total_spad_rows = MATMUL_M * MATMUL_N / DIM;         // 96*96/16 = 576 spad rows
  for (int r = 0; r < total_spad_rows; r += DIM) {
    gemmini_extended_mvout(c_base + r * DIM, SPAD_DEST + r, DIM, DIM);  // DIM rows x DIM FP8
  }
  gemmini_fence();
#else
  uint64_t* smem_start_addr = ((uint64_t*)SMEM) + SPAD_DEST * 2;
  printf("Address: %p \n", smem_start_addr);
  for (int i = 0; i < MATMUL_M; i++) {
    for (int j = 0; j < MATMUL_N / 8; j++) {
      C_hw[i][j] = *(smem_start_addr + (i * MATMUL_N / 8 + j));
    }
  }
#endif

  gemmini_fence();

  // ---- Elementwise check against C_out (fp8, byte-by-byte) ----
  int errors = 0;
  int diff1 = 0, diff2 = 0, diff3plus = 0;
  uint8_t *hw_bytes = (uint8_t *)C_hw;

  for (int i = 0; i < MATMUL_M; i++) {
    for (int j = 0; j < MATMUL_N; j++) {
      uint8_t got = hw_bytes[i * MATMUL_N + j];
      uint8_t exp = C_out[i][j];
      if (got != exp) {
        errors++;
        printf("Output[%d][%d], Got: %x, Exp: %x\n", i, j, got, exp);
        int bits = popcount8(got ^ exp);
        if      (bits == 1) diff1++;
        else if (bits == 2) diff2++;
        else                diff3plus++;
      }
    }
  }

  if (errors == 0) {
    printf("fp8 WS matmul test PASSED (no mismatches).\n");
  } else {
    printf("fp8 WS matmul test FAILED with %d mismatches.\n", errors);
    printf("  differ by 1 bit:   %d\n", diff1);
    printf("  differ by 2 bits:  %d\n", diff2);
    printf("  differ by 3+ bits: %d\n", diff3plus);
  }

#ifndef BAREMETAL
  exit(errors != 0);
#else
  return errors != 0;
#endif
}
