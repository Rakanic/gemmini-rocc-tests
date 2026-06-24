#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifndef BAREMETAL
#include <sys/mman.h>
#include <stdlib.h>
#endif

#include "include/gemmini_testutils.h"
#include "include/matmul_fp8_64x64.h"
#ifdef MX_ROCKET
#include "include/gemmini_mx_rocket.h"   // standalone: direct RoCC, flat scale window, mvout output
#endif

#define GEMMINI_SF_MEM 0x40088000
#define GEMMINI_SF_MEM_A (GEMMINI_SF_MEM + 0x2000)
#define GEMMINI_SF_MEM_B GEMMINI_SF_MEM
#define SMEM 0x40000000

#define DIM 16

#define GEMMINI_CTRL 0x40084000
#define GEMMINI_RS1_ADDR (GEMMINI_CTRL + 0x10)
#define GEMMINI_RS2_ADDR (GEMMINI_CTRL + 0x18)
#define GEMMINI_INST_ADDR (GEMMINI_CTRL + 0x0)

// Radiance RTL drives gemmini via an MMIO command mimic. The standalone rocket config (MX_ROCKET)
// is a real RoCC, so keep gemmini.h's default direct-RoCC macro there.
#if !defined(SPIKE_SIM) && !defined(MX_ROCKET)
#undef ROCC_INSTRUCTION_RS1_RS2
#define ROCC_INSTRUCTION_RS1_RS2(x, rs1, rs2, funct) { \
    *((volatile uint64_t *) GEMMINI_RS1_ADDR) = (rs1); \
    *((volatile uint64_t *) GEMMINI_RS2_ADDR) = (rs2); \
    *((volatile uint32_t*) GEMMINI_INST_ADDR) = (0x7B) | (0 << 7) | (3 << 12) | (1 << 15) | (2 << 20) | ((funct) << 25); \
}
#endif

#define ADDR_LEN 32

// BF16 values packed 4 per uint64_t output word
#define BF16_PER_WORD 4
#define OUT_COLS (MATMUL_M / BF16_PER_WORD)

typedef uint8_t  elem_t;   // fp8:e4m3 input elements
typedef uint8_t  welem_t;  // fp8:e4m3 weight elements
typedef uint64_t out_t;    // 4x bf16 packed per word

// ---- Scale factor loader ----
void load_scale_factors(volatile uint64_t *sf_mem, uint8_t *scale_factors, int INDIM, int K) {
  for (size_t k = 0; k < K/32; k++) {
    for (size_t i = 0; i < INDIM / 8; i++) {
//        printf("loading: %lx\n", ((uint64_t*) scale_factors)[k * INDIM/8 + i]);
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
  static out_t C_hw[MATMUL_M][MATMUL_N/8];
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

#ifdef SPIKE_SIM
  gemmini_mx_load_scales((uint64_t)&A_scales_row, sizeof(A_scales_row), 0);
  gemmini_mx_load_scales((uint64_t)&B_scales_col, sizeof(B_scales_col), 1);
#elif !defined(MX_ROCKET)
  load_scale_factors((volatile uint64_t *) GEMMINI_SF_MEM_A, (uint8_t *) &A_scales_row, MATMUL_M, MATMUL_K);
  load_scale_factors((volatile uint64_t *) GEMMINI_SF_MEM_B, (uint8_t *) &B_scales_col, MATMUL_N, MATMUL_K);
#endif
#ifdef MX_ROCKET
  // Flat scale window (same loader as radiance, different base): A=activation, W=weight(B).
  load_scale_factors((volatile uint64_t *) MX_SCALE_A, (uint8_t *) &A_scales_row, MATMUL_M, MATMUL_K);
  load_scale_factors((volatile uint64_t *) MX_SCALE_W, (uint8_t *) &B_scales_col, MATMUL_N, MATMUL_K);
#endif

  // ---- MVIN A: tile (i,k) -> a_base + (i*tiles_K + k)*DIM ----
  gemmini_config_ld(MATMUL_M * sizeof(elem_t));

  for (int i = 0; i < tiles_I; i++) {
    for (int k = 0; k < tiles_K; k++) {
      elem_t *dram_ptr = ((elem_t*)A_in) + i * DIM * MATMUL_M + k * DIM;
      uint32_t sp_addr = a_base + (i * tiles_K + k) * DIM;
      gemmini_extended_mvin((void *) dram_ptr, sp_addr, DIM, DIM);
    }
  }

  // ---- MVIN B: tile (k,j) -> b_base + (j*tiles_K + k)*DIM ----
  for (int j = 0; j < tiles_J; j++) {
    for (int k = 0; k < tiles_K; k++) {
      elem_t *dram_ptr = ((elem_t*)B_in) + j * DIM * MATMUL_M + k * DIM;
      uint32_t sp_addr = b_base + (j * tiles_K + k) * DIM;
      gemmini_extended_mvin((void *) dram_ptr, sp_addr, DIM, DIM);
    }
  }

  int SPAD_DEST = 128;

  gemmini_config_st(1 * sizeof(out_t));
  gemmini_mxquant_config_mvout((uint64_t)scale_factors, tiles_I, tiles_J, tiles_K, 0, 0, 1);

  // Standalone (MX_ROCKET) has ex_write_to_spad=false, so the requantized output never reaches
  // the scratchpad; it lives in the accumulator. Compute into the accumulator and mvout from
  // there. flag 0xb8 skips the spad store; 0x38 keeps it for radiance/smem.
#ifdef MX_ROCKET
  uint32_t out_dest = acc_addr;
  uint32_t out_flag = 0xb8;
#else
  uint32_t out_dest = SPAD_DEST;
  uint32_t out_flag = 0x38;
#endif

  // ---- Compute ----
  gemmini_loop_ws_spad(
      tiles_I, tiles_J, tiles_K,
      0, 0, 0,
      a_base,
      BANK_NUM * BANK_ROWS,
      0,
      out_dest,
      false, false,
      false, false, false,
      NO_ACTIVATION,
      0, 0,
      false,
      out_flag);

//  for (int i = 0; i < tiles_I; i++) {
//    for (int j = 0; j < tiles_J; j++) {
//      uint32_t acc_tile_addr = acc_addr + (i * tiles_J + j) * DIM;
//      out_t *dram_ptr = &C_hw[i * DIM][j * DIM];
//      gemmini_mvout((void *) dram_ptr, acc_tile_addr);
//    }
//  }

#ifdef SPIKE_SIM
  gemmini_mx_read_smem(&C_hw[0][0], SPAD_DEST * 16, MATMUL_M * MATMUL_N / 2);
#elif defined(MX_ROCKET)
  // Standalone: no shared memory; the requantized FP8 output lives in the ACCUMULATOR
  // (ex_write_to_spad=false, so nothing lands in the scratchpad). Requant applies during the
  // acc->DRAM mvout: the MxRequantizer is on the accumulator read path (acc_scale_unit), gated
  // by enable_MXQuant (set above by gemmini_mxquant_config_mvout), not by the destination.
  // The acc read port is HALF-WIDTH: one mvout returns 32 FP8 = cols 0-31 ("chunk 0"); cols
  // 32-63 are the upper half of the SAME acc rows, selected by mx_chunk_id=1. The stock
  // gemmini_mvout macro hardwires mx_chunk_id=0, so emit one mvout per chunk with mx_chunk_id
  // set directly in MvoutRs2.
  //   MvoutRs2: num_rows @ bit 48 (mvout_rows_bits = log2Up(2*DIM+1) = 6 for DIM=16),
  //   mx_chunk_id (3b) just above it -> bit 48+6 = 54. config_st = full FP8 row stride.
  #define MX_CHUNK_SHIFT (48 + 6)
  gemmini_fence();
  gemmini_config_st(MATMUL_N * sizeof(uint8_t));
  for (int i = 0; i < tiles_I; i++) {
    for (int ck = 0; ck < 2; ck++) {   // 2 half-chunks: ck0 = cols 0-31, ck1 = cols 32-63
      uint32_t acc_tile_addr = acc_addr + i * DIM;
      uint8_t *dram_ptr = (uint8_t *) C_hw + (i * DIM) * MATMUL_N + ck * 32;  // ck*32 FP8 bytes
      ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, (uint64_t)(uintptr_t)dram_ptr,
          ((uint64_t)(DIM) << (ADDR_LEN + 16)) | ((uint64_t)(ck) << MX_CHUNK_SHIFT) |
          ((uint64_t)(DIM) << ADDR_LEN) | (uint64_t)(acc_tile_addr), k_MVOUT);
    }
  }
  gemmini_fence();
#else
  uint64_t* smem_start_addr = ((uint64_t*)SMEM) + SPAD_DEST * 2;
  printf("Address: %p \n", smem_start_addr);
  for (int i = 0; i < MATMUL_M; i ++) {
    for (int j = 0; j < MATMUL_N / 8; j++) {
        C_hw[i][j] = *(smem_start_addr + (i*MATMUL_N / 8 + j));
    }
  }
#endif


//  gemmini_mvout((void*)&C_hw[0][0], 128 )

  gemmini_fence();

  // ---- Debug print tile (0,0) ----
//  printf("=== Tile (0,0) - acc_addr=0x%08x ===\n", acc_addr);
//  for (int i = 0; i < DIM; i++) {
//    for (int j = 0; j < DIM; j++) {
//      printf("C_hw[%d][%d] = 0x%016lx\n", i, j, (unsigned long)C_hw[i][j]);
//    }
//  }

  // ---- Elementwise check against C_out_bf16 ----

// ---- Elementwise check against C_out (fp8, byte-by-byte) ----



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