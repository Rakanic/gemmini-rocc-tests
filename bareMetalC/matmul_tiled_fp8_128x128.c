#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifndef BAREMETAL
#include <sys/mman.h>
#include <stdlib.h>
#endif

#include "include/gemmini_testutils.h"
#include "include/matmul_fp8_128x128.h"
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
#define GEMMINI_BUSY_ADDR (GEMMINI_CTRL + 0x20)

// Radiance drives gemmini via an MMIO command mimic; MX_ROCKET is a real RoCC, so keep gemmini.h's
// default direct-RoCC macro there.
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

// Spin until Gemmini reports not busy
static inline void gemmini_poll_until_ready() {
    volatile uint32_t *status = (volatile uint32_t*)(GEMMINI_BUSY_ADDR); // check your offset
    while (*status & 0x1) {
        // busy — keep polling
    }
}

int main() {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    perror("mlockall");
    return 1;
  }
#endif

  // ---- Output buffer ----
  static out_t C_hw[MATMUL_M][OUT_COLS];
  uint32_t scale_factors[512] = {0};
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
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0, 3, 0);

#ifdef SPIKE_SIM
  gemmini_mx_load_scales((uint64_t)&A_scales_row, sizeof(A_scales_row), 0);
  gemmini_mx_load_scales((uint64_t)&B_scales_col, sizeof(B_scales_col), 1);
#elif !defined(MX_ROCKET)
  load_scale_factors((volatile uint64_t *) GEMMINI_SF_MEM_A, (uint8_t *) &A_scales_row, MATMUL_M, MATMUL_K);
  load_scale_factors((volatile uint64_t *) GEMMINI_SF_MEM_B, (uint8_t *) &B_scales_col, MATMUL_N, MATMUL_K);
#endif
#ifdef MX_ROCKET
  // ISA parity: funct-27 MX_LOAD_SCALES DMA loader (same call as SPIKE_SIM), not the flat window.
  gemmini_mx_load_scales((uint64_t)&A_scales_row, sizeof(A_scales_row), 0);
  gemmini_mx_load_scales((uint64_t)&B_scales_col, sizeof(B_scales_col), 1);
  gemmini_fence();
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

  gemmini_config_st(OUT_COLS * sizeof(out_t));
  gemmini_mxquant_config_mvout((uint64_t)scale_factors, tiles_I, tiles_J, tiles_K, 0, 0, 1);

  // Standalone (MX_ROCKET) has ex_write_to_spad=false, so the BF16 mesh output never reaches the
  // scratchpad; it lives in the accumulator. Compute into the accumulator and mvout from there.
  // flag 0xb8 skips the spad store; 0x38 keeps it for radiance/smem.
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


//  gemmini_mvout((void*)&C_hw[0][0], 128 )

  gemmini_fence();
#if !defined(SPIKE_SIM) && !defined(MX_ROCKET)
  gemmini_poll_until_ready();   // reads MMIO busy reg (unbacked on MX_ROCKET); fence suffices there
#endif
  gemmini_fence();

#ifdef SPIKE_SIM
  gemmini_mx_read_smem(&C_hw[0][0], SPAD_DEST * 16, MATMUL_M * MATMUL_N);
#elif defined(MX_ROCKET)
  // Standalone: no shared memory; mvout the BF16 output from the accumulator to DRAM.
  // Acc geometry (from the loop-store HW, LoopMatmul.scala:825 acc_addr_offset=(i*iter_max_j+j)*
  // block_size): the acc packs 64 BF16/row (16 lanes x 4 BF16); output blocks are laid out row-major
  // by (row-tile i, 64-wide col-block cb) INTERLEAVED -> tile (i,cb) at acc row (i*col_blocks+cb)*DIM
  // (NOT cb*MATMUL_M+i*DIM; the col blocks interleave within each row-tile, not in separate halves).
  // Within a block the acc read is HALF-WIDTH: mx_chunk_id=0 -> low 32 cols, =1 -> high 32. Iterate
  // row-tiles x col-blocks x 2 chunks. Generalizes the 64x64 recipe (col_blocks=1). MvoutRs2: num_rows
  // @ bit 48 (mvout_rows_bits=6 for DIM=16), mx_chunk_id (3b) @ bit 54; config_st unchanged.
  #define MX_CHUNK_SHIFT (48 + 6)
  int col_blocks = MATMUL_N / 64;   // iter_max_j = # of 64-wide column blocks (N=64->1, N=128->2)
  gemmini_fence();
  for (int i = 0; i < tiles_I; i++) {
    for (int cb = 0; cb < col_blocks; cb++) {
      for (int ck = 0; ck < 2; ck++) {   // ck0 = low 32 cols of block, ck1 = high 32
        uint32_t acc_tile_addr = acc_addr + (i * col_blocks + cb) * DIM;
        // col offset in BF16 = cb*64 + ck*32 -> /BF16_PER_WORD out_t
        out_t *dram_ptr = &C_hw[i * DIM][(cb * 64 + ck * 32) / BF16_PER_WORD];
        ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, (uint64_t)(uintptr_t)dram_ptr,
            ((uint64_t)(DIM) << (ADDR_LEN + 16)) | ((uint64_t)(ck) << MX_CHUNK_SHIFT) |
            ((uint64_t)(DIM) << ADDR_LEN) | (uint64_t)(acc_tile_addr), k_MVOUT);
      }
    }
  }
  gemmini_fence();
#else
  uint64_t* smem_start_addr = ((uint64_t*)SMEM) + SPAD_DEST * 2;
  printf("Address: %p \n", smem_start_addr);
  for (int i = 0; i < MATMUL_M; i ++) {
    for (int j = 0; j < OUT_COLS; j++) {
        C_hw[i][j] = *(smem_start_addr + (i*OUT_COLS + j));
    }
  }
#endif

  // ---- Debug print tile (0,0) ----
//  printf("=== Tile (0,0) - acc_addr=0x%08x ===\n", acc_addr);
//  for (int i = 0; i < DIM; i++) {
//    for (int j = 0; j < DIM; j++) {
//      printf("C_hw[%d][%d] = 0x%016lx\n", i, j, (unsigned long)C_hw[i][j]);
//    }
//  }

  // ---- Elementwise check against C_out_bf16 ----
  int errors = 0;

  for (int i = 0; i < MATMUL_M; i++) {
    for (int j = 0; j < OUT_COLS; j++) {
      uint64_t got = C_hw[i][j];

      // Pack 4 consecutive bf16 golden values into expected uint64_t
      uint64_t exp = ((uint64_t)C_out_bf16[i][j*4 + 3] << 48) |
                     ((uint64_t)C_out_bf16[i][j*4 + 2] << 32) |
                     ((uint64_t)C_out_bf16[i][j*4 + 1] << 16) |
                     ((uint64_t)C_out_bf16[i][j*4 + 0]);

      if (got != exp) {
        for (int lane = 0; lane < BF16_PER_WORD; lane++) {
          uint16_t got_bf16 = (got >> (lane * 16)) & 0xFFFF;
          uint16_t exp_bf16 = C_out_bf16[i][j * BF16_PER_WORD + lane];
          if (got_bf16 != exp_bf16) {
            printf("MISMATCH @(%d,%d) HW=0x%04x EXP=0x%04x\n",
                   i, j * BF16_PER_WORD + lane, got_bf16, exp_bf16);
            errors++;
          }
        }
      }
    }
  }

  if (errors == 0) {
    printf("fp8 WS matmul test PASSED (no mismatches).\n");
  } else {
    printf("fp8 WS matmul test FAILED with %d mismatches.\n", errors);
  }

#ifndef BAREMETAL
  exit(errors != 0);
#else
  return errors != 0;
#endif
}