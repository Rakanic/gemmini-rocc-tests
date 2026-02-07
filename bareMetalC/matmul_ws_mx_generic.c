#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifndef BAREMETAL
#include <sys/mman.h>
#include <stdlib.h>
#include <sys/mman.h>
#endif

#include "include/gemmini.h"

// Select one of these
#define FP8
// #define FP6
// #define FP4

#ifdef FP8
#include "include/matmul_data_mx_fp8.h"
#else
#ifdef FP6
#include "include/matmul_data_mx_fp6.h"
#else
#include "include/matmul_data_mx_fp4.h"
#endif
#endif

#ifdef FP8
#define TILE 16
#define VALUES_PER_BYTE 1
#else
#define TILE 32
#define VALUES_PER_BYTE 2
#endif

#ifdef FP6
#define USE_LUT_DEF
#define USE_LUT 1
#else
#define USE_LUT 0
#endif

#define DIM 16

#define ADDR_LEN 32

#define GEMMINI_CTRL 0x40084000
#define GEMMINI_RS1_ADDR (GEMMINI_CTRL + 0x10)
#define GEMMINI_RS2_ADDR (GEMMINI_CTRL + 0x18)
#define GEMMINI_INST_ADDR (GEMMINI_CTRL + 0x0)

#define GEMMINI_LUT0_ADDR (GEMMINI_CTRL + 0x80)
#define GEMMINI_LUT1_ADDR (GEMMINI_CTRL + 0x200)
#define GEMMINI_LUT2_ADDR (GEMMINI_CTRL + 0x380)

#define GEMMINI_SF_MEM 0x40088000
#define GEMMINI_SF_MEM_A (GEMMINI_SF_MEM + 0x2000)
#define GEMMINI_SF_MEM_B GEMMINI_SF_MEM

#define GEMMINI_SPAD_ADDR_A (DIM + MATMUL_N * MATMUL_GK)
#define GEMMINI_SPAD_ADDR_B 0x0
#define GEMMINI_ACC_BASE_ADDR 0x80000000
#define GEMMINI_ACC_ADDR_C GEMMINI_ACC_BASE_ADDR

#define GEMMINI_FORMAT_FP8
#define GEMMINI_FORMAT_FP6
#define GEMMINI_FORMAT_FP4

// TODO: Confirm this enum's values are correct
#ifdef FP8
#define GEMMINI_FORMAT GEMMINI_FORMAT_FP8 0
#else
#ifdef FP6
#define GEMMINI_FORMAT GEMMINI_FORMAT_FP6 1
#else
#define GEMMINI_FORMAT GEMMINI_FORMAT_FP4 2
#endif
#endif

#undef ROCC_INSTRUCTION_RS1_RS2
#define ROCC_INSTRUCTION_RS1_RS2(x, rs1, rs2, funct) { \
  *((volatile uint64_t *) GEMMINI_RS1_ADDR) = (rs1); \
  *((volatile uint64_t *) GEMMINI_RS2_ADDR) = (rs2); \
  *((volatile uint32_t*) GEMMINI_INST_ADDR) = (0x7B) | (0 << 7) | (3 << 12) | (1 << 15) | (2 << 20) | ((funct) << 25); \
}

typedef uint8_t elem_t;
typedef uint8_t welem_t;
typedef uint8_t out_t;

void load_scale_factors(volatile uint64_t *sf_mem, uint8_t *scale_factors, int n) {
  uint64_t *dword_scale_factors = (uint64_t *) scale_factors;
  for (size_t i = 0; i < n / 8; i ++) {
    sf_mem[i] = dword_scale_factors[i];
  }
}

void load_lut(volatile uint32_t *lut_mem, uint8_t *lut) {
  lut_mem[0] = (uint32_t) (lut[0] | (lut[1] << 6) | (lut[2] << 12) | (lut[3] << 18) | (lut[4] << 24) | (lut[5] << 30));
  lut_mem[1] = (uint32_t) ((lut[5] >> 2) | (lut[6] << 4) | (lut[7] << 10) | (lut[8] << 16) | (lut[9] << 22) | (lut[10] << 28));
  lut_mem[2] = (uint32_t) ((lut[10] >> 4) | (lut[11] << 2) | (lut[12] << 8) | (lut[13] << 14) | (lut[14] << 20) | (lut[15] << 26));
}

int main() {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    perror("mlockall");
    return 1;
  }
#endif

  static out_t C_hw[MATMUL_M / VALUES_PER_BYTE][MATMUL_N] = {0};

  // Configure Gemmini
  gemmini_flush(0);
  // gemmini_config_ex(WEIGHT_STATIONARY, 0, 0); // Full version used instead to configure format
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC,
    ((uint64_t)acc_scale_t_to_acc_scale_t_bits((acc_scale_t)ACC_SCALE_IDENTITY) << 32)
    | ((uint64_t)(1) << 16) // A stride
    | (GEMMINI_FORMAT << 14) // C format
    | (GEMMINI_FORMAT << 12) // B format
    | (GEMMINI_FORMAT << 10) // A format
    | (0 << 9) // B transpose
    | (0 << 8) // A transpose
    | ((false) << 7) // Set only strides
    | ((USE_LUT) << 4)
    | ((0) << 3) // Activation function
    | ((WEIGHT_STATIONARY) << 2)
    | CONFIG_EX,
    ((uint64_t)(1) << 48) // C stride
    | (0),
    k_CONFIG);
  gemmini_extended_config_st(DIM * sizeof(out_t), NO_ACTIVATION, 1);

  // MVIN B
  gemmini_config_ld(DIM * sizeof(welem_t));
  gemmini_extended_mvin((void *) B_in, GEMMINI_SPAD_ADDR_B, MATMUL_N / VALUES_PER_BYTE, MATMUL_K); // TODO: Half one dimension for fp4/6

  // MVIN A
  gemmini_config_ld(DIM * sizeof(elem_t));
  gemmini_extended_mvin((void *) A_in, GEMMINI_SPAD_ADDR_A, MATMUL_K / VALUES_PER_BYTE, MATMUL_M); // TODO: Half one dimension for fp4/6

#ifdef USE_LUT_DEF
  for (size_t i = 0; i < 16; i ++) {
    load_lut(((volatile uint32_t *) GEMMINI_LUT0_ADDR) + 3 * i, B_lut);
    load_lut(((volatile uint32_t *) GEMMINI_LUT1_ADDR) + 3 * i, A_lut);
    load_lut(((volatile uint32_t *) GEMMINI_LUT2_ADDR) + 3 * i, C_lut);
  }
  for (size_t i = 16; i < 32; i ++) {
    load_lut(((volatile uint32_t *) GEMMINI_LUT2_ADDR) + 3 * i, C_lut);
  }
#endif

  for (size_t m = 0; m < MATMUL_M; m += TILE) { // TODO: 32 for fp6/4
    for (size_t n = 0; n < MATMUL_N; n += TILE) {

      for (size_t k = 0; k < MATMUL_K; k += TILE) {
        if (k % 32 == 0) {
          load_scale_factors((volatile uint64_t *) GEMMINI_SF_MEM_A, (uint8_t *) &A_scales_row[k / 32][m], 16);
          load_scale_factors((volatile uint64_t *) GEMMINI_SF_MEM_B, (uint8_t *) &B_scales_col[k / 32][n], 16);
        }
        gemmini_config_ld(DIM * sizeof(welem_t));
        gemmini_preload(GEMMINI_SPAD_ADDR_B + TILE * sizeof(welem_t) * MATMUL_N * k, GEMMINI_ACC_ADDR_C + TILE * (m + MATMUL_M * n)); // TODO: Check this math. The second argument is the position of the tile in the accumulator. I have low confidence in my math here. TILE should be the size of the tile, may need to be multiplied by sizeof(bf16)
        gemmini_config_ld(DIM * sizeof(elem_t));
        gemmini_compute_preloaded(GEMMINI_SPAD_ADDR_A + TILE * sizeof(elem_t) * MATMUL_M * k, k == 0 ? GARBAGE_ADDR : GEMMINI_ACC_ADDR_C);
      }
    }
  }
  
  // MVOUT
  gemmini_extended_mvout((void *) C_hw, GEMMINI_ACC_ADDR_C, MATMUL_M / VALUES_PER_BYTE, MATMUL_N);

  gemmini_fence();

  int errors = 0;
  for (int m = 0; m < MATMUL_M / VALUES_PER_BYTE; m ++) {
    for (int n = 0; n < MATMUL_N; n ++) {
      uint64_t got = C_hw[m][n];
      uint64_t exp = (uint64_t) C_out[m][n];
      if (got != exp) {
          errors ++;
          printf("Got: %d    Expected: %d\n", got, exp);
      }
    }
  }
}