#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifndef BAREMETAL
#include <sys/mman.h>
#include <stdlib.h>
#include <sys/mman.h>
#endif

// #include "include/gemmini_testutils.h"
#include "include/gemmini.h"
#include "include/matmul_data_mx_fp8.h"

#define FP8
// #define FP6
// #define FP4

// #ifdef FP8
// #define DIM 16
// #else
// #define DIM 32
// #endif
#define DIM 16

#ifdef FP6
#define USE_LUT
#endif

#define ADDR_LEN 32

#define GEMMINI_CTRL 0x40084000
#define GEMMINI_RS1_ADDR (GEMMINI_CTRL + 0x10)
#define GEMMINI_RS2_ADDR (GEMMINI_CTRL + 0x18)
#define GEMMINI_INST_ADDR (GEMMINI_CTRL + 0x0)

#define GEMMINI_SF_MEM 0x40088000
#define GEMMINI_SF_MEM_A GEMMINI_SF_MEM + 0x2000
#define GEMMINI_SF_MEM_B GEMMINI_SF_MEM

#define GEMMINI_SPAD_ADDR_A DIM + MATMUL_N * MATMUL_GK
#define GEMMINI_SPAD_ADDR_B 0x0
#define GEMMINI_ACC_BASE_ADDR 0x80000000
#define GEMMINI_ACC_ADDR_C GEMMINI_ACC_BASE_ADDR

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

void load_lut(uint8_t *lut, int n) {
  // TODO
}

int main() {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    perror("mlockall");
    return 1;
  }
#endif

  static out_t C_hw[MATMUL_M][MATMUL_N] = {0}; // TODO: Change for fp4/6

  // Configure Gemmini
  gemmini_flush(0);
  gemmini_config_ex(WEIGHT_STATIONARY, 0, 0); // TODO: Configure type (defaults to fp8, I think)
  gemmini_extended_config_st(DIM * sizeof(out_t), NO_ACTIVATION, 1);

  // MVIN B
  gemmini_config_ld(DIM * sizeof(welem_t));
  gemmini_extended_mvin((void *) B_in, GEMMINI_SPAD_ADDR_B, MATMUL_N, MATMUL_K); // TODO: Half one dimension for fp4/6

  // MVIN A
  gemmini_config_ld(DIM * sizeof(elem_t));
  gemmini_extended_mvin((void *) A_in, GEMMINI_SPAD_ADDR_A, MATMUL_K, MATMUL_M); // TODO: Half one dimension for fp4/6

  for (size_t m = 0; m < MATMUL_M; m += 16) { // TODO: 32 for fp6/4
    for (size_t n = 0; n < MATMUL_N; n += 16) {
#ifdef USE_LUT
      // load_lut(); // TODO
#endif

      for (size_t k = 0; k < MATMUL_K; k += 16) {
        if (k % 32 == 0) {
          load_scale_factors((volatile uint64_t *) GEMMINI_SF_MEM_A, (uint8_t *) &A_scales_row[k / 32][m], 16); // Can scale factors be indexed by the compute function? If so then these can go before the loop
          load_scale_factors((volatile uint64_t *) GEMMINI_SF_MEM_B, (uint8_t *) &B_scales_col[k / 32][n], 16);
        }
        gemmini_config_ld(DIM * sizeof(welem_t));
        gemmini_preload(GEMMINI_SPAD_ADDR_B + DIM * sizeof(welem_t) * MATMUL_N * k, GEMMINI_ACC_ADDR_C + DIM * (m + MATMUL_M * n)); // TODO: Check this math. The second argument is the position of the tile in the accumulator. I have low confidence in my math here. DIM should be the size of the tile, may need to be multiplied by sizeof(bf16)
        gemmini_config_ld(DIM * sizeof(elem_t));
        gemmini_compute_preloaded(GEMMINI_SPAD_ADDR_A + DIM * sizeof(elem_t) * MATMUL_M * k, k == 0 ? GARBAGE_ADDR : GEMMINI_ACC_ADDR_C);
      }
    }
  }
  
  // MVOUT
  gemmini_extended_mvout((void *) C_hw, GEMMINI_ACC_ADDR_C, MATMUL_M, MATMUL_N);

  gemmini_fence();

  int errors = 0;
  for (int m = 0; m < MATMUL_M; m ++) {
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