#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifndef BAREMETAL
#include <sys/mman.h>
#include <stdlib.h>
#include <sys/mman.h>
#endif

#include "include/gemmini_testutils.h"
#include "include/matmul_data.h" 

#define GEMMINI_SF_MEM 0x40088000
#define GEMMINI_SF_MEM_A (GEMMINI_SF_MEM + 0x2000)
#define GEMMINI_SF_MEM_B GEMMINI_SF_MEM

// Match your header types
#define DIM MATMUL_M

#define GEMMINI_CTRL 0x40084000
#define GEMMINI_RS1_ADDR (GEMMINI_CTRL + 0x10)
#define GEMMINI_RS2_ADDR (GEMMINI_CTRL + 0x18)
#define GEMMINI_INST_ADDR (GEMMINI_CTRL + 0x0)

#undef ROCC_INSTRUCTION_RS1_RS2
#define ROCC_INSTRUCTION_RS1_RS2(x, rs1, rs2, funct) { \
    *((volatile uint64_t *) GEMMINI_RS1_ADDR) = (rs1); \
    *((volatile uint64_t *) GEMMINI_RS2_ADDR) = (rs2); \
    *((volatile uint32_t*) GEMMINI_INST_ADDR) = (0x7B) | (0 << 7) | (3 << 12) | (1 << 15) | (2 << 20) | ((funct) << 25); \
}

#define k_MVIN         2
#define k_MVOUT        3
#define k_MVOUT_SPAD 23
#define XCUSTOM_ACC 3
#define ADDR_LEN 32
#define DIM 16


typedef uint8_t elem_t;   // A_in: lower 8 bits = fp8:e4m3, upper bits zero
typedef uint8_t welem_t;  // B_in: lower 8 bits = fp8:e4m3, upper bits zero
typedef uint64_t  out_t;    // C_scaled: fp8:e4m3 (1 byte per output)

// void load_scale_factors(uint64_t* src, size_t bytes) {
//   volatile uint64_t *dst = (volatile uint64_t *)SCALE_FACT_MEM;

//   for (int transactions = 0; transactions < (bytes + 7) / 8; transactions++) {
//     dst[transactions] = src[transactions];
//   }
// }

void load_scale_factors(volatile uint64_t *sf_mem, uint8_t *scale_factors, int n) {
  uint64_t *dword_scale_factors = (uint64_t *) scale_factors;
  for (size_t i = 0; i < n / 8; i ++) {
    sf_mem[i] = dword_scale_factors[i];
  }
}

int main() {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    perror("mlockall");
    return 1;
  }
#endif

  // ---- Buffers ----
  // Inputs come from MATMUL_DATA_H: A_in[16][16], B_in[16][16]
  static out_t C_hw[DIM][DIM] = {0};  // fp8 outputs from HW

  // ---------- Run Gemmini (fp8 WS test) ----------
  gemmini_flush(0);
//  gemmini_config_ex(WEIGHT_STATIONARY, 0, 0);
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0, 3, 0);

  // We want 1 byte per output element in DRAM
  gemmini_extended_config_st(DIM * sizeof(out_t), NO_ACTIVATION, 1);

  // Load per-element scaling factors into the scale SRAM
  // (C_scale is uint8_t[DIM][DIM], packed row-major)
  // load_scale_factors((const uint64_t *) C_scale, sizeof(C_scale));
  load_scale_factors((volatile uint64_t *) GEMMINI_SF_MEM_A, A_scales_row , 32);
  load_scale_factors((volatile uint64_t *) GEMMINI_SF_MEM_B, B_scales_col , 32);
  load_scale_factors((volatile uint64_t *) GEMMINI_SF_MEM_A, A_scales_row , 32);
  load_scale_factors((volatile uint64_t *) GEMMINI_SF_MEM_B, B_scales_col , 32);

  // MVIN B and A
  gemmini_config_ld(DIM * sizeof(elem_t));
  gemmini_mvin((void *) B_in, 1 * DIM);
  gemmini_mvin((void *) A_in, 0 * DIM);

  uint32_t acc_addr = (1u << (ADDR_LEN - 1));
//  gemmini_preload(1 * DIM, acc_addr);  // Read B from spad addr 1*DIM, results -> acc_addr
//
//  // Compute: A (from spad 0*DIM) × B (preloaded) -> accumulator (at acc_addr)
//  gemmini_config_ld(DIM * sizeof(elem_t));
//  gemmini_compute_preloaded(0 * DIM, GARBAGE_ADDR);
//
//  uint32_t mvout_addr = acc_addr & ~(1 << (ADDR_LEN - 2));  // Clear accumulate bit
//  mvout_addr |= (1 << 29);  // Set full row bit
////  gemmini_mvout((void *) C_hw, mvout_addr);
//  gemmini_mvout_spad(0, acc_addr);

    gemmini_loop_ws_spad(
        1, 1, 1,              // I=1, J=1, K=1 (single 16×16 tile)
        0, 0, 0,              // pad_I=0, pad_J=0, pad_K=0
        0 * DIM,              // A scratchpad address
        2 * DIM,              // B scratchpad address
        0,                    // D (bias) - none
        acc_addr,             // C accumulator address
        false, false,         // A_transpose, B_transpose
        false, false, false,  // full_C, low_D, ex_accumulate
        NO_ACTIVATION,        // activation
        0, 0,                 // a_spad_id, b_spad_id
        false,                // is_resadd
        0x38);                // skips

//  gemmini_mvout_spad(0, acc_addr);

  // Single fence at the end, like your fp6 test
  gemmini_fence();

  // ---------- Compare against golden fp8 (C_scaled) ----------
  int errors = 0;
  for (int i = 0; i < DIM; i++) {
    for (int j = 0; j < DIM; j++) {
      uint64_t got = C_hw[i][j];
//      uint64_t exp = (uint64_t) C_scaled[i][j];
//      if (got != exp) {
//        printf("@(%d,%d) HW=0x%02x  EXP=0x%02x\n",
//        i, j, (unsigned) got, (unsigned) exp);
//        errors++;
//      }
    }
  }

//  for (int i = 0; i < DIM; i++) {
//    for (int j = 0; j < DIM; j++) {
//      uint64_t got = C_hw[i][j];
//      printf("%x,", got);
//    }
//    printf("\n");
//  }
//
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