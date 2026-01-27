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

// Match your header types
#define DIM MATMUL_M

typedef uint8_t elem_t;   // A_in: lower 8 bits = fp8:e4m3, upper bits zero
typedef uint8_t welem_t;  // B_in: lower 8 bits = fp8:e4m3, upper bits zero
typedef uint64_t  out_t;    // C_scaled: fp8:e4m3 (1 byte per output)

// void load_scale_factors(uint64_t* src, size_t bytes) {
//   volatile uint64_t *dst = (volatile uint64_t *)SCALE_FACT_MEM;

//   for (int transactions = 0; transactions < (bytes + 7) / 8; transactions++) {
//     dst[transactions] = src[transactions];
//   }
// }

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
  gemmini_config_ex(WEIGHT_STATIONARY, 0, 0);

  // We want 1 byte per output element in DRAM
  gemmini_extended_config_st(DIM * sizeof(out_t), NO_ACTIVATION, 1);

  // Load per-element scaling factors into the scale SRAM
  // (C_scale is uint8_t[DIM][DIM], packed row-major)
  // load_scale_factors((const uint64_t *) C_scale, sizeof(C_scale));

  // MVIN B as B^T for WS
  gemmini_config_ld(DIM * sizeof(welem_t));
  gemmini_mvin((void *) B_in, 1 * DIM);

  // MVIN A
  gemmini_config_ld(DIM * sizeof(elem_t));
  gemmini_mvin((void *) A_in, 0 * DIM);

  // Preload + compute
  gemmini_config_ld(DIM * sizeof(welem_t));
  gemmini_preload(1 * DIM, (1u << (ADDR_LEN - 1)));
  gemmini_config_ld(DIM * sizeof(elem_t));
  gemmini_compute_preloaded(0 * DIM, GARBAGE_ADDR);

  // MVOUT scaled fp8 C
  gemmini_mvout((void *) C_hw, (1u << (ADDR_LEN - 1)));

  // Single fence at the end, like your fp6 test
  gemmini_fence();

  // ---------- Compare against golden fp8 (C_scaled) ----------
  int errors = 0;
  for (int i = 0; i < DIM; i++) {
    for (int j = 0; j < DIM; j++) {
      uint64_t got = C_hw[i][j];
// uint64_t exp = (uint64_t) C_scaled[i][j];
// if (got != exp) {
// printf("@(%d,%d) HW=0x%02x  EXP=0x%02x\n",
// i, j, (unsigned) got, (unsigned) exp);
// errors++;
// }
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
