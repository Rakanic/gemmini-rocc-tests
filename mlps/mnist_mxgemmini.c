#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifndef BAREMETAL
#include <sys/mman.h>
#include <stdlib.h>
#endif

#include "include/gemmini_testutils.h"
#include "mnist_mxgemmini_params.h"

#define DIM 16
#define ADDR_LEN 32
#define BF16_PER_WORD 4
#define OUT_COLS (MATMUL_N / BF16_PER_WORD)

static float bf16_to_float(uint16_t b) {
  union { uint32_t u; float f; } u;
  u.u = ((uint32_t)b) << 16;
  return u.f;
}

static int argmax_first_n(const uint16_t row[MATMUL_N], int n) {
  int best = 0;
  float best_v = bf16_to_float(row[0]);
  for (int j = 1; j < n; j++) {
    float v = bf16_to_float(row[j]);
    if (v > best_v) { best_v = v; best = j; }
  }
  return best;
}

int main() {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    perror("mlockall");
    return 1;
  }
#endif

  static uint64_t C_hw[MATMUL_M][OUT_COLS];
  uint32_t out_scales[512] = {0};
  memset(C_hw, 0, sizeof(C_hw));

  int tiles_I = MATMUL_M / DIM;
  int tiles_J = MATMUL_N / DIM;
  int tiles_K = MATMUL_K / DIM;

  uint32_t a_base = 0;
  uint32_t b_base = BANK_NUM * BANK_ROWS - tiles_K * tiles_J * DIM;

  gemmini_flush(0);
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY,
                              1, 1, 0, 0, false, 0, 0, 3, 0);

  gemmini_mx_load_scales((uint64_t)&A_scales_row, sizeof(A_scales_row), 0);
  gemmini_mx_load_scales((uint64_t)&B_scales_col, sizeof(B_scales_col), 1);

  gemmini_config_ld(MATMUL_K * sizeof(uint8_t));
  for (int i = 0; i < tiles_I; i++) {
    for (int k = 0; k < tiles_K; k++) {
      const uint8_t *dram_ptr = ((const uint8_t*)A_in) + i * DIM * MATMUL_K + k * DIM;
      uint32_t sp_addr = a_base + (i * tiles_K + k) * DIM;
      gemmini_extended_mvin((void *) dram_ptr, sp_addr, DIM, DIM);
    }
  }

  gemmini_config_ld(MATMUL_N * sizeof(uint8_t));
  for (int j = 0; j < tiles_J; j++) {
    for (int k = 0; k < tiles_K; k++) {
      const uint8_t *dram_ptr = ((const uint8_t*)B_in) + k * DIM * MATMUL_N + j * DIM;
      uint32_t sp_addr = b_base + (j * tiles_K + k) * DIM;
      gemmini_extended_mvin((void *) dram_ptr, sp_addr, DIM, DIM);
    }
  }

  int SPAD_DEST = 128;

  gemmini_config_st(OUT_COLS * sizeof(uint64_t));
  gemmini_mxquant_config_mvout((uint64_t)out_scales, tiles_I, tiles_J, tiles_K, 0, 0, 1);

  uint64_t start = read_cycles();
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
  uint64_t end = read_cycles();

  gemmini_mx_read_smem(&C_hw[0][0], SPAD_DEST * 16, MATMUL_M * MATMUL_N);
  gemmini_fence();

  int bf16_errors = 0;
  for (int i = 0; i < MATMUL_M; i++) {
    for (int j = 0; j < OUT_COLS; j++) {
      uint64_t got = C_hw[i][j];
      uint64_t exp = ((uint64_t)C_out_bf16[i][j*4 + 3] << 48) |
                     ((uint64_t)C_out_bf16[i][j*4 + 2] << 32) |
                     ((uint64_t)C_out_bf16[i][j*4 + 1] << 16) |
                     ((uint64_t)C_out_bf16[i][j*4 + 0]);
      if (got != exp) {
        for (int lane = 0; lane < BF16_PER_WORD; lane++) {
          uint16_t g = (got >> (lane * 16)) & 0xFFFF;
          uint16_t e = C_out_bf16[i][j * BF16_PER_WORD + lane];
          if (g != e) bf16_errors++;
        }
      }
    }
  }

  uint16_t logits[MATMUL_M][MATMUL_N];
  for (int i = 0; i < MATMUL_M; i++) {
    for (int j = 0; j < OUT_COLS; j++) {
      uint64_t w = C_hw[i][j];
      for (int lane = 0; lane < BF16_PER_WORD; lane++) {
        logits[i][j * BF16_PER_WORD + lane] = (w >> (lane * 16)) & 0xFFFF;
      }
    }
  }

  int correct = 0;
  for (int i = 0; i < MNIST_NUM_CLASSES; i++) {
    int pred = argmax_first_n(logits[i], MNIST_NUM_CLASSES);
    int label = mnist_labels[i];
    printf("img %d: predicted=%d label=%d %s\n",
           i, pred, label, pred == label ? "OK" : "WRONG");
    if (pred == label) correct++;
  }

  printf("MX-FP8 MNIST: %d/%d correct, bf16 mismatches=%d, cycles=%llu\n",
         correct, MNIST_NUM_CLASSES, bf16_errors, (unsigned long long)(end - start));

  int pass = (bf16_errors == 0) && (correct == MNIST_NUM_CLASSES);
  if (pass) {
    printf("PASSED\n");
  } else {
    printf("FAILED\n");
  }

#ifndef BAREMETAL
  exit(pass ? 0 : 1);
#else
  return pass ? 0 : 1;
#endif
}
