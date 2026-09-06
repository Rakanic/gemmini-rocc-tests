// A REAL TinyLlama MLP, back to back, on MxGemmini -- one ELF, real data, llama dimensions.
//
// Three matmuls on the mesh with the host's fp32 glue between them, exactly as the layer runs in
// the model. Operands are a real decoder layer captured from a real forward pass
// (npu-exploration/app/capture_llama_layer.py -> gen_llama_layer.py -> include/llama_mlp.h):
// hidden size D = 2048 kept FULL, F of the 5632 FFN neurons, M real tokens of wikitext2.
//
//   host   xn  = rmsnorm(h_mid, w_post_ln)      fp32, then MX-quantized to fp8 codes + E8M0
//   mesh   G   = Xn @ Wg     [M,D]x[D,F] -> bf16
//   mesh   U   = Xn @ Wu     [M,D]x[D,F] -> bf16   (Xn stays resident; only B is re-mvin'd)
//   host   h   = silu(G) * U                    fp32, then MX-quantized
//   mesh   Y   = H  @ Wd     [M,F]x[F,D] -> bf16, in two N-chunks (see the budget below)
//   host   out = h_mid + Y                      the residual
//
// gate/up are EXACT real llama values -- the projection input is the full 2048 -- while down_proj
// is an honest partial sum over the captured neurons, and REF_MLP is truncated the same way.
//
// SCRATCHPAD BUDGET. 16384 rows x 16 B hold the A tiles, the B tiles and the output together, and
// `mx_smem` accumulates and is never cleared (gemmini.cc:1190), so every matmul in the ELF needs a
// DISJOINT output region -- not merely a free one at the time it runs. Hence:
//
//   stage 1/2   A Xn      0    .. 4095      B Wg/Wu   8192 .. 16383     (B arg = 16384)
//               G smem    4096 .. 4351      U smem    4352 .. 4607
//   stage 3     A H       0    .. 127       B Wd_c     512 .. 4607      (B arg = 4608)
//               Y0 smem   4608 .. 8703      Y1 smem   8704 .. 12799
//
// Stage 3's B lands on stage 1/2's dead G/U output rows, which is fine -- they were drained -- but
// no two *smem* regions overlap, which is what the accumulation makes load-bearing. Un-chunked,
// down_proj would need 128 + 8192 + 8192 = 16512 rows, 128 over the budget.
//
// Unified real-RoCC instruction stream for Spike (-DSPIKE_SIM) and the standalone RTL
// (-DMX_ROCKET), mirroring matmul_tiled_fp8_64x64.c and matmul_tiled_fp8_64x64_chain.c.
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifndef BAREMETAL
#include <sys/mman.h>
#include <stdlib.h>
#endif

#include "include/gemmini_testutils.h"
#include "include/mx_host.h"
#include "include/llama_mlp.h"

#define DIM 16

#define GEMMINI_CTRL 0x40084000
#define GEMMINI_RS1_ADDR (GEMMINI_CTRL + 0x10)
#define GEMMINI_RS2_ADDR (GEMMINI_CTRL + 0x18)
#define GEMMINI_INST_ADDR (GEMMINI_CTRL + 0x0)

// Radiance RTL drives gemmini through an MMIO command mimic; the standalone rocket config is a
// real RoCC, so keep gemmini.h's direct macro on both paths we build for.
#if !defined(SPIKE_SIM) && !defined(MX_ROCKET)
#undef ROCC_INSTRUCTION_RS1_RS2
#define ROCC_INSTRUCTION_RS1_RS2(x, rs1, rs2, funct) { \
    *((volatile uint64_t *) GEMMINI_RS1_ADDR) = (rs1); \
    *((volatile uint64_t *) GEMMINI_RS2_ADDR) = (rs2); \
    *((volatile uint32_t*) GEMMINI_INST_ADDR) = (0x7B) | (0 << 7) | (3 << 12) | (1 << 15) | (2 << 20) | ((funct) << 25); \
}
#endif

#define OUT_BF16 3      // gemmini_extended3_config_ex out_mx_fmt: 3 = BF16, no requant
#define SPAD_STORE 0x38 // loop_ws skips: keep the full-width store into the internal scratchpad

#define SPAD_TOP (BANK_NUM * BANK_ROWS)
#define SPAD_A       0
#define SPAD_G    4096
#define SPAD_U    4352
#define SPAD_B3    512
#define SPAD_B3_ARG 4608
#define SPAD_Y0   4608
#define SPAD_Y1   8704

#define NCHUNK  (LLAMA_D / 2)          // down_proj output columns per pass
#define NCHUNKS (LLAMA_D / NCHUNK)

// ---- host buffers ----
static float    xn_f[LLAMA_M * LLAMA_D];
static uint8_t  xn_codes[LLAMA_M * LLAMA_D];
static uint8_t  xn_scales[LLAMA_GD * LLAMA_M];
static float    h_f[LLAMA_M * LLAMA_F];
static uint8_t  h_codes[LLAMA_M * LLAMA_F];
static uint8_t  h_scales[LLAMA_GF * LLAMA_M];
static uint8_t  wd_scales_chunk[LLAMA_GF * NCHUNK];
static uint16_t G_hw[LLAMA_M * LLAMA_F];
static uint16_t U_hw[LLAMA_M * LLAMA_F];
static uint16_t Ychunk[LLAMA_M * NCHUNK];
static uint16_t Y_hw[LLAMA_M * LLAMA_D];
static uint16_t OUT_hw[LLAMA_M * LLAMA_D];

// mvin A[M][K] as tiles: tile (i,k) -> a_spad + (i*tiles_K + k)*DIM, row stride K.
static void mvin_A(const uint8_t *A, int M, int K, uint32_t a_spad) {
  gemmini_config_ld(K * sizeof(uint8_t));
  int tiles_I = M / DIM, tiles_K = K / DIM;
  for (int i = 0; i < tiles_I; i++)
    for (int k = 0; k < tiles_K; k++)
      gemmini_extended_mvin((void *) (A + (size_t) i * DIM * K + (size_t) k * DIM),
                            a_spad + (i * tiles_K + k) * DIM, DIM, DIM);
}

// mvin B[K][N_full], columns [n0, n0+N): tile (k,j) -> b_spad + (k*tiles_J + j)*DIM, row stride
// N_full. Slot order matches the model's B_t = B_sp + (k_outer*TJ + j)*DIM (gemmini.cc:1218) and
// the non-square matmul_tiled_fp8_128x128x256 test.
static void mvin_B(const uint8_t *B, int K, int N_full, int n0, int N, uint32_t b_spad) {
  gemmini_config_ld(N_full * sizeof(uint8_t));
  int tiles_K = K / DIM, tiles_J = N / DIM;
  for (int k = 0; k < tiles_K; k++)
    for (int j = 0; j < tiles_J; j++)
      gemmini_extended_mvin((void *) (B + (size_t) k * DIM * N_full + (size_t) (n0 + j * DIM)),
                            b_spad + (k * tiles_J + j) * DIM, DIM, DIM);
}

// Drain a BF16 [M][N] tile the mesh left flat and row-major in the internal scratchpad.
static void mvout_bf16(uint16_t *dst, uint32_t spad, int M, int N) {
  gemmini_config_st(DIM * sizeof(uint8_t));
  int total_rows = M * N * 2 / DIM;
  uint8_t *b = (uint8_t *) dst;
  for (int r = 0; r < total_rows; r += DIM)
    gemmini_extended_mvout(b + (size_t) r * DIM, spad + r, DIM, DIM);
  gemmini_fence();
}

// One mesh matmul, BF16 out: A already resident at a_spad, B already resident under b_arg.
static void mesh_matmul(int M, int K, int N, uint32_t a_spad, uint32_t b_arg, uint32_t c_spad) {
  static uint32_t scale_sink[512] __attribute__((aligned(32)));
  int I = M / DIM, J = N / DIM, Kt = K / DIM;
  gemmini_config_st(N * sizeof(uint16_t));
  gemmini_mxquant_config_mvout((uint64_t) scale_sink, I, J, Kt, 0, 0, 1);
  gemmini_loop_ws_spad(I, J, Kt,
                       0, 0, 0,
                       a_spad,
                       b_arg,
                       0,
                       c_spad,
                       false, false,
                       false, false, false,
                       NO_ACTIVATION,
                       0, 0,
                       false,
                       SPAD_STORE);
  gemmini_fence();
}

// riscv-tests' own handle_trap is weak and exits 1337 with no cause, which is indistinguishable
// from a stale libgemmini.so. Overriding it turns that into one diagnosable line -- it is how the
// missing -DSPIKE_SIM on this test's first run was found (cause 7, a store fault, at the MMIO
// command-mimic address the non-spike path falls back to).
uintptr_t handle_trap(uintptr_t cause, uintptr_t epc, uintptr_t regs[32]) {
  printf("TRAP cause=%d epc=%lx tval?=%lx\n", (int) cause, (unsigned long) epc,
         (unsigned long) regs[10]);
  tohost_exit(1337);
  return 0;
}

int main() {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) { perror("mlockall"); return 1; }
#endif
  printf("llama MLP: M=%d D=%d F=%d  (real TinyLlama layer, fp8 e4m3 + E8M0)\n",
         LLAMA_M, LLAMA_D, LLAMA_F);

  gemmini_flush(0);
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false,
                              0, 0, OUT_BF16, 0);

  uint64_t t_host = 0, t_mesh = 0, t0;

  // ================= host stage 0: RMSNorm + MX quantization =================
  t0 = read_cycles();
  mx_rmsnorm((const uint16_t *) H_MID_BF16, W_POST_LN_BF16, LLAMA_M, LLAMA_D, LLAMA_RMS_EPS, xn_f);
  mx_quantize_rows(xn_f, LLAMA_M, LLAMA_D, xn_codes, xn_scales);
  t_host += read_cycles() - t0;

  int xn_cd = mx_count_diff_u8(xn_codes, (const uint8_t *) XN_CODES, LLAMA_M * LLAMA_D);
  int xn_sd = mx_count_diff_u8(xn_scales, (const uint8_t *) XN_SCALES_ROW, LLAMA_GD * LLAMA_M);
  printf("host  rmsnorm+quant: codes differ %d/%d, scales differ %d/%d vs golden\n",
         xn_cd, LLAMA_M * LLAMA_D, xn_sd, LLAMA_GD * LLAMA_M);

  // ================= mesh stages 1 and 2: gate_proj and up_proj =================
  t0 = read_cycles();
  mvin_A(xn_codes, LLAMA_M, LLAMA_D, SPAD_A);
  gemmini_mx_load_scales((uint64_t) xn_scales, sizeof(xn_scales), 0);
  gemmini_mx_load_scales((uint64_t) &WG_SCALES_COL, sizeof(WG_SCALES_COL), 1);
  gemmini_fence();
  mvin_B((const uint8_t *) WG_IN, LLAMA_D, LLAMA_F, 0, LLAMA_F, SPAD_TOP - LLAMA_D * LLAMA_F / DIM);
  mesh_matmul(LLAMA_M, LLAMA_D, LLAMA_F, SPAD_A, SPAD_TOP, SPAD_G);
  t_mesh += read_cycles() - t0;
  mvout_bf16(G_hw, SPAD_G, LLAMA_M, LLAMA_F);

  t0 = read_cycles();
  gemmini_mx_load_scales((uint64_t) &WU_SCALES_COL, sizeof(WU_SCALES_COL), 1);
  gemmini_fence();
  mvin_B((const uint8_t *) WU_IN, LLAMA_D, LLAMA_F, 0, LLAMA_F, SPAD_TOP - LLAMA_D * LLAMA_F / DIM);
  mesh_matmul(LLAMA_M, LLAMA_D, LLAMA_F, SPAD_A, SPAD_TOP, SPAD_U);
  t_mesh += read_cycles() - t0;
  mvout_bf16(U_hw, SPAD_U, LLAMA_M, LLAMA_F);

  int g_d = mx_count_diff_u16(G_hw, (const uint16_t *) G_OUT_BF16, LLAMA_M * LLAMA_F);
  int u_d = mx_count_diff_u16(U_hw, (const uint16_t *) U_OUT_BF16, LLAMA_M * LLAMA_F);
  printf("mesh  G = Xn @ Wg : %d/%d differ from golden\n", g_d, LLAMA_M * LLAMA_F);
  printf("mesh  U = Xn @ Wu : %d/%d differ from golden\n", u_d, LLAMA_M * LLAMA_F);

  // ================= host stage 3: SwiGLU + MX quantization =================
  t0 = read_cycles();
  mx_swiglu(G_hw, U_hw, LLAMA_M * LLAMA_F, h_f);
  mx_quantize_rows(h_f, LLAMA_M, LLAMA_F, h_codes, h_scales);
  t_host += read_cycles() - t0;

  int h_cd = mx_count_diff_u8(h_codes, (const uint8_t *) H_CODES, LLAMA_M * LLAMA_F);
  int h_sd = mx_count_diff_u8(h_scales, (const uint8_t *) H_SCALES_ROW, LLAMA_GF * LLAMA_M);
  printf("host  silu(G)*U + quant: codes differ %d/%d, scales differ %d/%d vs golden\n",
         h_cd, LLAMA_M * LLAMA_F, h_sd, LLAMA_GF * LLAMA_M);

  // ================= mesh stage 4: down_proj, in N-chunks =================
  t0 = read_cycles();
  mvin_A(h_codes, LLAMA_M, LLAMA_F, SPAD_A);
  gemmini_mx_load_scales((uint64_t) h_scales, sizeof(h_scales), 0);
  gemmini_fence();
  t_mesh += read_cycles() - t0;

  const uint32_t y_spad[NCHUNKS] = { SPAD_Y0, SPAD_Y1 };
  for (int c = 0; c < NCHUNKS; c++) {
    // The B-side scale window is indexed b_off = group * (this pass's N) + col, so it needs the
    // chunk's columns packed contiguously rather than a stride into the [GF][D] array.
    for (int g = 0; g < LLAMA_GF; g++)
      memcpy(wd_scales_chunk + (size_t) g * NCHUNK, &WD_SCALES_COL[g][c * NCHUNK], NCHUNK);

    t0 = read_cycles();
    gemmini_mx_load_scales((uint64_t) wd_scales_chunk, sizeof(wd_scales_chunk), 1);
    gemmini_fence();
    mvin_B((const uint8_t *) WD_IN, LLAMA_F, LLAMA_D, c * NCHUNK, NCHUNK, SPAD_B3);
    mesh_matmul(LLAMA_M, LLAMA_F, NCHUNK, SPAD_A, SPAD_B3_ARG, y_spad[c]);
    t_mesh += read_cycles() - t0;

    mvout_bf16(Ychunk, y_spad[c], LLAMA_M, NCHUNK);
    for (int m = 0; m < LLAMA_M; m++)
      memcpy(&Y_hw[(size_t) m * LLAMA_D + c * NCHUNK], &Ychunk[(size_t) m * NCHUNK],
             NCHUNK * sizeof(uint16_t));
  }

  int y_d = mx_count_diff_u16(Y_hw, (const uint16_t *) Y_OUT_BF16, LLAMA_M * LLAMA_D);
  printf("mesh  Y = H @ Wd  : %d/%d differ from golden (%d chunks of %d)\n",
         y_d, LLAMA_M * LLAMA_D, NCHUNKS, NCHUNK);

  // ================= host stage 5: the residual =================
  t0 = read_cycles();
  for (int i = 0; i < LLAMA_M * LLAMA_D; i++)
    OUT_hw[i] = mx_f32_to_bf16_rne(mx_bf16_to_f32(((const uint16_t *) H_MID_BF16)[i])
                                   + mx_bf16_to_f32(Y_hw[i]));
  t_host += read_cycles() - t0;

  // ---- report ----
  int rel_y = MX_PPM(mx_rel_fro_bf16(Y_hw, (const uint16_t *) REF_MLP_BF16, LLAMA_M * LLAMA_D));
  int rel_o = MX_PPM(mx_rel_fro_bf16(OUT_hw, (const uint16_t *) REF_OUT_BF16, LLAMA_M * LLAMA_D));
  printf("grade MLP output vs fp32 reference : rel_fro %d ppm\n", rel_y);
  printf("grade residual out vs fp32 reference: rel_fro %d ppm\n", rel_o);
  printf("cycles mesh %d, host %d\n", (int) t_mesh, (int) t_host);

  int exact = g_d + u_d + y_d;
  int host_drift = xn_cd + xn_sd + h_cd + h_sd;
  // The bit-exact gate is only meaningful while the host produced the golden's own operands; if it
  // did not, say so rather than reporting a mismatch the goldens could not have predicted.
  if (exact == 0 && host_drift == 0)
    printf("llama MLP test PASSED (every mesh stage bit-exact, host stages match the golden).\n");
  else if (exact == 0)
    printf("llama MLP test PASSED WITH DRIFT: mesh stages bit-exact, but %d host byte(s) differ "
           "from the golden -- the fp32 glue diverged, and the goldens describe a different "
           "input.\n", host_drift);
  else
    printf("llama MLP test FAILED: %d mesh element(s) differ from golden (%d host byte(s) "
           "differed too).\n", exact, host_drift);

#ifndef BAREMETAL
  exit(exact != 0);
#else
  return exact != 0;
#endif
}
