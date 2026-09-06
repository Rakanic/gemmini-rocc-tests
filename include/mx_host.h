// mx_host.h -- the fp32 host side of an MX layer, for baremetal C on the Rocket core.
//
// The mesh does matmuls. Everything else in a llama layer -- RMSNorm, SiLU, the elementwise
// product, softmax, RoPE, the residual -- has no hardware here (the target contract declares one
// compute unit, `mx_systolic_mesh`, `ops: [matmul]`, and there is no reduction hardware), so it
// runs on the scalar core in fp32. This header is that side: the glue, plus the MX quantizer that
// hands its result back to the mesh.
//
// The quantizer follows MXQuant's END-TO-END convention, which is what the FP8 datapath was
// migrated to (npu-exploration/planning/chain_seam_hw_notes.md section 8, and gemmini.cc's FP8
// post-pass):
//
//     X    = 2^floor(log2(max(amax, FLT_EPSILON)))     per 32-element block, no `- log2_pmax` term
//     code = e4m3(v / X)                               round to nearest, TIES AWAY FROM ZERO,
//                                                      saturating at +-448, subnormals allowed
//
// so a block's max lands in [1, 2) rather than [256, 512). Rounding is done by binary search over
// the E4M3 code space, which is monotone in magnitude for a sign-magnitude format: that makes
// "nearest, ties away" exact by construction rather than a rounding-mode argument, and it costs
// 7 comparisons per element.
//
// These routines are NOT mirrored bit-for-bit in the Python golden -- the generated header carries
// the golden's own codes so a test can report how far its fp32 differs, and the expectation is
// zero, because E4M3 keeps 3 mantissa bits and absorbs a last-ulp difference between newlib and
// numpy. See planning/llama_layer_hw_plan.md section 4.

#ifndef INCLUDE_MX_HOST_H
#define INCLUDE_MX_HOST_H

#include <stdint.h>
#include <stddef.h>
#include <math.h>

#define MX_BLOCK 32
#define MX_E8M0_BIAS 127
#define MX_E4M3_MAX_CODE 0x7E   // 448.0; 0x7F is NaN/480 and is never emitted

// The baremetal environment links -nostdlib, so newlib's libm has no libc to report domain errors
// to. `expf` (in SiLU) is the only libm call left below -- everything power-of-two is done in bits
// instead -- so one stub satisfies the linker without pulling libc in.
#ifdef BAREMETAL
int *__errno(void) { static int mx_errno_slot; return &mx_errno_slot; }
#endif

// ---- exact powers of two, in bits ------------------------------------------------------------
//
// Every scale in an MX format is a power of two, and every place this header needs one it needs it
// EXACTLY. Doing that in bits rather than through ldexpf/log2f/floorf removes both a libm
// dependency and the question of whether the library rounded -- log2f(8) returning 2.9999997 would
// silently move a block scale by one exponent.

static inline float mx_pow2(int e) {
  union { uint32_t u; float f; } c;
  if (e >= -126 && e <= 127) { c.u = ((uint32_t) (e + 127)) << 23; return c.f; }
  if (e > 127) return INFINITY;
  if (e >= -149) { c.u = 1u << (23 + (e + 126)); return c.f; }   // subnormal
  return 0.0f;
}

// floor(log2(x)) for finite x > 0, exact: the IEEE exponent field, or a mantissa scan if subnormal.
static inline int mx_floor_log2(float x) {
  union { float f; uint32_t u; } c;
  c.f = x;
  int e = (int) ((c.u >> 23) & 0xFF);
  if (e) return e - 127;
  uint32_t m = c.u & 0x7FFFFF;
  if (!m) return -149;                                            // x == 0: caller clamps first
  int msb = 22;
  while (!(m & (1u << msb))) msb--;
  return -126 - (23 - msb);
}

// ---- element formats -----------------------------------------------------------------------

static inline float mx_bf16_to_f32(uint16_t b) {
  union { uint32_t u; float f; } c;
  c.u = ((uint32_t) b) << 16;
  return c.f;
}

static inline uint16_t mx_f32_to_bf16_rne(float x) {
  union { float f; uint32_t u; } c;
  c.f = x;
  if (((c.u >> 23) & 0xFF) == 0xFF) return (uint16_t) (c.u >> 16);   // nan/inf: truncate
  uint32_t lsb = (c.u >> 16) & 1;
  return (uint16_t) ((c.u + 0x7FFF + lsb) >> 16);
}

// E4M3 decode, mirroring mx_fp_math.h: subnormals are m * 2^-9, normals (1 + m/8) * 2^(e-7).
static inline float mx_e4m3_decode(uint8_t code) {
  int s = (code >> 7) & 1, e = (code >> 3) & 0xF, m = code & 0x7;
  float v = (e == 0) ? ((float) m) * 0.001953125f              // 2^-9
                     : (1.0f + 0.125f * (float) m) * mx_pow2(e - 7);
  return s ? -v : v;
}

static inline float mx_e8m0_decode(uint8_t code) { return mx_pow2((int) code - MX_E8M0_BIAS); }

// The 127 finite non-negative E4M3 magnitudes, in code order -- which is also increasing order,
// because E4M3 is sign-magnitude. Built once so the encoder can binary-search it.
static float mx_e4m3_mag[MX_E4M3_MAX_CODE + 1];
static int mx_e4m3_ready = 0;

static void mx_host_init(void) {
  if (mx_e4m3_ready) return;
  for (int c = 0; c <= MX_E4M3_MAX_CODE; c++) mx_e4m3_mag[c] = mx_e4m3_decode((uint8_t) c);
  mx_e4m3_ready = 1;
}

// Encode one already-scaled value to an E4M3 code: nearest, ties away from zero, saturating.
static inline uint8_t mx_e4m3_encode(float v) {
  uint8_t sign = 0;
  if (v < 0.0f || (v == 0.0f && signbit(v))) { sign = 0x80; v = -v; }
  if (!(v == v)) return sign | MX_E4M3_MAX_CODE;                    // NaN cannot be emitted
  if (v >= mx_e4m3_mag[MX_E4M3_MAX_CODE]) return sign | MX_E4M3_MAX_CODE;
  int lo = 0, hi = MX_E4M3_MAX_CODE;                                // largest code with mag <= v
  while (lo < hi) {
    int mid = (lo + hi + 1) >> 1;
    if (mx_e4m3_mag[mid] <= v) lo = mid; else hi = mid - 1;
  }
  // Ties away from zero: take the upper neighbour when the distances are equal.
  float d_lo = v - mx_e4m3_mag[lo], d_hi = mx_e4m3_mag[lo + 1] - v;
  return sign | (uint8_t) ((d_hi <= d_lo) ? (lo + 1) : lo);
}

// ---- block quantization --------------------------------------------------------------------

// Quantize V[M][K] into per-row 32-element blocks: MX codes plus one E8M0 byte per (row, block).
//
// `codes` is [M][K], the layout `mvin` reads. `scales_a` is [K/32][M] -- the A-side scale window's
// own layout (`a_off = group * M + row`, gemmini.cc:1240), i.e. the TRANSPOSE of the [M][K/32] the
// requantizer writes. Writing it transposed here is the whole of the chain seam's scale work
// (chain_seam_hw_notes.md section 2) and costs one store per block.
static void mx_quantize_rows(const float *V, int M, int K, uint8_t *codes, uint8_t *scales_a) {
  mx_host_init();
  const int GK = K / MX_BLOCK;
  for (int m = 0; m < M; m++) {
    for (int g = 0; g < GK; g++) {
      const float *blk = V + (size_t) m * K + (size_t) g * MX_BLOCK;
      float amax = 0.0f;
      for (int i = 0; i < MX_BLOCK; i++) {
        float a = fabsf(blk[i]);
        if (a > amax) amax = a;
      }
      if (!(amax >= 1.1920929e-7f)) amax = 1.1920929e-7f;           // FLT_EPSILON clamp
      int s = mx_floor_log2(amax) + MX_E8M0_BIAS;
      if (s < 0) s = 0;
      if (s > 254) s = 254;
      scales_a[(size_t) g * M + m] = (uint8_t) s;
      float inv = mx_pow2(-(s - MX_E8M0_BIAS));
      uint8_t *dst = codes + (size_t) m * K + (size_t) g * MX_BLOCK;
      for (int i = 0; i < MX_BLOCK; i++) dst[i] = mx_e4m3_encode(blk[i] * inv);
    }
  }
}

// Quantize V[R][C] into 32-element blocks along its ROWS -- the B-side operand layout, one E8M0
// byte per (row-group, column), `b_off = group * C + col` (gemmini.cc:1241). `codes` stays [R][C],
// which is what mvin reads.
static void mx_quantize_cols(const float *V, int R, int C, uint8_t *codes, uint8_t *scales_b) {
  mx_host_init();
  for (int g = 0; g < R / MX_BLOCK; g++) {
    for (int c = 0; c < C; c++) {
      float amax = 0.0f;
      for (int i = 0; i < MX_BLOCK; i++) {
        float a = fabsf(V[(size_t) (g * MX_BLOCK + i) * C + c]);
        if (a > amax) amax = a;
      }
      if (!(amax >= 1.1920929e-7f)) amax = 1.1920929e-7f;
      int s = mx_floor_log2(amax) + MX_E8M0_BIAS;
      if (s < 0) s = 0;
      if (s > 254) s = 254;
      scales_b[(size_t) g * C + c] = (uint8_t) s;
      float inv = mx_pow2(-(s - MX_E8M0_BIAS));
      for (int i = 0; i < MX_BLOCK; i++) {
        size_t o = (size_t) (g * MX_BLOCK + i) * C + c;
        codes[o] = mx_e4m3_encode(V[o] * inv);
      }
    }
  }
}

static void mx_transpose_f32(const float *src, int R, int C, float *dst) {
  for (int r = 0; r < R; r++)
    for (int c = 0; c < C; c++) dst[(size_t) c * R + r] = src[(size_t) r * C + c];
}

// ---- llama glue, all fp32 --------------------------------------------------------------------

// out = h / sqrt(mean(h^2) + eps) * w, over the FULL hidden size -- so this is exact, not sliced.
static void mx_rmsnorm(const uint16_t *h_bf16, const uint16_t *w_bf16, int M, int D, float eps,
                       float *out) {
  for (int m = 0; m < M; m++) {
    const uint16_t *row = h_bf16 + (size_t) m * D;
    // Accumulated in double: this is a 2048-long reduction, and numpy's mean (which the golden
    // uses) sums pairwise, so a sequential fp32 sum would drift from it for no reason.
    double ss = 0.0;
    for (int d = 0; d < D; d++) { float v = mx_bf16_to_f32(row[d]); ss += (double) v * v; }
    float inv = 1.0f / sqrtf((float) (ss / (double) D) + eps);
    for (int d = 0; d < D; d++)
      out[(size_t) m * D + d] = mx_bf16_to_f32(row[d]) * inv * mx_bf16_to_f32(w_bf16[d]);
  }
}

static inline float mx_silu(float x) { return x / (1.0f + expf(-x)); }

// h = silu(G) * U, reading the two BF16 tiles the mesh produced.
static void mx_swiglu(const uint16_t *g_bf16, const uint16_t *u_bf16, int n, float *out) {
  for (int i = 0; i < n; i++)
    out[i] = mx_silu(mx_bf16_to_f32(g_bf16[i])) * mx_bf16_to_f32(u_bf16[i]);
}

// HF llama rotary, on a BF16 tile the mesh produced: out = x*cos + rotate_half(x)*sin, where
// rotate_half swaps the halves of the head dim and negates the upper one. cos/sin are the model's
// own tables, baked as fp32 bit patterns.
static void mx_rope(const uint16_t *x_bf16, const uint32_t *cos_f32, const uint32_t *sin_f32,
                    int M, int H, float *out) {
  const int half = H / 2;
  union { uint32_t u; float f; } c;
  for (int m = 0; m < M; m++) {
    for (int i = 0; i < H; i++) {
      size_t o = (size_t) m * H + i;
      float x = mx_bf16_to_f32(x_bf16[o]);
      float rot = (i < half) ? -mx_bf16_to_f32(x_bf16[o + half])
                             :  mx_bf16_to_f32(x_bf16[o - half]);
      c.u = cos_f32[o]; float co = c.f;
      c.u = sin_f32[o]; float si = c.f;
      out[o] = x * co + rot * si;
    }
  }
}

// Row softmax of the score tile with a causal mask: token m attends to 0..m only. The max is
// subtracted per row before exp, as every reference implementation does -- without it the tile
// would be arithmetically the same and numerically not.
static void mx_softmax_causal(const uint16_t *s_bf16, int M, float scale, float *out) {
  for (int m = 0; m < M; m++) {
    const uint16_t *row = s_bf16 + (size_t) m * M;
    float *dst = out + (size_t) m * M;
    float mx = -INFINITY;
    for (int j = 0; j <= m; j++) {
      float v = mx_bf16_to_f32(row[j]) * scale;
      dst[j] = v;
      if (v > mx) mx = v;
    }
    float sum = 0.0f;
    for (int j = 0; j <= m; j++) { dst[j] = expf(dst[j] - mx); sum += dst[j]; }
    float inv = 1.0f / sum;
    for (int j = 0; j <= m; j++) dst[j] *= inv;
    for (int j = m + 1; j < M; j++) dst[j] = 0.0f;
  }
}

// ---- reporting --------------------------------------------------------------------------------

// Relative Frobenius error of a BF16 tile against a BF16 reference: ||a - b|| / ||b||.
static float mx_rel_fro_bf16(const uint16_t *a, const uint16_t *b, int n) {
  float num = 0.0f, den = 0.0f;
  for (int i = 0; i < n; i++) {
    float x = mx_bf16_to_f32(a[i]), y = mx_bf16_to_f32(b[i]);
    num += (x - y) * (x - y);
    den += y * y;
  }
  return (den > 0.0f) ? sqrtf(num) / sqrtf(den) : 0.0f;
}

static int mx_count_diff_u8(const uint8_t *a, const uint8_t *b, int n) {
  int d = 0;
  for (int i = 0; i < n; i++) if (a[i] != b[i]) d++;
  return d;
}

static int mx_count_diff_u16(const uint16_t *a, const uint16_t *b, int n) {
  int d = 0;
  for (int i = 0; i < n; i++) if (a[i] != b[i]) d++;
  return d;
}

// printf has no %f on this baremetal path, so print a float as a scaled integer instead.
#define MX_PPM(x) ((int) ((x) * 1000000.0f))

#endif // INCLUDE_MX_HOST_H
