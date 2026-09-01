// Standalone (MxGemminiRocketConfig) MX helpers. Active only under -DMX_ROCKET.
// In this config the gemmini is a real RoCC (no MMIO command mimic), there is no shared
// memory (outputs come back via gemmini_mvout from the scratchpad), and scale factors are
// written to a flat RAM window (mx_scale_mgr_node) exactly like radiance's shared-mem path,
// just at a different base. Base/size come from the gemmini scale_mem config.
#ifndef GEMMINI_MX_ROCKET_H
#define GEMMINI_MX_ROCKET_H

// Flat scale-factor RAM window: weight (B) scales at +0x0000, activation (A) at +0x2000.
#define MX_SCALE_BASE 0x20000000UL
#define MX_SCALE_W (MX_SCALE_BASE + 0x0000)   // weight (B) scales
#define MX_SCALE_A (MX_SCALE_BASE + 0x2000)   // activation (A) scales

// FP6 LUT regmap (go-triggered, whole-table). Each table = 64 entries x 96b = 96 u64 words,
// then a write to the GO field commits it. Ports: 0=weight(B), 1=activation(A), 2=output(C).
#include <stdint.h>
#define MX_LUT_BASE 0x20010000UL
#define MX_LUT0 (MX_LUT_BASE + 0x100)   // weight
#define MX_LUT1 (MX_LUT_BASE + 0x500)   // activation-in
#define MX_LUT2 (MX_LUT_BASE + 0x900)   // activation-out
#define MX_LUT_GO_OFF 0x300

static inline void mx_load_lut(unsigned long base, const uint32_t *lut /*[64][3]*/) {
  volatile uint64_t *w   = (volatile uint64_t *) base;
  const    uint64_t *src = (const uint64_t *) lut;
  for (int i = 0; i < 96; i++) w[i] = src[i];
  *((volatile uint32_t *) (base + MX_LUT_GO_OFF)) = 1;
}

#endif // GEMMINI_MX_ROCKET_H
