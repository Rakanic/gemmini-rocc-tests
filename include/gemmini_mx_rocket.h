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

#endif // GEMMINI_MX_ROCKET_H
