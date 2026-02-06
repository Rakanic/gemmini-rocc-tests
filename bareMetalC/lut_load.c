#include <stdio.h>

#include <stdint.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdlib.h>


#define GEMMINI_CTRL 0x40084000
#define GEMMINI_INST_ADDR (GEMMINI_CTRL + 0x0)
#define GEMMINI_RS1_ADDR (GEMMINI_CTRL + 0x10)
#define GEMMINI_RS2_ADDR (GEMMINI_CTRL + 0x18)

#define GEMMINI_LUT0_ADDR (GEMMINI_CTRL + 0x80)
#define GEMMINI_LUT1_ADDR (GEMMINI_CTRL + 0x200)
#define GEMMINI_LUT2_ADDR (GEMMINI_CTRL + 0x380)



volatile unsigned long sink;
int main(void) {
  (*(volatile uint32_t *)(GEMMINI_LUT0_ADDR)) = 0x11223344;
  (*(volatile uint32_t *)(GEMMINI_LUT0_ADDR + 4)) = 0x55667788;
  (*(volatile uint32_t *)(GEMMINI_LUT0_ADDR + 8)) = 0xDEADBEEF;
  (*(volatile uint32_t *)(GEMMINI_LUT0_ADDR)) = 0xDEADBEEF;
  (*(volatile uint32_t *)(GEMMINI_LUT0_ADDR + 4)) = 0xDEADBEEF;
  (*(volatile uint32_t *)(GEMMINI_LUT0_ADDR + 8)) = 0xDEADBEEF;                                                                                                                            (*(volatile uint32_t *)(GEMMINI_LUT1_ADDR)) = 0x11223344;
  (*(volatile uint32_t *)(GEMMINI_LUT1_ADDR + 4)) = 0x55667788;
  (*(volatile uint32_t *)(GEMMINI_LUT1_ADDR + 8)) = 0xDEADBEEF;
   (*(volatile uint32_t *)(GEMMINI_LUT1_ADDR)) = 0xDEADBEEF;
  (*(volatile uint32_t *)(GEMMINI_LUT1_ADDR + 4)) = 0xDEADBEEF;
  (*(volatile uint32_t *)(GEMMINI_LUT1_ADDR + 8)) = 0xDEADBEEF;

  (*(volatile uint32_t *)(GEMMINI_LUT2_ADDR)) = 0x11223344;
  (*(volatile uint32_t *)(GEMMINI_LUT2_ADDR + 4)) = 0x55667788;
  (*(volatile uint32_t *)(GEMMINI_LUT2_ADDR + 8)) = 0xDEADBEEF;
   (*(volatile uint32_t *)(GEMMINI_LUT2_ADDR)) = 0xDEADBEEF;
  (*(volatile uint32_t *)(GEMMINI_LUT2_ADDR + 4)) = 0xDEADBEEF;
  (*(volatile uint32_t *)(GEMMINI_LUT2_ADDR + 8)) = 0xDEADBEEF;

  exit(0);
}