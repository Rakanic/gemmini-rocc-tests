#include <stdio.h>

#include <stdint.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdlib.h>

#define SCALE_FACT_MEM 0x40088000

volatile unsigned long sink;
int main(void) {
  (*(volatile uint64_t *)(SCALE_FACT_MEM)) = 0xdeadbeefdeadbeefULL;
  (*(volatile uint64_t *)(SCALE_FACT_MEM + 8)) = 0xdeadbeefdeadbeefULL;
  (*(volatile uint64_t *)(SCALE_FACT_MEM + 16)) = 0xdeadbeefdeadbeefULL;
  (*(volatile uint64_t *)(SCALE_FACT_MEM + 24)) = 0xdeadbeefdeadbeefULL;
  (*(volatile uint64_t *)(SCALE_FACT_MEM + 0)) = 0x1122334411223344ULL;
  (*(volatile uint64_t *)(SCALE_FACT_MEM + 8)) = 0x1122334411223344ULL;
  (*(volatile uint64_t *)(SCALE_FACT_MEM + 16)) = 0x1122334411223344ULL;
  (*(volatile uint64_t *)(SCALE_FACT_MEM + 24)) = 0x1122334411223344ULL;
  (*(volatile uint64_t *)(SCALE_FACT_MEM + 32)) = 0x5566778855667788ULL;
  (*(volatile uint64_t *)(SCALE_FACT_MEM + 40)) = 0x5566778855667788ULL;
  (*(volatile uint64_t *)(SCALE_FACT_MEM + 48)) = 0x5566778855667788ULL;
  (*(volatile uint64_t *)(SCALE_FACT_MEM + 56)) = 0x5566778855667788ULL;
  (*(volatile uint64_t *)(SCALE_FACT_MEM + 32)) = 0xDEADBEEFDEADBEEFULL;
  (*(volatile uint64_t *)(SCALE_FACT_MEM + 40)) = 0xDEADBEEFDEADBEEFULL;
  (*(volatile uint64_t *)(SCALE_FACT_MEM + 48)) = 0xDEADBEEFDEADBEEFULL;
  (*(volatile uint64_t *)(SCALE_FACT_MEM + 56)) = 0xDEADBEEFDEADBEEFULL;

  (*(volatile uint64_t *)(SCALE_FACT_MEM + 8192)) = 0xdeadbeefdeadbeefULL;
  (*(volatile uint64_t *)(SCALE_FACT_MEM + 8192 + 8)) = 0xdeadbeefdeadbeefULL;
  (*(volatile uint64_t *)(SCALE_FACT_MEM + 8192 + 16)) = 0xdeadbeefdeadbeefULL;
  (*(volatile uint64_t *)(SCALE_FACT_MEM + 8192 + 24)) = 0xdeadbeefdeadbeefULL;
  (*(volatile uint64_t *)(SCALE_FACT_MEM + 8192 + 32)) = 0x1122334411223344ULL;
  (*(volatile uint64_t *)(SCALE_FACT_MEM + 8192 + 40)) = 0x5566778855667788ULL;
  (*(volatile uint64_t *)(SCALE_FACT_MEM + 8192 + 48)) = 0x1122334411223344ULL;
  (*(volatile uint64_t *)(SCALE_FACT_MEM + 8192 + 56)) = 0xDEADBEEFDEADBEEFULL;
  (*(volatile uint64_t *)(SCALE_FACT_MEM + 8192 + 64)) = 0x1122334411223344ULL;
  (*(volatile uint64_t *)(SCALE_FACT_MEM + 8192 + 72)) = 0x5566778855667788ULL;
  (*(volatile uint64_t *)(SCALE_FACT_MEM + 8192 + 80)) = 0xDEADBEEFDEADBEEFULL;
  (*(volatile uint64_t *)(SCALE_FACT_MEM + 8192 + 88)) = 0x5566778855667788ULL;
  
  exit(0);
}
