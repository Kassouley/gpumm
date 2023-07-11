#include <stdint.h>

#ifdef __i386
uint64_t rdtsc() {
   uint64_t x;
   __asm__ volatile ("rdtsc" : "=A" (x));
   return x;
}
#elif defined __amd64
uint64_t rdtsc() {
   uint64_t a, d;
   __asm__ volatile ("rdtsc" : "=a" (a), "=d" (d));
   return (d<<32) | a;
}
#elif defined __aarch64__ 
uint64_t rdtsc() {
   uint64_t val;
   __asm__ volatile("mrs %0, cntvct_el0" : "=r" (val));
   return val;
}
#endif

// uint64_t pmccntr;
// uint64_t pmuseren;
// uint64_t pmcntenset;
// uint64_t tmp = 1;
// __asm__ volatile("mrs %0, pmuserenr_el0" : "=r"(pmuseren));
// printf("test\n");
// pmuseren=pmuseren|1;
// __asm__ volatile("msr pmuserenr_el0, %0" : : "r"(pmuseren));
// printf("test\n");
// if (pmuseren & 1) {
//    __asm__ volatile("mrs %0, pmcntenset_el0" : "=r"(pmcntenset));
// printf("test\n");
//    __asm__ volatile("msr pmcntenset_el0, %0" : : "r"(pmcntenset|(tmp<<32)));
// printf("test\n");
//    if (pmcntenset & 0x80000000ul) {
// printf("test\n");
//       __asm__ volatile("mrs %0, pmccntr_el0" : "=r"(pmccntr));
//       return (uint64_t)pmccntr * 64;
//    }
// }