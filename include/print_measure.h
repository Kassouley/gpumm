#ifndef PRINT_MEASURE_H
#define PRINT_MEASURE_H
#ifndef NB_META
#define NB_META 31
#endif
#include <stdint.h>
void print_measure(int n, unsigned int nrep, uint64_t tdiff[NB_META]);
static int cmp_uint64 (const void *a, const void *b);
#endif