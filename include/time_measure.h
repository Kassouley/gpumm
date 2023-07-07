#ifndef MEASURE_TIME_H
#define MEASURE_TIME_H
#include <stdint.h>
#ifdef MS
#define MEASURE_UNITE "ms"
#else
#define MEASURE_UNITE "RDTSC-cycles"
#endif
uint64_t measure_clock();
#endif