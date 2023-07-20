#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include "time_measure.h"
#ifdef MS
#include<sys/time.h>
uint64_t measure_clock()
{
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return (((uint64_t)tv.tv_sec)*1000)+((uint64_t)tv.tv_usec/1000);    
}
#else
extern uint64_t rdtsc ();
uint64_t measure_clock()
{
    return rdtsc();
}
#endif