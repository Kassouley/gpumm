#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include "time_measure.h"
#include "print_measure.h"
#ifndef NB_META
#define NB_META 31
#endif

static int cmp_uint64 (const void *a, const void *b)
{
    const uint64_t va = *((uint64_t *) a);
    const uint64_t vb = *((uint64_t *) b);

    if (va < vb) return -1;
    if (va > vb) return 1;
    return 0;
}

void print_measure(int n, unsigned int nrep, uint64_t tdiff[NB_META])
{
    FILE * output = NULL;
    
    const unsigned long nbitr = (unsigned long)n*(unsigned long)n*(unsigned long)n*(unsigned long)nrep;

    qsort (tdiff, NB_META, sizeof tdiff[0], cmp_uint64);
    printf("Minimum : %.6g %s (%.3g par itération)\n", (float)tdiff[0]/(float)nrep        , MEASURE_UNITE, (float)tdiff[0]/(float)nbitr);
    printf("Median  : %.6g %s (%.3g par itération)\n", (float)tdiff[NB_META/2]/(float)nrep, MEASURE_UNITE, (float)tdiff[NB_META/2]/(float)nbitr);
    printf("Maximum : %.6g %s (%.3g par itération)\n", (float)tdiff[NB_META-1]/(float)nrep, MEASURE_UNITE, (float)tdiff[NB_META-1]/(float)nbitr);
    
    const float stabilite = (tdiff[NB_META/2] - tdiff[0]) * 100.0f / tdiff[0];
    
    if (stabilite >= 10)
        printf("Bad Stability : %.2f %%\n", stabilite);
    else if ( stabilite >= 5 )
        printf("Average Stability : %.2f %%\n", stabilite);
    else
        printf("Good Stability : %.2f %%\n", stabilite);

    output = fopen("./output/measure_tmp.out", "a");
    if (output != NULL) 
    {
        fprintf(output, " | %15.4f | %14.3f | %13f | %12f\n", (float)tdiff[0]/nrep, (float)tdiff[NB_META/2]/nrep, (float)tdiff[NB_META/2]/nbitr, stabilite);
        fclose(output);
    }
    else
    {
        char cwd[1028];
        if (getcwd(cwd, sizeof(cwd)) != NULL) 
        {
            printf("Couldn't open '%s/output/measure_tmp.out' file\n Measure not saved\n", cwd);
        }
    }
}