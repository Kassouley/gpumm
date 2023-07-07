#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include <string.h>
#include "kernel.h"
#include "tab.h"

#define NB_META 31
#define OUTOUT_FILE "../output/output_calibrate.txt"

extern uint64_t rdtsc ();

static int cmp_uint64 (const void *a, const void *b)
{
    const uint64_t va = *((uint64_t *) a);
    const uint64_t vb = *((uint64_t *) b);

    if (va < vb) return -1;
    if (va > vb) return 1;
    return 0;
}

int main(int argc, char **argv)
{
    unsigned int n, repm;
    if (argc != 3) 
    {
        fprintf (stderr, "Usage: %s <problem size> <nb repeat>\n", argv[0]);
        return 1;
    }
    else
    {
        n = atoi(argv[1]);
        repm = atoi(argv[2]);
    }

    uint64_t **tdiff = malloc( repm * sizeof(tdiff[0][0]));
    for(unsigned int k = 0 ; k < repm ; k++)
    {
        tdiff[k] = malloc( NB_META * sizeof(tdiff[0]));
    }

    
    srand(0);
    float (*b)[n] = malloc(n * n * sizeof(b[0][0]));
    float (*c)[n] = malloc(n * n * sizeof(c[0][0]));
    init_tab2d_random(n, b);
    init_tab2d_random(n, c);
    
    for (unsigned int m = 0; m < NB_META; m++)
    {
        float (*a)[n] = malloc(n * n * sizeof(a[0][0]));

        for (unsigned int k = 0; k < repm; k++)
        {
            const uint64_t t1 = rdtsc();
            kernel(n, a, b, c);
            const uint64_t t2 = rdtsc();
            tdiff[k][m] = t2 - t1;
        }
        
        free(a);
        sleep(3);
    }
    free(b);
    free(c);

    double* average = calloc(repm , sizeof(average[0]));
    uint64_t* repm_min = malloc(repm * sizeof(repm_min[0]));
    uint64_t* repm_median = malloc(repm * sizeof(repm_median[0]));
    double* deviation = malloc(repm * sizeof(deviation[0]));

    for(unsigned int k = 0 ; k < repm ; k++)
    {
        for(unsigned int m = 0 ; m < NB_META ; m++)
        {
            average[k] += tdiff[k][m];
        }
        average[k] /= NB_META;
    }

    for(unsigned int k = 0 ; k < repm ; k++)
    {
        deviation[k] = 0;
        for(unsigned int m = 0 ; m < NB_META ; m++)
        {
            deviation[k] += pow(average[k] - tdiff[k][m], 2);
        }
        deviation[k] /= NB_META;
        deviation[k] = sqrt(deviation[k]);
    }


    for(unsigned int k = 0 ; k < repm ; k++)
    {
        qsort( tdiff[k], NB_META, sizeof tdiff[0][0], cmp_uint64);
        repm_min[k] = tdiff[k][0];
        repm_median[k] = tdiff[k][NB_META/2];
    }

    
    FILE * output = fopen(OUTOUT_FILE, "w+");
    printf("Time per iteration (RDTSC-cycles) :\n");
    printf("%5s %14s %11s %11s %11s %11s\n", "Step", "Average", "Minimum", "Median", "Stability","Deviation");
    for(unsigned int k = 0 ; k <  repm ; k++)
    {
        const double stabilite = (repm_median[k] - repm_min[k]) * 100.0f / repm_min[k];
        printf("%5d %14.2lf %11lu %11lu %9.2f%% %9.2f%%\n", k, average[k], repm_min[k], repm_median[k], stabilite, deviation[k]/average[k]*100.0f);
        fprintf(output, "%d %lf %lu %lu %f %f\n", k, average[k], repm_min[k], repm_median[k], stabilite, deviation[k]/average[k]*100.0f);
    }
    fclose(output);

    qsort(average, repm, sizeof average[0], cmp_uint64);
    printf("-----------------------------------------------------\n");
    printf("\033[1m%s %3s \033[42m%10.2lf\033[0m\n",
        "Average minimum time per iteration :", "", average[0]);

    printf("\033[1m%s %3s \033[42m%10.2lf\033[0m\n",
        "Average median time per iteration  :", "", average[(int)(repm/2)]);
    
    printf("\033[1m%s %3s \033[42m%10.2lf\033[0m\n",

        "Average maximum time per iteration :", "", average[repm-1]);
    printf("-----------------------------------------------------\n");

    for(unsigned int k = 0 ; k < repm ; k++)
    {
        free(tdiff[k]);
    }
    free(tdiff);
    free(average);
    free(repm_min);
    free(repm_median);
    return EXIT_SUCCESS;
}


