#include <stdio.h>
#include <stdlib.h>
#include "kernel.h"

extern "C" {
#include "tab.h"
#include "print_measure.h"
#include "time_measure.h"
}

#define NB_META 31

#ifdef GPUMM_HANDLE_ENABLE
GPUMM_BLAS_HANDLE handle;
#endif

int main(int argc, char* argv[])
{
    unsigned int n, nwu, nrep;
    if (argc != 4) 
    {
        fprintf (stderr, "Usage: %s <problem size> <nb warmup> <nb rep>\n", argv[0]);
        return 1;
    }
    else
    {
        n = atoi(argv[1]);
        nwu = atoi(argv[2]);
        nrep = atoi(argv[3]);
    }

    uint64_t tdiff[NB_META];
    srand(0);

    int size = n * n * sizeof(double);
    
    double *b = (double*)malloc(size);
    double *c = (double*)malloc(size);

    init_tab2d_random(n, &b);
    init_tab2d_random(n, &c);

    double* d_b;
    double* d_c;

	GPUMM_ALLOC(d_b, size);
	GPUMM_ALLOC(d_c, size);

    GPUMM_MEMCPY_HtD(d_b, b, size);
    GPUMM_MEMCPY_HtD(d_c, c, size);

    #ifdef GPUMM_HANDLE_ENABLE
    GPUMM_HANDLE_CREATE(handle);
    #endif

    for (unsigned int m = 0; m < NB_META; m++)
    {
        double *a = (double*)malloc(size);
        double* d_a;
        GPUMM_ALLOC(d_a, size);

        if ( m == 0 )
        {
            for (unsigned int i = 0; i < nwu; i++)
            {
                kernel(n, d_a, d_b, d_c);
            }
            GPUMM_DEVICE_SYNC();
        }
        else
        {
            kernel(n, d_a, d_b, d_c);
            GPUMM_DEVICE_SYNC();
        }

        const uint64_t t1 = measure_clock();
        for (unsigned int i = 0; i < nrep; i++)
        {
            kernel(n, d_a, d_b, d_c);
        }
        GPUMM_DEVICE_SYNC();
        const uint64_t t2 = measure_clock();

        tdiff[m] = t2 - t1;
        
        GPUMM_MEMCPY_DtH(a, d_a, size);
        GPUMM_FREE(d_a);
        free(a);
    }
    
    #ifdef GPUMM_HANDLE_ENABLE
    GPUMM_HANDLE_DESTROY(handle);
    #endif     

    GPUMM_FREE(d_b);
    GPUMM_FREE(d_c);

    free(b);
    free(c);

    print_measure(n, nrep, tdiff);
    
    return EXIT_SUCCESS;
}