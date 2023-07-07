#include <stdio.h>
#include <omp.h>
#include "kernel.h"
#include "device_omp.h"

#ifdef GPU_OMP
void kernel (unsigned int n, double* a, const double* b, const double* c)
{
    #pragma omp target map(to: b[0:n*n], c[0:n*n]) map(tofrom: a[0:n*n])
    {
        #pragma omp teams distribute parallel for simd collapse(2) 
        for(unsigned int i = 0; i < n; i++)
        {
            for(unsigned int j = 0; j < n; j++)
            {
                a[i*n+j] = 0;
                for(unsigned int k = 0; k < n; k++)
                {
                    a[i*n+j] += b[i*n+k] * c[k*n+j];
                }
            }
        }
    }
}
#endif

#ifdef GPU_OMP_WO_DT
void kernel (unsigned int n, double* a, const double* b, const double* c)
{
    #pragma omp target is_device_ptr(a,b,c)
    {
        #pragma omp teams distribute parallel for collapse(2) 
        for(unsigned int i = 0; i < n; i++)
        {
            for(unsigned int j = 0; j < n; j++)
            {
                a[i*n+j] = 0;
                for(unsigned int k = 0; k < n; k++)
                {
                    a[i*n+j] += b[i*n+k] * c[k*n+j];
                }
            }
        }
    }
}
#endif