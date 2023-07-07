#include <stdio.h>
#include "kernel.h"

#ifdef BASIS
void kernel (unsigned int n, double* a, const double* b, const double* c)
{
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
#endif

#ifdef CPU_OMP
#include <omp.h>
void kernel (unsigned int n, double* a, const double* b, const double* c)
{
    #pragma omp parallel for schedule(dynamic)
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
#endif

#ifdef CBLAS
#include <cblas.h>
void kernel (unsigned int n, double* a, const double* b, const double* c)
{
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, b, n, c, n, 0.0, a, n);
}
#endif
